import cv2
import numpy as np
import os
from utils import get_case_defects, calc_alignment_error, \
    align_images, to_gray, overlay_region_masks, quickshow

from analyses.contours import get_smoothened_contours, contours_to_masks
from functools import reduce
from stats import calc_stats_single_ref_img


def local_defect_analysis(img, mask, mean, std_dev, thresh_type, neighbors=4):
    """
    Detect localized defects by identifying pixels that deviate significantly from the specified mean intensity.

    Parameters:
    - img (np.array): Grayscale image in which to identify defects.
    - mask (np.array): Binary mask to define the region of interest within `img`.
    - mean (float): Mean intensity value for the reference area.
    - std_dev (float): Standard deviation of intensity for the reference area.
    - thresh_type (str): Threshold direction; either 'lower' (for detecting higher intensities)
                         or 'upper' (for detecting lower intensities).

    Returns:
    - def_mask (np.array): Binary mask highlighting regions marked as defects.

    Details:
    - A main threshold is calculated based on the `mean` and `std_dev` to identify outlying pixels in the image:
        - For `lower`, outliers are pixels with intensities significantly higher than the mean.
        - For `upper`, outliers are pixels with intensities significantly lower than the mean.
    - Defects are identified by checking if a given outlying pixel has a local neighborhood containing
      at least four other outlier pixels, indicating a potential defect cluster.
    """

    h, w = img.shape

    # Define threshold and detect outlying pixels
    if thresh_type == 'lower':
        main_threshold = mean + 1.3 * std_dev
        _, outliers = cv2.threshold(img, main_threshold, 255, cv2.THRESH_BINARY)
    elif thresh_type == 'upper':
        main_threshold = mean - 1.3 * std_dev
        _, outliers = cv2.threshold(img, main_threshold, 255, cv2.THRESH_BINARY_INV)

    # Restrict analysis to relevant regions defined by the mask
    outliers = cv2.bitwise_and(outliers, outliers, mask=mask)

    # Initialize an empty mask to store defect locations
    def_mask = np.zeros_like(img, dtype=np.uint8)

    # Analyze each outlying pixel in the region of interest
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if outliers[i, j]:  # Pixel is an outlier
                # Check the 5x5 neighborhood around the pixel
                neighborhood = img[max(i - 1, 0):min(i + 2, h), max(j - 1, 0):min(j + 2, w)]

                # Mark as defect if at least 4 other neighboring pixels are also outliers
                if np.sum(neighborhood > main_threshold) >= neighbors:
                    def_mask[i, j] = 255

    return def_mask


def propose_defects_rov(insp, ref, masks, visualize=False):
    """
    Detect defects in an inspected image (insp) using Range of Values (ROV) analysis
    based on thresholding within specified inner and outer regions. This method is meant to catch the small
    detailed defects that don't necessarily have high volume, but is limited only to the dark and bright regions,
    and would function badly close to the edges.

    Parameters:
    - insp (np.array): Inspected image to analyze for defects.
    - ref (np.array): Reference image used for statistical comparisons.
    - masks (dict): Dictionary of masks indicating specific regions:
        - 'mask_in_insp_relevant': Relevant inner region mask in the inspected image.
        - 'mask_out_insp_relevant': Relevant outer region mask in the inspected image.
        - 'mask_edges_ref': Mask indicating edge regions in the reference image.
    - visualize (bool): If True, displays intermediate processing steps for debugging.

    Returns:
    - proposed_defects_rov (np.array): Binary mask indicating regions of proposed defects.

    Steps:
    1. Preprocess the inspected image with Gaussian blur to reduce noise.
    2. Calculate mean and standard deviation statistics for the reference image's regions.
    3. Create masks to exclude edge pixels from analysis for more accurate comparisons.
    4. Analyze inner and outer regions for potential defects based on thresholding with the regions' statistics.
    """

    # Convert inspected image to grayscale and apply slight Gaussian blur to reduce noise
    insp_gr = to_gray(insp)
    insp_blr = cv2.GaussianBlur(insp_gr, (5, 5), sigmaX=2)  # sigma can be adjusted for best results
    insp_dil = cv2.dilate(insp_blr, np.ones([3,3]), iterations=2)

    # Calculate mean and standard deviation for the inner and outer regions of the reference image
    stats_in, stats_out = calc_stats_single_ref_img(ref)

    # Get the  region masks for the relevant pixels in the inspected image
    mask_in_insp_relevant = masks['mask_in_insp_relevant']
    mask_out_insp_relevant = masks['mask_out_insp_relevant']

    # Get the
    sm_cont_insp = get_smoothened_contours(insp)
    mask_in_insp, mask_edges_insp, mask_out_insp = contours_to_masks(sm_cont_insp, insp.shape[0:2], thickness=24)

    # Combine the edge mask with relevant region masks for background exclusion  # todo remove?
    mask_edges = mask_edges_insp

    # Define the inner, non-edge region by excluding edge pixels from the inner mask
    mask_in_insp_non_edges = cv2.bitwise_and(cv2.bitwise_not(mask_edges), mask_in_insp_relevant)
    img_in_insp_non_edges = cv2.bitwise_and(insp_dil, insp_dil, mask=mask_in_insp_non_edges)

    # Define the outer, non-edge region by excluding edge pixels from the outer mask
    mask_out_insp_non_edges = cv2.bitwise_and(cv2.bitwise_not(mask_edges), mask_out_insp_relevant)
    img_out_insp_non_edges = cv2.bitwise_and(insp_dil, insp_dil, mask=mask_out_insp_non_edges)

    ###### PART A - Thresholding in the inner (bright) region
    # Upper threshold: Detect pixels significantly darker (3 sigma below mean) than the average of inner region
    prop_defs_in_low = local_defect_analysis(
        img_in_insp_non_edges, mask=mask_in_insp_relevant,
        mean=stats_in['mean'], std_dev=stats_in['std_dev'], thresh_type='upper'
    )

    # Lower threshold: Detect pixels significantly brighter (3 sigma above mean) than the average of inner region
    prop_defs_in_high = local_defect_analysis(
        img_in_insp_non_edges, mask=mask_in_insp_relevant,
        mean=stats_in['mean'], std_dev=stats_in['std_dev'], thresh_type='lower'
    )

    ###### PART B - Thresholding in the outer (dark) region
    # Lower threshold only: Detect brighter pixels in the outer (darker) region, indicating potential defects
    prop_defs_out_high = local_defect_analysis(
        img_out_insp_non_edges, mask=mask_out_insp_relevant,
        mean=stats_out['mean'], std_dev=stats_out['std_dev'], thresh_type='lower'
    )

    # Combine masks for inner and outer region defects
    blank_mask = np.zeros(insp.shape[0:2], dtype=np.uint8)
    proposed_defects_rov = reduce(cv2.bitwise_or, [prop_defs_in_low, prop_defs_in_high, prop_defs_out_high], blank_mask)


    # Optional: Morphological closing to smooth small gaps and unify detected regions
    proposed_defects_rov = cv2.morphologyEx(proposed_defects_rov, cv2.MORPH_OPEN, np.ones([3, 3]), iterations=2)

    # To negate the dilation from the beginning
    proposed_defects_rov = cv2.erode(proposed_defects_rov, np.ones([3, 3]), iterations=1)

    return proposed_defects_rov

