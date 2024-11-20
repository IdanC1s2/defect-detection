import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from utils import get_case_defects, calc_alignment_error,\
    align_images, to_gray, overlay_region_masks, quickshow


def contours_to_masks(contours, img_sz, thickness=1):
    """
    This function generates three masks based on the provided contours:
    - mask_in: Pixels inside the contours (bright regions).
    - mask_edges: Pixels that belong to the contour edges.
    - mask_out: Pixels outside the contours (dark regions).

    Parameters:
    - contours (list): A list of contours detected from the image.
    - img_sz (tuple): The size (height, width) of the image to generate the masks.
    - thickness (int): The thickness of the contour edges. Default is 1.

    Returns:
    - mask_in (np.array): Mask with the pixels inside the contours.
    - mask_edges (np.array): Mask with the contour edges.
    - mask_out (np.array): Mask with the pixels outside the contours.
    """
    # Initialize the edge mask (where contours will be drawn)
    mask_edges = np.zeros(img_sz, dtype=np.uint8)
    cv2.drawContours(mask_edges, contours, -1, color=255, thickness=thickness)

    # Initialize the mask for pixels inside the contours
    mask_in = np.zeros(img_sz, dtype=np.uint8)
    cv2.drawContours(mask_in, contours, -1, color=255, thickness=cv2.FILLED)

    # Remove the edge pixels from the inside mask
    mask_in = cv2.bitwise_and(cv2.bitwise_not(mask_edges), mask_in)

    # The mask for the outside of the contours is just the inverse of the inside mask
    mask_out = cv2.bitwise_not(mask_in)
    mask_out = cv2.bitwise_and(cv2.bitwise_not(mask_edges), mask_out)

    return mask_in, mask_edges, mask_out


def get_smoothened_contours(img, threshold_value=127, kernel_size=5, epsilon=1, visualize=False):
    """
    This function processes an image by denoising, blurring, thresholding, and performing morphological operations
    (erosion and dilation) to extract and smoothen the contours.

    Parameters:
    - img (np.array): The input image to process.
    - threshold_value (int): The threshold value for binary thresholding. Default is 127.
    - kernel_size (int): The size of the kernel for erosion and dilation. Default is 5.
    - epsilon (float): The approximation accuracy for the contours. Default is 1.
    - visualize (bool): If True, intermediate images will be shown for debugging/visualization. Default is False.

    Returns:
    - smoothed_contours (list): A list of smoothened contours approximated from the input image.
    """
    # Grayscaling the image
    img = to_gray(img)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    if visualize: quickshow(denoised)

    # Apply blur to the image
    blurred = cv2.blur(denoised, (5, 5))
    if visualize: quickshow(blurred)

    # Threshold the image using Otsu's method
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if visualize: quickshow(binary)

    # Apply morphological transformations: Erosion followed by Dilation
    kernel = np.ones([kernel_size, kernel_size], dtype=np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    if visualize: quickshow(erosion)

    dilation = cv2.dilate(erosion, kernel, iterations=1)
    if visualize: quickshow(dilation)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Smoothen contours by approximating each one  # Smoothing the contours appears to be counter productive
    # smoothed_contours = []
    # for contour in contours:
    #     approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    #     smoothed_contours.append(approx_contour)
    smoothed_contours = contours

    return smoothed_contours


#     masks = {}
#     masks['mask_in_ref'] = mask_in_ref   # Mask of the pixels within contours in the reference image
#     masks['mask_out_ref'] = mask_out_ref   # Mask of the pixels out of the contours in the reference image
#     masks['mask_in_insp'] = mask_in_insp   # Mask of the pixels within contours in the inspected image
#     masks['mask_out_insp'] = mask_out_insp   # Mask of the pixels out of the contours in the inspected image
#     masks['mask_ref_relevant'] = mask_ref_relevant  # Mask of the relevant pixels in the aligned reference image.
#     # It contains all the pixels in the aligned reference image that were not zeroed due to the warp transformation
#
#     masks['mask_in_insp_relevant'] = mask_in_insp_relevant  # Mask of the pixels *within* contours in the *inspected* image that
#     # are within the **relevant pixel range**. **The relevant pixel range** is determined by mask_ref_relevant.
#
#     masks['mask_out_insp_relevant'] = mask_out_insp_relevant  # Mask of the pixels *outside* of the contours in the *inspected* image that
#     # are within the relevant pixel range.
#
#     masks['mask_diff'] = diff  # Absolute difference between the masks, computed over the relevant pixels
#     masks['mask_edges_insp'] = mask_edges_insp
#     masks['mask_edges_ref'] = mask_edges_ref
#
#
#     return aligned_ref, masks


def align_using_contours(insp, ref, optimize_alignment=True, visualize=False):
    """
    Aligns the inspected image to the reference image using contours, computes the relevant masks,
    and optionally visualizes the alignment process.

    Parameters:
    - insp (np.array): The inspected image (color or grayscale).
    - ref (np.array): The reference image (color or grayscale).
    - visualize (bool, optional): If True, visualizes the alignment results. Default is False.

    Returns:
    - aligned_ref (np.array): The aligned reference image using the computed transformation.
    - masks (dict): A dictionary containing the various masks involved in the alignment process.
    """

    # Get smoothened contours and masks for inspected and reference images
    sm_cont_insp = get_smoothened_contours(insp)
    mask_in_insp, mask_edges_insp, mask_out_insp = contours_to_masks(sm_cont_insp, insp.shape[:2], thickness=1)

    sm_cont_ref = get_smoothened_contours(ref)
    mask_in_ref, mask_edges_ref, mask_out_ref = contours_to_masks(sm_cont_ref, ref.shape[:2], thickness=1)

    # Align the masks and obtain the relevant alignment information
    aligned_mask_ref, mask_ref_relevant, h_mat = \
        align_images(mask_in_insp, mask_in_ref, keepPercent=0.15,
                     optimize_keepPercetnt=optimize_alignment, visualize=visualize)

    # Apply mask refinement
    mask_in_insp_relevant = cv2.bitwise_and(mask_in_insp, mask_ref_relevant)
    mask_out_insp_relevant = cv2.bitwise_and(mask_out_insp, mask_ref_relevant)

    # Align the reference image based on the computed homography matrix
    aligned_ref = cv2.warpPerspective(ref, h_mat, (insp.shape[1], insp.shape[0]))

    # Compute the absolute difference between the aligned reference and the inspected mask
    diff = cv2.absdiff(aligned_mask_ref, mask_in_insp_relevant)

    # Visualize the results if requested
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))

        axes[0, 0].imshow(mask_in_insp)
        axes[0, 0].set_title('Inspected Mask')

        axes[1, 0].imshow(mask_in_ref)
        axes[1, 0].set_title('Reference Mask')

        axes[0, 1].imshow(aligned_mask_ref)
        axes[0, 1].set_title('Aligned Reference Mask')

        axes[1, 1].imshow(diff)
        axes[1, 1].set_title('Difference Mask')


    # Store all relevant masks in a dictionary for future use.
    # Here, 'in', 'out' or 'edges' refers to the different regions in the image. (bright, dark and edges respectively)
    # 'relevant' denotes the relevant pixels - those in places that were not zeroed by the warp transformation.
    masks = {
        'mask_in_ref': mask_in_ref,
        'mask_out_ref': mask_out_ref,
        'mask_in_insp': mask_in_insp,
        'mask_out_insp': mask_out_insp,
        'mask_ref_relevant': mask_ref_relevant,
        'mask_in_insp_relevant': mask_in_insp_relevant,
        'mask_out_insp_relevant': mask_out_insp_relevant,
        'mask_diff': diff,
        'mask_edges_insp': mask_edges_insp,
        'mask_edges_ref': mask_edges_ref
    }

    return aligned_ref, masks

def propose_defects_contours_subtraction(masks, visualize=False):
    " Propose defects based on overlapping contours have misaligned edges"

    diff = masks['mask_diff']
    ker = np.ones([3,3], dtype=np.uint8)
    eroded_diff = cv2.erode(diff, ker, iterations=1)
    dilated_diff = cv2.dilate(eroded_diff,ker,iterations=2)
    _, thresh_diff = cv2.threshold(dilated_diff, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        plt.suptitle('Contour Difference Through Processing\nDiff=|Inspected_mask - Aligned_Reference_mask|')
        axes[0].imshow(diff)
        axes[0].set_title('Diff')

        axes[1].imshow(eroded_diff)
        axes[1].set_title('Eroded Diff')

        axes[2].imshow(dilated_diff)
        axes[2].set_title('Eroded Then Dilated Diff')

        axes[3].imshow(thresh_diff)
        axes[3].set_title('Eroded, Dilated then Thresholded Diff')

        # fig.savefig(os.path.join(path_res, f"misalignment_proposed_pixels_case{case}.png"))

    mask_proposed = thresh_diff
    return mask_proposed








