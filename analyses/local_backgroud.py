import numpy as np
import cv2
from utils import align_images, quickshow, to_gray
from functools import reduce


def propose_defects_local_background(insp, ref, intensity_diff_thresh=0.1, visualize=False):
    """
    Detects defects in an inspected image by comparing its regions to the local background intensity in an aligned reference image.

    This function aligns the inspected image with a reference image, then calculates a difference image to identify
    areas that may be defective. Suspected defect regions are analyzed based on their intensity relative to their
    local background to finalize defect classification.

    Parameters:
    - insp (np.array): Input inspection image.
    - ref (np.array): Reference image to which the inspection image is aligned.
    - intensity_diff_thresh (float): Threshold for intensity difference (default is 0.1, or 10% brighter/darker)
      to classify regions as defects.
    - visualize (bool): If True, displays intermediate images for debugging.

    Returns:
    - proposed_defects_bright (np.array): Binary mask of regions classified as bright defects.
    - proposed_defects_dark (np.array): Binary mask of regions classified as dark defects.
    - aligned_ref (np.array): Aligned version of the reference image.

    Steps:
    1. Convert `insp` and `ref` images to grayscale
    2. Align the inspection image (`insp`) with the reference image (`ref`)
    3. Calculate a difference image by subtracting the aligned reference from the inspected image.
    4. Calculate the mean and standard deviation of the difference image.
    5. Apply a threshold to the difference image, keeping pixels that are 1.5 standard deviations above the mean.
    6. Contour the thresholded image to identify regions suspected of being defects.
    7. For each suspected defect contour:
       - Define the "local background" as the pixels surrounding the contour in the inspected image. This is calculated
         by subtracting the inspected image from its dilation, yielding a mask for nearby pixels.
       - Locally threshold the "local background" using OTSU's method to isolate background pixels around each contour.
       - Compare the mean intensity of the suspected defect in the inspected image to the mean intensity of its local
         background.
       - Classify a contour as a defect if its intensity is at least 10% brighter or darker than the local background.
    8. Perform final morphological operations on the defect masks to clean up small artifacts.

    """

    # Convert images to grayscale
    insp_gr = to_gray(insp)
    ref_gr = to_gray(ref)

    # Align the inspection image to the reference for consistent comparison
    aligned_ref, mask_relevant, _ = align_images(insp_gr, ref_gr)

    # Apply Gaussian blur to reduce noise and make large features more detectable
    reference_blur = cv2.GaussianBlur(aligned_ref, (5, 5), 0)
    test_blur = cv2.GaussianBlur(insp_gr, (5, 5), 0)

    # Compute absolute difference between blurred images to highlight differences
    diff_map = cv2.absdiff(reference_blur, test_blur)
    diff_map = cv2.bitwise_and(diff_map, diff_map, mask=mask_relevant)
    if visualize:
        quickshow(diff_map)

    # Calculate statistics for the difference map
    mean_diff = np.mean(diff_map)
    std_diff = np.std(diff_map)

    # Set a threshold for bright defect detection based on mean and std deviation
    bright_thresh = mean_diff + 1.5 * std_diff
    _, diff_mask = cv2.threshold(diff_map, bright_thresh, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove small noise in bright mask
    kernel = np.ones((3, 3), np.uint8)
    if visualize:
        quickshow(diff_mask)

    # Identify contours in the bright mask
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum and maximum area for defect detection to filter out noise and large irrelevant regions
    min_area = 15
    max_area = 1600

    # Create a local background mask by dilating and subtracting from original image
    local_background = cv2.dilate(insp_gr, kernel) - insp_gr

    # Initialize lists for storing masks of bright and dark defects
    proposed_defects_bright = []
    proposed_defects_dark = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Expand bounding box slightly to include more context
            margin = 3
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(insp_gr.shape[1] - x, w + 2 * margin)
            h = min(insp_gr.shape[0] - y, h + 2 * margin)

            # Create mask for the defect and calculate mean intensity
            mask_contour = np.zeros_like(insp_gr, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 255, thickness=cv2.FILLED)
            mean_intensity_defect, *_ = cv2.mean(insp_gr, mask=mask_contour)

            # Region of interest (ROI) for local background analysis
            inspected_region = insp_gr[y:y + h, x:x + w]
            local_background_region = local_background[y:y + h, x:x + w]

            # Mask local background and calculate mean intensity
            _, mask_local_background = cv2.threshold(local_background_region, 127, 255, cv2.THRESH_OTSU)
            mean_intensity_local_background, *_ = cv2.mean(inspected_region, mask=mask_local_background)

            # Evaluate bright and dark defects based on intensity difference from local background
            if mean_intensity_defect > mean_intensity_local_background * (1 + intensity_diff_thresh):
                proposed_defects_bright.append(mask_contour)

            if mean_intensity_defect < mean_intensity_local_background * (1 - intensity_diff_thresh):
                proposed_defects_dark.append(mask_contour)

    # Combine all detected bright and dark defects into separate masks
    blank_mask = np.zeros_like(insp_gr, dtype=np.uint8)
    proposed_defects_bright = reduce(cv2.bitwise_or, proposed_defects_bright, blank_mask)
    blank_mask = np.zeros_like(insp_gr, dtype=np.uint8)
    proposed_defects_dark = reduce(cv2.bitwise_or, proposed_defects_dark, blank_mask)


    proposed_defects_bright = cv2.morphologyEx(proposed_defects_bright, cv2.MORPH_OPEN, np.ones([3, 3]), iterations=2)
    proposed_defects_dark = cv2.morphologyEx(proposed_defects_dark, cv2.MORPH_OPEN, np.ones([3, 3]), iterations=2)

    return proposed_defects_bright, proposed_defects_dark
