import cv2
import numpy as np
import os
from analyses import contours, rov, local_backgroud, template_matching
from matplotlib import pyplot as plt
from utils import get_case_defects, quickshow, to_gray
import time
from functools import reduce
from skimage.exposure import match_histograms


def combine_defect_masks(defect_masks, mask_shape, iterations):
    """
    Morphologically combines separate defect masks into a single mask.

    This function takes individual binary masks representing detected defect regions
    from different detection methods, dilates each mask to ensure overlapping or
    nearby defects are combined, and then erodes the combined mask to restore
    the original defect sizes.

    Args:
        defect_masks (list): A list of binary masks, where each mask is an individual
            defect mask obtained from various detection methods.
        mask_shape (tuple): The shape of the final combined mask, typically the same
            as the shape of the input images.
        iterations (int): The number of iterations for morphological operations
            (dilation and erosion), which controls how much the masks expand and contract.

    Returns:
        np.ndarray: A single binary mask with all combined defect regions.

    Process:
        - Dilates each mask in `defect_masks` to bridge gaps between defects.
        - Combines all dilated masks into a single mask using a bitwise OR operation.
        - Erodes the combined mask to revert it to the original defect size.
    """

    ker = np.ones([3, 3], dtype=np.uint8)  # Define a 3x3 kernel for morphological operations
    dilated_masks = []
    for mask in defect_masks:
        mask_dilated = cv2.dilate(mask, ker, iterations=iterations)  # Dilate each defect mask
        dilated_masks.append(mask_dilated)

    # Combine all dilated masks into one
    blank_mask = np.zeros(mask_shape, dtype=np.uint8)
    true_defects_combined = reduce(cv2.bitwise_or, dilated_masks, blank_mask)

    # Erode the combined mask to reduce back to original defect size
    true_defects_combined = cv2.erode(true_defects_combined, ker, iterations=iterations)

    return true_defects_combined


def extract_defects_locations(defects_mask):
    defect_locations = []
    num_compopnents, labels = cv2.connectedComponents(defects_mask)

    for idx_comp in range(1,num_compopnents):
        comp_mask = (labels == idx_comp).astype(np.uint8) * 255
        x_loc,y_loc = np.mean(np.argwhere(comp_mask),axis=0).astype(np.int32)
        defect_locations.append([y_loc,x_loc])

    return defect_locations


def save_defect_locations(defect_locations, savepath):
    with open(savepath, 'w') as file:
        if not defect_locations:
            file.write("No defects were detected!\n")
        else:
            file.write("Detected defects:\n")
            for i, (x_loc, y_loc) in enumerate(defect_locations, start=1):
                file.write(f"defect #{i} at x={x_loc}, y={y_loc}\n")
    return


def save_defect_mask_vs_inspected(insp, predicted_defects_mask, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Remove labels and ticks from each axis
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    axes[0].imshow(to_gray(insp))
    axes[0].set_title('Inspected Image')
    axes[1].imshow(predicted_defects_mask)
    axes[1].set_title('Predicted Defects')

    fig.savefig(savepath)
    plt.close()



def save_defect_mask_vs_inspected_vs_ref(ref, insp, predicted_defects_mask, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Remove labels and ticks from each axis
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    axes[0].imshow(to_gray(ref))
    axes[0].set_title('Reference Image')
    axes[1].imshow(to_gray(insp))
    axes[1].set_title('Inspected Image')
    axes[2].imshow(predicted_defects_mask)
    axes[2].set_title('Predicted Defects')

    fig.savefig(savepath)
    plt.close()


def defect_detection_pipeline(paths):
    """
        A comprehensive defect detection pipeline that processes images to identify, classify, and save defect locations.

        Parameters:
        - paths (dict): A dictionary containing paths to necessary files:
            - 'insp': Path to the inspected image.
            - 'ref': Path to the reference image.
            - 'results_case_<case>': Path to the results directory for saving outputs.

        Steps:
        1. **Load and Preprocess Images**
           - Load the inspection and reference images and convert them to grayscale.
           - Match the histogram of the inspection image to that of the reference image for consistency in intensity.

        2. **Segmentation Stage: Detect Proposed Defects**
           - Apply contour-based segmentation to align the reference image with the inspected image, creating initial masks.
           - Use three methods to propose defects:
               - **Contour Misalignment (Method 1)**: Propose defects from contours subtraction.
               - **Range of Value (ROV) Analysis (Method 2)**: Detect defects based on value ranges.
               - **Local Background Analysis (Method 3)**: Segment defects based on intensity relative to local background.

        3. **Classification Stage: Validate Proposed Defects**
           - Use template matching to classify proposed defects as true defects:
               - Validate defects from contours subtraction, ROV analysis, and local background analysis for both bright and dark areas.

        4. **Combine Detected Defect Masks**
           - Morphologically combine individual defect masks from each method to create a single unified mask.
           - Display the combined defect mask for verification.

        5. **Extract and Save Defect Locations**
           - Identify and record locations of detected defects.
           - Save the list of defect locations as a .txt file and overlay the combined defect mask on the inspected image for visual comparison.

        Returns:
        - None
        """

    # Load images
    insp = cv2.imread(paths['insp'])
    ref = cv2.imread(paths['ref'])

    # Convert to grayscale
    insp = to_gray(insp)
    ref = to_gray(ref)

    # Match histogram of inspected to the reference image
    insp = match_histograms(insp, ref).astype(np.uint8)

    # Segmentation Stage:
    # Run each analysis method separately

    # Contour based segmentation with initial alignment using the contours
    ref_aligned, masks = contours.align_using_contours(insp, ref)

    # Defects from contours subtraction (Method1):
    defects_from_contour = contours.propose_defects_contours_subtraction(masks)

    # Defects from Range of Value (ROV) analysis (Method2):
    defects_from_rov = rov.propose_defects_rov(insp, ref, masks)

    # Defects from local background image analysis (Method3):
    defects_local_background_bright, defects_local_background_dark =\
        local_backgroud.propose_defects_local_background(insp, ref)
    # quickshow(defects_local_background_bright)
    # quickshow(insp)


    # Classification Stage:
    # Template matching for validating which defects are truly defects:
    # Method1
    predicted_defects_contours = template_matching.classify_proposed_defects(insp, ref_aligned, defects_from_contour)
    # quickshow(predicted_defects_contours)
    # Method2
    predicted_defects_rov = template_matching.classify_proposed_defects(insp, ref_aligned, defects_from_rov)
    # Method3
    predicted_defects_loc_back_bright = template_matching.classify_proposed_defects(insp, ref_aligned, defects_local_background_bright)
    predicted_defects_loc_back_dark = template_matching.classify_proposed_defects(insp, ref_aligned, defects_local_background_dark)
    # quickshow(predicted_defects_loc_back_bright)
    # quickshow(predicted_defects_loc_back_dark)


    # Morphologically combine the separate detected defect-masks into a single mask to complete holes if exist
    defect_masks = [predicted_defects_contours,
                    predicted_defects_rov,
                    predicted_defects_loc_back_bright,
                    predicted_defects_loc_back_dark]
    predicted_defects_combined_mask = combine_defect_masks(defect_masks, insp.shape[0:2], iterations=3)
    quickshow(predicted_defects_combined_mask)

    # Extract locations for the defects
    defect_locations = extract_defects_locations(predicted_defects_combined_mask)

    # Save the defect locations
    savepath_locations = os.path.join(paths[f'results_case_{case}'], 'defects_locations.txt')
    save_defect_locations(defect_locations, savepath_locations)

    # Save the binary defect mask Vs. the inspected image
    savepath_insp_vs_defect_mask = os.path.join(paths[f'results_case_{case}'], 'inspected_vs_defect_mask.png')
    save_defect_mask_vs_inspected(insp, predicted_defects_combined_mask, savepath_insp_vs_defect_mask)

    savepath_ref_vs_insp_vs_defect_mask = os.path.join(paths[f'results_case_{case}'], 'ref_vs_inspected_vs_defect_mask.png')
    save_defect_mask_vs_inspected_vs_ref(ref, insp,
                                         predicted_defects_combined_mask,
                                         savepath_ref_vs_insp_vs_defect_mask)
    return



if __name__ == "__main__":
    # TODO - change path_main to where the main folder is
    path_main = r"E:\Desktop\Academy\defect_detection".replace('\\', '/')


    global paths
    paths = {}
    paths['main'] = path_main
    paths['results'] = os.path.join(paths['main'], "results")
    os.makedirs(paths['results'], exist_ok=True)
    paths['stat_file'] = os.path.join(paths['results'], 'reference_image_stats.txt')
    paths['defect_folder'] = os.path.join(paths['main'], "defect")
    paths['non_defect_folder'] = os.path.join(paths['main'], "non_defect")
    paths['defect_locations_file'] = os.path.join(paths['defect_folder'], "defects locations.txt")

    defect_cases = [1, 2]
    non_defect_cases = [3]

    # Starting with the defect cases:
    for case in defect_cases:
        # Create a result folder for the case
        paths[f'results_case_{case}'] = os.path.join(paths['results'], f"case_{case}")
        os.makedirs(paths[f'results_case_{case}'], exist_ok=True)

        # Paths to load the images
        paths['insp'] = os.path.join(paths['defect_folder'], f"case{case}_inspected_image.tif")
        paths['ref'] = os.path.join(paths['defect_folder'], f"case{case}_reference_image.tif")

        # Enter the paths into the pipeline
        defect_detection_pipeline(paths)

    # Moving on to the non-defect case:
    for case in non_defect_cases:
        # Create a result folder for the case
        paths[f'results_case_{case}'] = os.path.join(paths['results'], f"case_{case}")
        os.makedirs(paths[f'results_case_{case}'], exist_ok=True)

        # Paths to load the images
        paths['insp'] = os.path.join(paths['non_defect_folder'], f"case{case}_inspected_image.tif")
        paths['ref'] = os.path.join(paths['non_defect_folder'], f"case{case}_reference_image.tif")
        paths[f'results_current_case'] = paths[f'results_case_{case}']

        # Enter the paths into the pipeline
        defect_detection_pipeline(paths)

        print('Detection Completed.')


