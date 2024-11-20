import cv2
from utils import *
from analyses.contours import get_smoothened_contours, contours_to_masks


def calc_stats_single_ref_img(ref):
    """
    Calculate statistical properties of the inner (bright) and outer (dark) regions in a reference image.

    Parameters:
    - ref (np.array): Reference image from which to calculate statistics.

    Returns:
    - stats_in (dict): Dictionary containing minimum, maximum, mean, and standard deviation of pixel values
                       in the inner (bright) region.
    - stats_out (dict): Dictionary containing minimum, maximum, mean, and standard deviation of pixel values
                        in the outer (dark) region.

    Steps:
    1. Generate smoothed contours to segment the different regions in the reference image.
    2. Create masks for the inner (bright), edge, and outer (dark) regions, excluding edge artifacts.
    3. Use masks to separate and calculate pixel statistics for each region.

    Notes:
    - The 'thickness' parameter in `contours_to_masks` is set high to ensure edges are clearly segmented,
      preventing any overlap between inner and outer regions.
    """

    # Generate smoothed contours for accurate region separation in the reference image
    smooth_contours_ref = get_smoothened_contours(ref)

    # Create masks for each region (inner, edge, outer) based on the smoothed contours.
    # Edge thickness is set high to minimize overlap with inner/outer regions.
    mask_in, mask_edges, mask_out = contours_to_masks(smooth_contours_ref, ref.shape[0:2], thickness=10)

    # Convert the reference image to grayscale for uniform intensity processing
    ref_gr = to_gray(ref)

    # Apply the inner mask to extract pixel values in the bright (inner) region
    ref_in = cv2.bitwise_and(mask_in, ref_gr)
    pixels_in = ref_in[ref_in > 0]

    # Apply the outer mask to extract pixel values in the dark (outer) region
    ref_out = cv2.bitwise_and(mask_out, ref_gr)
    pixels_out = ref_out[ref_out > 0]

    # Calculate and store statistics for the inner region
    stats_in = {
        'min_pix': np.min(pixels_in),
        'max_pix': np.max(pixels_in),
        'mean': np.mean(pixels_in),
        'std_dev': np.std(pixels_in)
    }

    # Calculate and store statistics for the outer region
    stats_out = {
        'min_pix': np.min(pixels_out),
        'max_pix': np.max(pixels_out),
        'mean': np.mean(pixels_out),
        'std_dev': np.std(pixels_out)
    }

    return stats_in, stats_out


def read_stats_file(stats_path):
    with open(stats_path, 'r') as f:
        stats_in = {}
        stats_out = {}
        lines = f.readlines()
        # Parse each line and assign to the correct dictionary
        current_dict = None
        for line in lines:
            line = line.strip()  # Remove any leading/trailing whitespace

            # Determine which dictionary to use based on the section
            if line.startswith("In contour"):
                current_dict = stats_in
            elif line.startswith("Out of contour"):
                current_dict = stats_out
            elif '=' in line and current_dict is not None:
                # Split the line into key-value pairs
                key, value = line.split('=')
                key = key.strip()
                value = float(value.strip())  # Convert value to float
                # Add to the current dictionary
                current_dict[key] = value
    return stats_in, stats_out
