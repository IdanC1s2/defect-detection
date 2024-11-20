import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_case_defects(case, defect_loc_path):
    """
    Function that reads the defect positions from the defects_locations.txt file for each case.
    :param case: case index, integer
    :param defect_loc_path: path to the defects_locations.txt file
    :return: defects list, each elenent is an (x,y) tuple.
    """
    defects = []
    case_exists = False

    with open(defect_loc_path, 'r') as f:
        for line in f:
            if f"case {case}:" in line:
                case_exists = True
                continue  # go to next line

            if case_exists:
                if "defect #" in line:  # append tuple if a new defect is found
                    parts = line.strip().split(',')
                    x = int(parts[0].split('=')[1])
                    y = int(parts[1].split('=')[1])
                    defects.append((x, y))

                if "case" in line:  # break if next case appears
                    break

    if case_exists:
        return defects
    else:
        raise ValueError("Expected case not found in the file.")


def calc_alignment_error(insp, aligned_ref, h_mat):
    """ This function calculates the alignment error (MAE) between the inspected image and the aligned reference image.
    It ignores the black regions (irrelevant pixels) introduced due to the warp transformation applied on the reference image.
    It also returns the mask for the relevant pixels.

    :param insp: Inspected image
    :param aligned_ref: Aligned reference image
    :param h_mat: The warp transformation used to align the reference
    :return: mae (the mean absolute error) , mask_relevant (mask of the relevant pixels)
    """
    # Create a white mask with the same shape as the reference image
    h, w = insp.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255  # Fill with white

    # Apply the same transformation to the mask
    mask_relevant = cv2.warpPerspective(mask, h_mat, (w, h), flags=cv2.INTER_NEAREST)

    # Calculate the absolute difference between the inspected image and aligned reference image
    diff = cv2.absdiff(insp, aligned_ref)

    # Apply the transformed mask to ignore black regions due to warp transformation
    diff_masked = cv2.bitwise_and(diff, diff, mask=mask_relevant)

    # Compute the mean absolute error only over the masked area
    mae = cv2.mean(diff_masked, mask=mask_relevant)[0]  # Mean of non-zero values

    return mae, mask_relevant


def find_homography_RANSAC(insp, ref, keepPercent=0.2, nFeatures=5000):
    """ This function finds a homography between the inspected image and the reference image, using the RANSAC algorithm.
    :param insp: Inspected image.
    :param ref: Reference image. This is the image we aim to align.
    :param keepPercent: Percentage of top matches we use for RANSAC.
    :param nFeatures: The maximum number of features for the RANSAC algorithm.
    :return: h_mat
    """
    orb = cv2.ORB_create(nFeatures)
    keypoints1, descriptors1 = orb.detectAndCompute(insp, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref, None)

    # Use BFMatcher to find the best matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use only the top matches (you can adjust this number)
    n_good_matches = int(len(matches) * keepPercent)  # Keep top 15% of matches
    if n_good_matches < 4:
        return False
    matches = matches[:n_good_matches]

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix
    h_mat, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    return h_mat


def to_gray(img):
    """ This function turns an image to its grayscale version.
    :param img: An image.
    :return: The grayscaled image.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    else:  # already gray
        img_gr = img

    return img_gr


def align_images(insp, ref, nFeatures=5000, keepPercent=0.15, optimize_keepPercetnt=True, visualize=False):
    """ This function aligns the reference image over the inspected image. It picks the optimal 'keepPercent' value
    for the RANSAC algorithm to minimize the alignment error.
    :param insp: Inspected image.
    :param ref: Reference image. This is the image we aim to align.
    :param nFeatures: The maximum number of features for the RANSAC algorithm.
    :param visualize:  bool, whether to create a visualization figure or not.
    :return:
    """

    # convert both the input image and reference to grayscale
    gray_insp = to_gray(insp)
    gray_ref = to_gray(ref)

    blurred_insp = cv2.GaussianBlur(gray_insp, (5, 5), cv2.BORDER_DEFAULT)
    blurred_ref = cv2.GaussianBlur(gray_ref, (5, 5), cv2.BORDER_DEFAULT)


    if optimize_keepPercetnt:
        # The optimal value of keepPercent is searched for an optimized alignment.
        best_mae = np.inf
        for keepPercent in tqdm(np.arange(0.1, 1, 0.01)):
            h_mat = find_homography_RANSAC(ref=blurred_ref, insp=blurred_insp, keepPercent=keepPercent, nFeatures=nFeatures)
            if h_mat is not False:
                aligned_ref = cv2.warpPerspective(ref, h_mat, (insp.shape[1], insp.shape[0]))

                mae, mask_relevant = calc_alignment_error(insp, aligned_ref, h_mat)
            if mae < best_mae:
                best_mae = mae
                h_mat_best = h_mat
                mask_relevant_best = mask_relevant
                aligned_ref_best = aligned_ref
    else:
        h_mat_best = find_homography_RANSAC(ref=blurred_ref, insp=blurred_insp, keepPercent=keepPercent, nFeatures=nFeatures)
        aligned_ref_best = cv2.warpPerspective(ref, h_mat_best, (insp.shape[1], insp.shape[0]))
        mae, mask_relevant_best = calc_alignment_error(insp, aligned_ref_best, h_mat_best)

    # Visualizing the alignment:
    if visualize:
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].imshow(insp)
        axes[0, 0].set_title('Inspected')

        axes[1, 0].imshow(ref)
        axes[1, 0].set_title('Reference')

        axes[0, 1].imshow(aligned_ref_best)
        axes[0, 1].set_title('Aligned_Reference')

        diff = cv2.absdiff(aligned_ref_best, insp)
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title('diff = |Inspected - Aligned_Reference|')

        return aligned_ref_best, mask_relevant_best, h_mat_best, fig

    return aligned_ref_best, mask_relevant_best, h_mat_best






def overlay_region_masks(img, mask_in, mask_out, mask_edges):
    """
        Overlays colored masks on an image to distinguish between regions.

        Parameters:
        - img (np.array): Original image.
        - mask_in (np.array): Inner (bright) region mask, displayed in red.
        - mask_out (np.array): Outer (dark) region mask, displayed in blue.
        - mask_edges (np.array): Edge region mask, displayed in green.

        Returns:
        - overlayed_img (np.array): Image with blended, colored overlays.

        Each mask is colored (red for `mask_in`, green for `mask_edges`, blue for `mask_out`)
        and blended onto `img` with transparency.
        """

    # Set the desired color for each mask
    mask_in_colored = cv2.merge([np.zeros_like(mask_in), np.zeros_like(mask_in), mask_in])  # Red
    mask_edges_colored = cv2.merge([np.zeros_like(mask_edges), mask_edges, np.zeros_like(mask_edges)])  # Green
    mask_out_colored = cv2.merge([mask_out, np.zeros_like(mask_out), np.zeros_like(mask_out)])  # Blue

    # Level for blending
    alpha = 0.5

    # Convert original image to float32 for blending
    img = img.astype(np.float32)

    # Overlay each colored mask onto the original image with transparency
    overlayed_img = cv2.addWeighted(img, 1, mask_in_colored.astype(np.float32), alpha, 0)
    overlayed_img = cv2.addWeighted(overlayed_img, 1, mask_edges_colored.astype(np.float32), alpha, 0)
    overlayed_img = cv2.addWeighted(overlayed_img, 1, mask_out_colored.astype(np.float32), alpha, 0)

    # Convert back to uint8 for display
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
    return overlayed_img


def quickshow(img):
    plt.figure()
    plt.imshow(img)
    return

#
# def get_stats(pixels):
#     stats = {}
#     stats['min_pix'] = np.min(pixels)
#     stats['max_pix'] = np.max(pixels)
#     stats['mean'] = np.mean(pixels)
#     stats['std_dev'] = np.std(pixels)
#
#     # Set a threshold for outliers
#     # Values beyond 3 standard deviations from the mean are considered outliers
#     stats['thresh_low'] = stats['mean'] - 3 * stats['std_dev']
#     stats['thresh_high'] = stats['mean'] + 3 * stats['std_dev']
#
#     return stats

#
# def align_images_using_Canny_gradients(insp, ref, nFeatures=5000, visualize=False):
#     """ Feature based image alignment using RANSAC algorithm on the Canny gradients.
#
#     :param insp: Inspected image
#     :param ref: Reference image. This is the image to be aligned
#     :param nFeatures: The maximum number of features for the RANSAC algorithm
#
#     :return:
#     """
#
#     # Convert images to grayscale
#     gray_insp = cv2.cvtColor(insp, cv2.COLOR_BGR2GRAY)
#     gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#
#     blurred_insp = cv2.GaussianBlur(gray_insp, (5, 5), cv2.BORDER_DEFAULT)
#     blurred_ref = cv2.GaussianBlur(gray_ref, (5, 5), cv2.BORDER_DEFAULT)
#
#     # 50 and 200 kinda works
#     minval = 50
#     maxval = 200
#
#     best_mae = np.inf
#     canny_grad_insp = cv2.Canny(blurred_insp, threshold1=minval, threshold2=maxval)
#     canny_grad_ref = cv2.Canny(blurred_ref, threshold1=minval, threshold2=maxval)
#     for keepPercent in tqdm(np.arange(0.1, 1, 0.01)):
#         h_mat = find_homography_RANSAC(ref=canny_grad_ref, insp=canny_grad_insp, keepPercent=keepPercent,
#                                        nFeatures=nFeatures)
#         if h_mat is not False:
#             aligned_ref = cv2.warpPerspective(ref, h_mat, (insp.shape[1], insp.shape[0]))
#
#             mae, mask = calc_alignment_error(insp, aligned_ref, h_mat)
#         if mae < best_mae:
#             best_mae = mae
#             h_mat_best = h_mat
#             mask_best = mask
#             best_keepPercent = keepPercent
#             aligned_ref_best = aligned_ref
#
#     if visualize:
#         # Visualizing the alignment:
#         fig, axes = plt.subplots(2, 3)
#
#         axes[0, 0].imshow(insp)
#         axes[0, 0].set_title('Inspected')
#
#         axes[1, 0].imshow(ref)
#         axes[1, 0].set_title('Reference')
#
#         axes[0, 1].imshow(canny_grad_insp)
#         axes[0, 1].set_title('Gradients of Inspected')
#
#         axes[1, 1].imshow(canny_grad_ref)
#         axes[1, 1].set_title('Gradients of Reference')
#
#         axes[0, 2].imshow(aligned_ref_best)
#         axes[0, 2].set_title('Aligned Reference')
#
#         diff = cv2.absdiff(aligned_ref_best, insp)
#         axes[1, 2].imshow(diff)
#         axes[1, 2].set_title('diff = |Inspected - Aligned Reference|')
#         return aligned_ref_best, mask_best, h_mat_best, fig
#
#     return aligned_ref_best, mask_best, h_mat_best
#
#
# def img_to_gradient(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     grad = cv2.magnitude(grad_x, grad_y)
#     grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     return grad
