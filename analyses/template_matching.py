import numpy as np
import cv2
from utils import quickshow, to_gray
from functools import reduce


def classify_proposed_defects(insp, aligned_ref, proposed_defects):
    """
        Classifies suspected defects in an inspected image by performing template matching with an aligned reference image.
        If a suspected defect matches the corresponding area in the reference image, it is discarded; otherwise, it is
        classified as a true defect.

        :param insp: Grayscale inspected image.
        :param aligned_ref: Grayscale aligned reference image.
        :param proposed_defects: Binary mask where suspected defect pixels have a value of 255; other pixels are 0.
        :return: Binary mask of true defects from a single method in the inspected image.
        """

    # Ensure images are in grayscale
    insp = to_gray(insp)
    aligned_ref = to_gray(aligned_ref)

    # Label connected components in the defects mask
    num_compopnents, labels = cv2.connectedComponents(proposed_defects)

    max_size_template = 900  # Threshold for switching to masked template matching
    true_defects = []  # List to hold confirmed defect masks

    # Loop over each component, excluding the background
    for idx_comp in range(1, num_compopnents):  # Skip background label 0
        component_mask = (labels == idx_comp).astype(np.uint8) * 255

        # Set template box (bounding box around the component)
        x, y, w, h = cv2.boundingRect(component_mask)
        bounding_box = (x, y, w, h)
        # Increase the bounding box size by a bit for better template matching
        # expand = 2
        # template_box = (max(0, x - expand), max(0, y - expand),
        #                 min(aligned_ref.shape[1] - x, w + 2 * expand), min(aligned_ref.shape[0] - y, h + 2 * expand))
        template_box = bounding_box
        template_box_area = template_box[2] * template_box[3]

        # Perform template matching based on the size of the bounding box
        if template_box_area <= max_size_template:
            # For small components, use regular template matching
            is_defect, max_val = local_template_matching(insp, aligned_ref, template_box)
            if is_defect:
                true_defects.append(component_mask)
        else:
            # For larger components, use masked template matching
            is_defect, max_val = local_template_matching(insp, aligned_ref, template_box, tm_type='masked', mask=component_mask)
            if is_defect:
                true_defects.append(component_mask)

    # Combine all defects in single mask
    blank_mask = np.zeros_like(insp, dtype=np.uint8)
    true_defects_mask = reduce(cv2.bitwise_or, true_defects, blank_mask)

    return true_defects_mask


def local_template_matching(insp, aligned_ref, template_box, margin=30, similarity_threshold=0.80, tm_type='regular',
                            mask=None):
    """ This function performs template matching on regions suspected as defects in the inspected image, comparing them
    to the *local* area in the corresponding reference image. A match would signify that the suspected region exists
    in the local reference as well, and the suspected region would then be discarded. The reason we can use *local* template
    matching is because the reference is already aligned, but not perfectly. Part of this function's job is to determine
    whether a suspected region is a defect or just a result of edges misalignment.
    This implementation also allows for masked template matching, used when the bounding box of a suspected region
    is too large and may contain other suspected regions. The metric used for matching is the cv2.TM_CCOEFF_NORMED.


    :param insp: Inspected image.
    :param aligned_ref: An already aligned reference image.
    :param template_box: A bounding box (x,y,w,h) around the suspected region in the inspected image.
    :param margin: The number of pixels we extend the reference region in each direction. Default value is 30
    :param similarity_threshold: The threshold value to detemine if a template contains defect or not. Default value is 0.75
    :param tm_type: template_matching type - either 'regular' or 'masked'.
    :param mask: A binary mask for 'masked' type template matching.
    :return: is_defect (bool)

    Notes:
    - An optimal value for similarity_threshold could be gained by looking at the reference images of non-defective
    samples and analyzing the similarity values achieved there using template matching on a bunch of random local areas.
    """

    # Convert to grayscale
    insp = to_gray(insp)
    aligned_ref = to_gray(aligned_ref)

    x, y, w, h = template_box
    # The expanded reference region coordinates
    x_ref = max(0, x - margin)
    y_ref = max(0, y - margin)
    w_ref = min(aligned_ref.shape[1] - x_ref, w + 2 * margin)
    h_ref = min(aligned_ref.shape[0] - y_ref, h + 2 * margin)

    # The suspected region from the inspected image (template)
    template = insp[y:y + h, x:x + w]

    # The expanded reference region from the reference image
    reference_region = aligned_ref[y_ref:y_ref + h_ref, x_ref:x_ref + w_ref]

    # Perform template matching
    if tm_type == 'regular':
        match_result = cv2.matchTemplate(reference_region, template, cv2.TM_CCOEFF_NORMED)
    elif tm_type == 'masked':
        mask_template = mask[y:y + h, x:x + w]
        match_result = cv2.matchTemplate(reference_region, template, cv2.TM_CCOEFF_NORMED, mask=mask_template)

    # Get the best match location and the match score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    # Classification - according to the similarity_threshold
    is_defect = max_val < similarity_threshold
    return is_defect, max_val
