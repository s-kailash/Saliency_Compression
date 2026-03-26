import numpy as np


def acrd_function(x):
    """
    Ascending Cosine Roll-down (ACRD) function.
    As described in the paper: Saliency Segmentation Oriented Deep Image Compression.
    This function helps in allocating bits to important pixels.
    """
    # Normalized x should be in [0, 1]
    # f(x) = 0.5 * (1 + cos(pi * (1 - x))) = 0.5 * (1 - cos(pi * x))
    return 0.5 * (1 - np.cos(np.pi * x))


def allocate_bits(saliency_map, object_map=None, spectral_map=None, threshold=0.1):
    """
    Allocates bit weights based on the combined saliency, object, and spectral maps using the ACRD function.
    saliency_map: 2D numpy array with values in [0, 1] (from U2Net)
    object_map: 2D binary numpy array from YOLO (optional)
    spectral_map: 2D numpy array with values in [0, 1] (from Spectral Residual, optional)
    threshold: Minimal saliency score to be considered for enhancement.
    """
    # 1. Combine maps (OR operation / Maximum)
    # We start with the base U2Net saliency map
    combined_map = saliency_map.copy()

    # If a pixel is detected by YOLO, we want to allocate bits to it
    if object_map is not None:
        # We ensure they are the same size
        if object_map.shape != combined_map.shape:
            from PIL import Image
            obj_img = Image.fromarray(object_map).resize(
                (combined_map.shape[1], combined_map.shape[0]), resample=Image.BILINEAR
            )
            object_map = np.array(obj_img)

        # Combine using maximum (keeps the highest score from either map)
        combined_map = np.maximum(combined_map, object_map)

    # If a pixel is detected by the Spectral Residual method, we include it
    if spectral_map is not None:
        if spectral_map.shape != combined_map.shape:
            from PIL import Image
            spec_img = Image.fromarray(spectral_map).resize(
                (combined_map.shape[1], combined_map.shape[0]), resample=Image.BILINEAR
            )
            spectral_map = np.array(spec_img)

        # Combine using maximum
        combined_map = np.maximum(combined_map, spectral_map)

    # 2. Apply a threshold to eliminate noise in non-salient/non-detected regions
    # This helps in reducing the bits used for enhancement in background areas
    combined_map_thresholded = np.where(combined_map < threshold, 0, combined_map)

    # 3. ACRD function to transform combined scores into bit weights
    bit_weights = acrd_function(combined_map_thresholded)

    return bit_weights
