from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image


def get_object_segmentation_map(image_path, model_name="yolov8n-seg.pt"):
    """
    Uses YOLOv8 Nano Segmentation to detect objects and create a binary mask.
    This helps in complex scenes where saliency might miss semantically important objects.
    """
    # Load the YOLOv8 model (Nano version for speed)
    model = YOLO(model_name)

    # Run inference on CPU to avoid CUDA/torchvision version mismatch issues with NMS
    results = model(image_path, verbose=False, device='cpu')

    # Get original image size
    img = Image.open(image_path)
    w, h = img.size

    # Initialize an empty mask
    combined_mask = np.zeros((h, w), dtype=np.float32)

    # Iterate through results
    for result in results:
        if result.masks is not None:
            # result.masks.data contains the binary masks for each detected object
            # We combine all masks into one
            masks = result.masks.data.cpu().numpy()  # (num_objects, mask_h, mask_w)

            for mask in masks:
                # Resize mask to original image size if needed
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask_img = Image.fromarray(mask).resize((w, h), resample=Image.BILINEAR)
                    mask = np.array(mask_img)

                # Add to combined mask (OR operation)
                combined_mask = np.maximum(combined_mask, mask)

    return combined_mask
