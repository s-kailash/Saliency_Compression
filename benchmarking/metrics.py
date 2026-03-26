import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import os


def _read_image_bgr(path):
    """
    Read an image as a BGR numpy array.
    Falls back to Pillow for formats OpenCV cannot decode (e.g. AVIF).
    """
    img = cv2.imread(path)
    if img is None and path.lower().endswith('.avif'):
        try:
            import pillow_avif  # registers AVIF handler with Pillow
            from PIL import Image
            pil_img = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not read AVIF image from {path}: {e}"
            )
    return img


def calculate_metrics(original_path, compressed_path, size_path=None):
    """
    Calculates PSNR, SSIM, and BPP for a pair of original and compressed images.

    size_path: optional path whose file size should be used for BPP instead of
               compressed_path.  Use this when compressed_path is a temporary
               decoded file (e.g. for BPG) and the actual encoded file is
               different (size_path = the real .bpg file).
    """
    original_img = _read_image_bgr(original_path)
    compressed_img = _read_image_bgr(compressed_path)

    if original_img is None:
        raise FileNotFoundError(f"Could not read original image from {original_path}")
    if compressed_img is None:
        raise FileNotFoundError(f"Could not read compressed image from {compressed_path}")

    # Ensure images are the same size
    if original_img.shape != compressed_img.shape:
        compressed_img = cv2.resize(compressed_img, (original_img.shape[1], original_img.shape[0]))

    # PSNR
    psnr_value = psnr(original_img, compressed_img)

    # SSIM
    ssim_value = ssim(original_img, compressed_img, multichannel=True, channel_axis=2, data_range=255)

    # BPP — use size_path when provided (e.g. actual .bpg file instead of temp decoded PNG)
    height, width, _ = original_img.shape
    num_pixels = height * width
    compressed_size_bytes = os.path.getsize(size_path if size_path else compressed_path)
    bpp_value = (compressed_size_bytes * 8) / num_pixels

    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "bpp": bpp_value
    }
