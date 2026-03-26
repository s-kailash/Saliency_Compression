import numpy as np
import cv2


def _compute_spectral_residual(gray_img: np.ndarray) -> np.ndarray:
    """Internal helper to compute spectral residual on a single-channel image."""
    # 1. Transform to frequency domain (DFT)
    dft = np.fft.fft2(gray_img.astype(np.float32))
    magnitude = np.abs(dft)
    phase = np.angle(dft)

    # 2. Find unusual frequencies (the "spectral residual")
    log_magnitude = np.log(magnitude + 1e-8)
    smoothed = cv2.GaussianBlur(log_magnitude, (3, 3), 0)
    residual = log_magnitude - smoothed

    # 3. Convert back to spatial domain
    saliency_fft = np.exp(residual) * np.exp(1j * phase)
    saliency = np.abs(np.fft.ifft2(saliency_fft))

    return saliency


def detect_spectral_residual(image_path: str, scales: list = [0.5, 1.0, 1.5]) -> np.ndarray:
    """
    Computes the saliency map of an image using the Multi-Scale Spectral Residual method.
    This captures both fine details and larger structures, resulting in better moderate-bit regions.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    h, w = image.shape[:2]

    # 1. Prepare image (convert to grayscale)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Multi-scale detection
    saliency_maps = []
    for scale in scales:
        # Resize image for current scale
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < 10 or new_w < 10:  # Minimum size for meaningful DFT
            continue

        scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute spectral residual at this scale
        sal = _compute_spectral_residual(scaled_gray)

        # Resize back to original size
        sal_resized = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        saliency_maps.append(sal_resized)

    # 3. Fuse scales (Mean)
    if not saliency_maps:
        # Fallback to single scale if something went wrong
        fused_saliency = _compute_spectral_residual(gray)
    else:
        fused_saliency = np.mean(saliency_maps, axis=0)

    # 4. Final smoothing and normalization
    fused_saliency = cv2.GaussianBlur(fused_saliency.astype(np.float32), (9, 9), 0)

    # Normalize to [0, 1]
    f_min = fused_saliency.min()
    f_max = fused_saliency.max()
    if f_max > f_min:
        fused_saliency = (fused_saliency - f_min) / (f_max - f_min + 1e-8)
    else:
        fused_saliency = np.zeros_like(fused_saliency)

    return fused_saliency
