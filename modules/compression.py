import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# A simple Convolutional Autoencoder to simulate Deep Image Compression
class SimpleCompressionNet(nn.Module):
    def __init__(self):
        super(SimpleCompressionNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 1/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 1/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 1/8
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        # In real compression, we would quantize 'latent' here
        # For this implementation, we simulate distortion by adding noise or quantizing
        reconstructed = self.decoder(latent)
        return reconstructed


def compress_image_pytorch(image_path, quality_factor=0.1, is_base=False):
    """
    Simulates deep image compression using a simple autoencoder.
    quality_factor: simulates bit-depth/noise (higher means better quality)
    is_base: if True, applies EXTREME compression strategies to background
    """
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size

    if is_base:
        # EXTREME STRATEGY 1: Aggressive Downsampling
        # Resize to 1/6 of the size, then resize back. This destroys high-frequency data.
        small_size = (max(16, orig_size[0] // 6), max(16, orig_size[1] // 6))
        img = img.resize(small_size, resample=Image.BILINEAR)
        img = img.resize(orig_size, resample=Image.BILINEAR)

    orig_np = np.array(img).astype(float) / 255.0

    # 1. Simulate compression artifacts (blur + noise)
    noise_sigma = 0.05 * (1.0 - quality_factor)
    if is_base:
        noise_sigma *= 2.0  # Even more noise/distortion in background

    noise = np.random.normal(0, noise_sigma, orig_np.shape)
    compressed_np = np.clip(orig_np + noise, 0, 1)

    # 2. Add blur to simulate high-frequency loss
    # More blur for background (is_base) = better compression later
    blur_sigma = 1.0 * (1.0 - quality_factor)
    if is_base:
        # EXTREME STRATEGY 2: Heavy blurring
        # Optimized: reduced from 4.0 to 2.5 to be less 'totally blurred' but still highly compressible
        blur_sigma *= 2.5

    if blur_sigma > 0:
        # Optimization: Use cv2.boxFilter instead of scipy.ndimage.gaussian_filter for speed
        import cv2
        # Approximate Gaussian blur with box blur kernel size
        # Kernel size roughly 2*sigma*sqrt(3) or just a reasonable odd integer
        ksize = int(blur_sigma * 3) | 1  # Ensure odd
        if ksize > 1:
            compressed_np = cv2.boxFilter(compressed_np, -1, (ksize, ksize))

    if is_base:
        # STRATEGY 3: Posterization REMOVED to avoid blocky artifacts
        # Instead, we rely on heavy blur and downsampling, which is more visually pleasing
        pass
        # # EXTREME STRATEGY 3: Posterization (Bit-depth reduction)
        # # Reduce to 4 levels per channel (2 bits) to destroy smooth gradients
        # compressed_np = np.round(compressed_np * 4) / 4

    compressed_img = Image.fromarray((compressed_np * 255).astype(np.uint8))
    return compressed_img


def layered_compression(image_path, bit_weights, base_quality=0.2, enhancement_quality=0.9):
    """
    Implements the Base + Enhancement layer logic.
    - Base layer: Low quality (simulated).
    - Enhancement layer: High quality (simulated).
    """
    # 1. Base Layer (Low quality + aggressive blur)
    base_img = compress_image_pytorch(image_path, quality_factor=base_quality, is_base=True)
    base_np = np.array(base_img).astype(float) / 255.0

    # 2. Enhancement Layer (High quality)
    enhanced_img = compress_image_pytorch(image_path, quality_factor=enhancement_quality, is_base=False)
    enhanced_np = np.array(enhanced_img).astype(float) / 255.0

    # 3. Blending based on bit weights
    weights_3d = np.stack([bit_weights] * 3, axis=-1)

    # Final = (1 - weight) * Base + weight * Enhanced
    final_np = (1.0 - weights_3d) * base_np + weights_3d * enhanced_np

    final_img = Image.fromarray((final_np * 255).astype(np.uint8))
    return final_img, base_img, enhanced_img
