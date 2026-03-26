import cv2
import os
import sys


def compress_webp(input_path, output_path, quality=95):
    """
    Compresses an image using WebP.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")

    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress_webp.py <input_image> <output_image> <quality>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    webp_quality = int(sys.argv[3])

    compress_webp(input_image, output_image, webp_quality)
