import cv2
import os
import sys


def compress_jpeg2000(input_path, output_path, quality=95):
    """
    Compresses an image using JPEG2000.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")

    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress_jpeg2000.py <input_image> <output_image> <quality>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    jpeg2000_quality = int(sys.argv[3])

    compress_jpeg2000(input_image, output_image, jpeg2000_quality)
