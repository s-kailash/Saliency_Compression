import os
import sys
import subprocess


def compress_bpg(input_path, output_path, quality=28):
    """
    Compresses an image using BPG.
    """
    bpgenc_path = os.path.join(os.path.dirname(__file__), "bpgenc.exe")
    command = [bpgenc_path, "-o", output_path, "-q", str(quality), input_path]
    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress_bpg.py <input_image> <output_image> <quality>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    bpg_quality = int(sys.argv[3])

    compress_bpg(input_image, output_image, bpg_quality)
