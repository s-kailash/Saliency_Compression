import os
import sys
import subprocess
import shutil
import tempfile


# Path to the project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")


def compress_ours(input_path, output_path, quality=50):
    """
    Compresses an image using the custom saliency-based algorithm (main.py).

    quality maps to --avif_quality (1-100, lower = smaller file / higher compression).
    The pipeline also runs with default base_quality=0.1 and enhancement_quality=0.9.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, MAIN_PY,
                "--input", input_path,
                "--output_dir", tmpdir,
                "--avif_quality", str(quality),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )

        # main.py names the output <stem>_step4_final_compressed.avif
        input_stem = os.path.splitext(os.path.basename(input_path))[0]
        avif_file = os.path.join(tmpdir, f"{input_stem}_step4_final_compressed.avif")

        if not os.path.exists(avif_file):
            raise FileNotFoundError(
                f"Expected output not found: {avif_file}\n"
                f"STDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
            )

        shutil.copy2(avif_file, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress.py <input_image> <output_path> <quality>")
        sys.exit(1)

    compress_ours(sys.argv[1], sys.argv[2], int(sys.argv[3]))
