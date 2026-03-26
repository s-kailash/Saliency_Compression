import os
import sys
import subprocess
import csv
from metrics import calculate_metrics

DATASETS = ["KODAK", "CLIC"]
ALGOS = ["JPEG", "JPEG2000", "WebP", "BPG", "HEVC"]
QUALITIES = {
    "JPEG": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "JPEG2000": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "WebP": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "BPG": [40, 35, 30, 25, 20, 15, 10, 5],
    "HEVC": [40, 35, 30, 25, 20, 15, 10, 5],
}


def run_benchmark():
    results = []
    for dataset in DATASETS:
        dataset_path = os.path.join("..", "datasets", dataset)
        for image_file in os.listdir(dataset_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            original_image_path = os.path.abspath(os.path.join(dataset_path, image_file))

            for algo in ALGOS:
                for quality in QUALITIES[algo]:
                    output_dir = os.path.abspath(os.path.join(algo, "output"))
                    os.makedirs(output_dir, exist_ok=True)

                    output_filename = f"{os.path.splitext(image_file)[0]}_{algo}_{quality}.{algo.lower()}"
                    if algo == "JPEG2000":
                        output_filename = f"{os.path.splitext(image_file)[0]}_{algo}_{quality}.jp2"

                    output_path = os.path.join(output_dir, output_filename)

                    compress_script = os.path.join(algo, "compress.py")

                    command = [sys.executable, compress_script, original_image_path, output_path, str(quality)]

                    try:
                        subprocess.run(command, check=True, capture_output=True, text=True)

                        metrics = calculate_metrics(original_image_path, output_path)

                        results.append([
                            dataset,
                            image_file,
                            algo,
                            quality,
                            metrics["psnr"],
                            metrics["ssim"],
                            metrics["bpp"]
                        ])
                        print(f"Processed {image_file} with {algo} at quality {quality}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to process {image_file} with {algo} at quality {quality}: {e}")
                        print(f"STDOUT: {e.stdout}")
                        print(f"STDERR: {e.stderr}")
                    except FileNotFoundError as e:
                        print(f"Failed to find file for {image_file} with {algo} at quality {quality}: {e}")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image", "algorithm", "quality", "psnr", "ssim", "bpp"])
        writer.writerows(results)


if __name__ == "__main__":
    run_benchmark()
