import os
import argparse
from PIL import Image
import pillow_avif  # For AVIF support
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from modules.saliency import get_saliency_map, download_weights
from modules.object_detection import get_object_segmentation_map
from modules.saliency_spectral import detect_spectral_residual
from modules.bit_allocation import allocate_bits
from modules.compression import layered_compression


def main():
    parser = argparse.ArgumentParser(description="Saliency Segmentation Oriented Deep Image Compression")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for outputs")
    parser.add_argument("--base_quality", type=float, default=0.1, help="Quality for base layer (0.0-1.0)")
    parser.add_argument("--enhancement_quality", type=float, default=0.9, help="Quality for enhancement layer (0.0-1.0)")
    parser.add_argument("--avif_quality", type=int, default=28, help="AVIF export quality (1-100)")
    parser.add_argument("--saliency_threshold", type=float, default=0.15, help="Threshold for saliency enhancement")
    parser.add_argument("--use_yolo", action="store_true", default=True, help="Use YOLO segmentation to enhance saliency")
    parser.add_argument("--use_spectral", action="store_true", default=True, help="Use Spectral Residual saliency to enhance")

    args = parser.parse_args()

    # 1. Create output directory and get input filename prefix
    os.makedirs(args.output_dir, exist_ok=True)
    input_filename = os.path.basename(args.input).split('.')[0]

    print(f"--- Starting Compression Pipeline for: {args.input} ---")

    # 2. Saliency Detection (U2NetP)
    print("Step 1a: Running Saliency Detection (U2NetP)...")
    weight_path = download_weights("models")
    saliency_map = get_saliency_map(args.input, weight_path)

    # 3. Object Detection (YOLOv8 Nano Seg)
    object_map = None
    if args.use_yolo:
        print("Step 1b: Running Object Segmentation (YOLOv8n-seg)...")
        object_map = get_object_segmentation_map(args.input)

        # Save Object Map
        obj_img = Image.fromarray((object_map * 255).astype(np.uint8))
        obj_path = os.path.join(args.output_dir, f"{input_filename}_step1b_object_map.png")
        obj_img.save(obj_path)
        print(f"Saved object map to {obj_path}")

    # 4. Spectral Saliency
    spectral_map = None
    if args.use_spectral:
        print("Step 1c: Running Spectral Residual Saliency...")
        spectral_map = detect_spectral_residual(args.input)

        # Save Spectral Map
        spec_img = Image.fromarray((spectral_map * 255).astype(np.uint8))
        spec_path = os.path.join(args.output_dir, f"{input_filename}_step1c_spectral_map.png")
        spec_img.save(spec_path)
        print(f"Saved spectral map to {spec_path}")

    # Save Saliency Map
    sal_img = Image.fromarray((saliency_map * 255).astype(np.uint8))
    sal_path = os.path.join(args.output_dir, f"{input_filename}_step1a_saliency_map.png")
    sal_img.save(sal_path)
    print(f"Saved saliency map to {sal_path}")

    # 5. Bit Allocation (ACRD Function)
    print(f"Step 2: Calculating Combined Bit Allocation (ACRD, threshold={args.saliency_threshold})...")
    bit_weights = allocate_bits(
        saliency_map,
        object_map=object_map,
        spectral_map=spectral_map,
        threshold=args.saliency_threshold
    )

    # Save Bit Weight Map (Visual Representation)
    bw_img = Image.fromarray((bit_weights * 255).astype(np.uint8))
    bw_path = os.path.join(args.output_dir, f"{input_filename}_step2_bit_weights.png")
    bw_img.save(bw_path)
    print(f"Saved bit weight map to {bw_path}")

    # 4. Layered Compression (Base + Enhancement)
    print("Step 3: Performing Layered Compression...")
    final_img, base_img, enhanced_img = layered_compression(
        args.input,
        bit_weights,
        base_quality=args.base_quality,
        enhancement_quality=args.enhancement_quality
    )

    # Save Intermediate and Final Results
    base_path = os.path.join(args.output_dir, f"{input_filename}_step3_base_layer.png")
    base_img.save(base_path)

    enhanced_path = os.path.join(args.output_dir, f"{input_filename}_step3_enhancement_full.png")
    enhanced_img.save(enhanced_path)

    final_path = os.path.join(args.output_dir, f"{input_filename}_step4_final_compressed.avif")
    final_img.save(final_path, format="AVIF", quality=args.avif_quality, subsampling='4:2:0')  # Optimized AVIF
    print(f"Saved final compressed image to {final_path}")

    # 5. Compression Metrics
    print("\n--- Compression Metrics ---")
    orig_size = os.path.getsize(args.input)
    comp_size = os.path.getsize(final_path)
    ratio = orig_size / comp_size if comp_size > 0 else 0

    print(f"Original Image Size: {orig_size / 1024:.2f} KB")
    print(f"Compressed Image Size: {comp_size / 1024:.2f} KB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print("--------------------------\n")

    # 7. Visual Summary (Optional)
    print("Generating visual summary...")
    num_plots = 4 + args.use_yolo + args.use_spectral
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    orig_img = Image.open(args.input)

    idx = 0
    axes[idx].imshow(orig_img)
    axes[idx].set_title("Original Image")
    idx += 1

    axes[idx].imshow(saliency_map, cmap='gray')
    axes[idx].set_title("Saliency (U2NetP)")
    idx += 1

    if args.use_yolo:
        axes[idx].imshow(object_map, cmap='gray')
        axes[idx].set_title("Objects (YOLOv8n)")
        idx += 1

    if args.use_spectral:
        axes[idx].imshow(spectral_map, cmap='gray')
        axes[idx].set_title("Saliency (Spectral)")
        idx += 1

    axes[idx].imshow(base_img)
    axes[idx].set_title(f"Base Layer (Q={args.base_quality})")
    idx += 1

    axes[idx].imshow(final_img)
    axes[idx].set_title(f"Final (Ratio: {ratio:.2f}x)")

    for ax in axes:
        ax.axis('off')

    summary_path = os.path.join(args.output_dir, f"{input_filename}_compression_summary.png")
    plt.savefig(summary_path)
    print(f"Saved visual summary to {summary_path}")

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
