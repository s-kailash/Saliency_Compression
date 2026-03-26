import os
import sys
import argparse
import cv2
import numpy as np
import time
from PIL import Image
import pillow_avif  # For AVIF support

# Add parent directory to sys.path to access root modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import root modules
from modules.saliency import get_saliency_map, download_weights
from modules.object_detection import get_object_segmentation_map
from modules.saliency_spectral import detect_spectral_residual
from modules.bit_allocation import allocate_bits
from modules.compression import layered_compression

# Import video modules
from video_modules.temporal_tracking import calculate_optical_flow, warp_saliency_map, calculate_frame_change, temporal_smoothing
from video_modules.video_processing import extract_frames, frames_to_video, frames_to_video_ffmpeg


def main():
    parser = argparse.ArgumentParser(description="Saliency Segmentation Oriented Video Compression")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="video_output", help="Directory for outputs")
    parser.add_argument("--base_quality", type=float, default=0.05, help="Quality for base layer (0.0-1.0)")
    parser.add_argument("--enhancement_quality", type=float, default=0.9, help="Quality for enhancement layer (0.0-1.0)")
    parser.add_argument("--crf", type=int, default=32, help="FFmpeg H.265 CRF (18-51, higher = smaller size)")
    parser.add_argument("--preset", type=str, default="medium", help="FFmpeg preset (faster, medium, slow, veryslow)")
    parser.add_argument("--saliency_threshold", type=float, default=0.25, help="Threshold for saliency enhancement")
    parser.add_argument("--gop_size", type=int, default=60, help="Group of Pictures size for full detection")
    parser.add_argument("--change_threshold", type=float, default=150.0, help="MSE threshold for forced detection")
    parser.add_argument("--check_interval", type=int, default=5, help="Interval to check for forced detection")
    parser.add_argument("--clarity_reduction", type=float, default=0.9, help="Reduction factor for salient clarity")
    parser.add_argument("--use_yolo", action="store_true", default=True, help="Use YOLO segmentation to enhance saliency")
    parser.add_argument("--use_spectral", action="store_true", default=True, help="Use Spectral Residual saliency to enhance")

    args = parser.parse_args()

    overall_start_time = time.time()

    # 1. Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    temp_frames_dir = os.path.join(args.output_dir, "temp_frames")
    processed_frames_dir = os.path.join(args.output_dir, "processed_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)

    video_name = os.path.basename(args.input).split('.')[0]
    weight_path = download_weights("models")

    # 2. Extract frames
    print(f"--- Step 1: Extracting frames from {args.input} ---")
    fps, frame_count = extract_frames(args.input, temp_frames_dir)
    print(f"Extracted {frame_count} frames at {fps:.2f} FPS")

    # 3. Process frames
    print(f"--- Step 2: Processing {frame_count} frames ---")

    last_full_saliency_map = None
    prev_frame = None

    # Create subfolder for saliency maps to show Step 2 process
    saliency_viz_dir = os.path.join(args.output_dir, f"{video_name}_saliency_steps")
    os.makedirs(saliency_viz_dir, exist_ok=True)

    for i in range(frame_count):
        frame_path = os.path.join(temp_frames_dir, f"frame_{i:05d}.jpg")
        curr_frame = cv2.imread(frame_path)

        is_keyframe = (i % args.gop_size == 0)
        force_detection = False

        # Check for global change at specified intervals
        if prev_frame is not None and i % args.check_interval == 0:
            mse_change = calculate_frame_change(prev_frame, curr_frame)
            if mse_change > args.change_threshold:
                force_detection = True
                print(f"Frame {i}: Forced detection due to large change (MSE={mse_change:.2f})")

        # Decide saliency path
        status_msg = ""
        if is_keyframe or force_detection or last_full_saliency_map is None:
            # Full detection (U2Net + YOLO + Spectral)
            status_msg = "Full Detection"
            saliency_map = get_saliency_map(frame_path, weight_path)

            object_map = None
            if args.use_yolo:
                object_map = get_object_segmentation_map(frame_path)

            spectral_map = None
            if args.use_spectral:
                spectral_map = detect_spectral_residual(frame_path)

            # Fuse maps
            current_saliency_map = allocate_bits(
                saliency_map, object_map=object_map, spectral_map=spectral_map, threshold=args.saliency_threshold
            )

            if last_full_saliency_map is not None:
                # Apply temporal smoothing to prevent flickering
                current_saliency_map = temporal_smoothing(last_full_saliency_map, current_saliency_map)

            last_full_saliency_map = current_saliency_map

        else:
            # Propagation Path (Optical Flow Tracking)
            status_msg = "Optical Flow Tracking"
            flow = calculate_optical_flow(prev_frame, curr_frame)
            tracked_map = warp_saliency_map(last_full_saliency_map, flow)

            # Use tracked map as current map
            current_saliency_map = tracked_map
            last_full_saliency_map = current_saliency_map

        # Apply clarity reduction to the final saliency weights
        # This reduces the fidelity of the salient parts by 0.9x to save more bits
        current_saliency_map = current_saliency_map * args.clarity_reduction

        # Save a sample of saliency maps every 10 frames for visualization
        if i % 10 == 0:
            viz_path = os.path.join(saliency_viz_dir, f"map_{i:05d}.png")
            viz_img = Image.fromarray((current_saliency_map * 255).astype(np.uint8))
            viz_img.save(viz_path)
            print(f"Frame {i}: {status_msg} -> Map saved to {viz_path}")
        elif i % 5 == 0:
            print(f"Frame {i}: {status_msg}...")

        # 4. Apply compression
        # We need bit_weights for the layered_compression
        # But wait, bit_weights *is* current_saliency_map (already went through allocate_bits)
        # Note: layered_compression expects bit_weights which is the ACRD output

        # Need to re-save current frame as a temp image for compression logic
        # OR modify layered_compression to take numpy arrays.
        # For now, we reuse the existing frame_path.

        final_img, _, _ = layered_compression(
            frame_path,
            current_saliency_map,
            base_quality=args.base_quality,
            enhancement_quality=args.enhancement_quality
        )

        # 5. Save processed frame
        # We save as JPG for now to build the video, but could save as AVIF individually
        processed_frame_path = os.path.join(processed_frames_dir, f"frame_{i:05d}.jpg")
        final_img.save(processed_frame_path)

        prev_frame = curr_frame

    # 6. Reconstruct video
    print(f"--- Step 3: Reconstructing video from processed frames (FFmpeg H.265) ---")
    output_video_path = os.path.join(args.output_dir, f"{video_name}_compressed.mp4")
    frames_to_video_ffmpeg(processed_frames_dir, output_video_path, fps, crf=args.crf, preset=args.preset)

    # 7. Video Compression Metrics
    print("--- Video Compression Metrics ---")
    orig_size = os.path.getsize(args.input)
    comp_size = os.path.getsize(output_video_path)
    ratio = orig_size / comp_size if comp_size > 0 else 0

    print(f"Original Video Size: {orig_size / (1024*1024):.2f} MB")
    print(f"Compressed Video Size: {comp_size / (1024*1024):.2f} MB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print("--------------------------")

    print(f"Saved compressed video to {output_video_path}")

    # Optional: Clean up temp frames
    # import shutil
    # shutil.rmtree(temp_frames_dir)
    # shutil.rmtree(processed_frames_dir)

    overall_end_time = time.time()
    duration = overall_end_time - overall_start_time

    print(f"Total Processing Time: {duration:.2f} seconds")
    print(f"Average Speed: {duration/frame_count:.2f} seconds per frame")

    print("--- Video Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
