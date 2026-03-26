import cv2
import os


def extract_frames(video_path, output_dir):
    """
    Extracts all frames from a video file into a directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    return fps, frame_count


def frames_to_video_ffmpeg(frames_dir, output_video_path, fps, crf=32, preset='veryslow'):
    """
    Reconstructs a video from frames in a directory using FFmpeg with H.265.
    This provides massive compression by exploiting the blurry backgrounds.
    """
    import subprocess

    # Ensure the output path ends with .mp4
    if not output_video_path.endswith('.mp4'):
        output_video_path = os.path.splitext(output_video_path)[0] + '.mp4'

    # FFmpeg command:
    # -y: Overwrite output file
    # -framerate: Set input frame rate
    # -i: Input pattern (frame_%05d.jpg)
    # -c:v libx265: Use H.265 (HEVC) codec
    # -crf: Constant Rate Factor (higher = more compression)
    # -preset: Encoding speed (slower = better compression)
    # -pix_fmt yuv420p: Ensure compatibility with most players
    command = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%05d.jpg'),
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tune', 'grain',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]

    print(f"Running FFmpeg: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"FFmpeg reconstruction complete: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        # Fallback to OpenCV if FFmpeg fails
        print("Falling back to OpenCV VideoWriter...")
        frames_to_video(frames_dir, output_video_path, fps)


def frames_to_video(frames_dir, output_video_path, fps):
    """
    Reconstructs a video from frames in a directory.
    """
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png') or f.endswith('.jpg')])
    if not frames:
        return

    # Read the first frame to get the size
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    h, w, _ = first_frame.shape

    # Use libx264 codec for better compression and wider compatibility
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Old codec
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # If X264 is not available, try 'avc1' or 'mp4v'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Check if writer opened successfully
    if not out.isOpened():
        print("Warning: X264 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
