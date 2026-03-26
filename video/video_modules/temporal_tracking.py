import cv2
import numpy as np


def calculate_optical_flow(prev_frame, curr_frame):
    """
    Calculates Farneback dense optical flow between two frames.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def warp_saliency_map(prev_map, flow):
    """
    Warps the previous saliency map using the calculated optical flow.
    """
    h, w = prev_map.shape[:2]
    # Create the meshgrid for warping
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply the flow to the grid
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    # Remap the previous saliency map to the new positions
    warped_map = cv2.remap(
        prev_map.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped_map


def calculate_frame_change(prev_frame, curr_frame, downsample_size=(64, 64)):
    """
    Calculates the global change metric between two frames using downsampled MSE.
    Downsampling helps in filtering out sensor noise and makes calculation nearly instant.
    """
    # Resize to tiny dimensions
    prev_tiny = cv2.resize(prev_frame, downsample_size, interpolation=cv2.INTER_AREA)
    curr_tiny = cv2.resize(curr_frame, downsample_size, interpolation=cv2.INTER_AREA)

    prev_gray = cv2.cvtColor(prev_tiny, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_tiny, cv2.COLOR_BGR2GRAY)

    mse = np.mean((prev_gray.astype(np.float32) - curr_gray.astype(np.float32)) ** 2)
    return mse


def temporal_smoothing(prev_map, curr_map, alpha=0.7):
    """
    Applies Exponential Moving Average (EMA) to smooth saliency maps over time.
    alpha: weight of the current map (0 to 1)
    """
    return alpha * curr_map + (1 - alpha) * prev_map
