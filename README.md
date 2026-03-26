# Saliency-Based Compression

This project focuses on image and video compression techniques that leverage saliency detection to achieve higher compression ratios while preserving perceptual quality in important regions.

## Description

The core idea is to identify salient regions in an image or video and allocate more bits to those regions during compression, while compressing the less salient background more aggressively. This allows for a better trade-off between file size and visual quality.

The project includes modules for:
- Saliency detection (using spectral analysis and object detection)
- Bit allocation based on saliency maps
- Image and video compression

## Getting Started

### Prerequisites

- Python 3.x
- Dependencies from `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/s-kailash/Saliency_Compression.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to perform compression on an image:
```bash
python main.py --input <path_to_image>
```
