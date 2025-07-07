# MotionFX - Simple Motion Blur 🌊

A simple Python package for creating motion blur effects on images. Used in research for clinical hand movement analysis and video deblurring applications.

## 📄 Research Paper

This tool was used in the research paper:
**"Parallel scale de-blur net for sharpening video images for remote clinical assessment of hand movements"**
*Expert Systems with Applications*, 2023
https://doi.org/10.1016/j.eswa.2023.121093

### Citation
```bibtex
@article{li2024parallel,
  title={Parallel scale de-blur net for sharpening video images for remote clinical assessment of hand movements},
  author={Li, Renjie and Huang, Guan and Wang, Xinyi and Chen, Yanyu and Tran, Son N and Garg, Saurabh and St George, Rebecca J and Lawler, Katherine and Alty, Jane and Bai, Quan},
  journal={Expert Systems with Applications},
  volume={235},
  pages={121093},
  year={2024},
  publisher={Elsevier}
}
```

## Installation

```bash
git clone https://github.com/54hg0220/motionfx.git
cd motionfx
pip install .
```

## Usage

### 🎯 Single Image Motion Blur

```bash
# Average motion blur
motionfx blur avg -i input.jpg -o output.jpg -v "20,5" -f 10

# Flow-based blur
motionfx blur flow -i input.jpg -o output.jpg -v "1,0" -s 15

# Gaussian weighted blur (natural looking)
motionfx blur gaussian -i input.jpg -o output.jpg -v "15,0" -f 12

# Light trails effect
motionfx blur max -i input.jpg -o output.jpg -v "25,0" -f 15
```

### 📊 Dataset Creation

**Create motion blur datasets for training deblurring models:**

```bash
# Create general motion blur dataset
motionfx create-dataset -i ./sharp_images/ -o ./blur_dataset/ -n 5

# Create clinical hand movement dataset (as used in the paper)
motionfx create-dataset -i ./hand_images/ -o ./clinical_dataset/ -n 5 --clinical

# Process video into dataset frames
motionfx create-dataset --video hand_movement.mp4 -o ./video_dataset/ -n 3 --frame-interval 30
```

This creates a structured dataset:
```
blur_dataset/
├── train/
│   ├── sharp/          # Original sharp images
│   └── blurred/        # Motion blurred variants
├── val/
└── test/
```

## Methods

- **avg**: Simple average of motion frames
- **flow**: Flow-based directional blur  
- **gaussian**: Gaussian weighted blur (most natural)
- **linear**: Linear weighted blur (center-focused)
- **max**: Maximum value blur (light trails)
- **weighted**: Custom weighted blur

## Dataset Creation Features

### 🏥 Clinical Hand Movement Dataset
Specifically designed for medical applications:
- Realistic hand motion blur patterns (5-25 pixel displacement)
- Multiple blur methods per image
- Ready for training deblurring neural networks
- Train/validation/test splits included

### 📹 Video Processing
- Extract frames from clinical videos
- Automatic motion blur generation
- Suitable for remote assessment applications

### ⚙️ Customizable Parameters
- Motion vector ranges
- Number of blur variants
- Different blur techniques
- Training/validation splits

## Examples

```bash
# Create dataset for hand movement deblurring research
motionfx create-dataset -i ./hand_videos/ -o ./research_dataset/ --clinical -n 8

# Process single image with specific motion
motionfx blur gaussian -i hand_photo.jpg -o blurred_hand.jpg -v "12,8" -f 15

# Create light streak effects
motionfx blur max -i night_scene.jpg -o light_trails.jpg -v "40,5" -f 20

# Custom weighted blur for artistic effects  
motionfx blur weighted -i photo.jpg -o artistic.jpg -v "15,3" -w "0.1,0.2,0.4,0.2,0.1"
```

## Research Applications

This tool is particularly useful for:
- **Clinical Assessment**: Creating training data for hand movement analysis
- **Video Deblurring**: Generating paired sharp/blurred datasets
- **Remote Healthcare**: Processing video data for medical diagnosis
- **Computer Vision Research**: Motion blur simulation and analysis

## Parameters

### Blur Commands
- `-i, --input`: Input image path
- `-o, --output`: Output image path  
- `-v, --vector`: Motion vector "x,y" (pixels)
- `-f, --frames`: Number of frames (default: 10)
- `-s, --strength`: Blur strength for flow method (default: 10)
- `-w, --weights`: Custom weights for weighted method

### Dataset Commands
- `-i, --input`: Input directory with sharp images
- `-o, --output`: Output dataset directory
- `-n, --variants`: Number of blur variants per image
- `--clinical`: Use clinical hand movement parameters
- `--video`: Process video file instead of images
- `--frame-interval`: Extract every N frames from video

## Requirements

- Python >= 3.7
- NumPy
- OpenCV
- Pillow


### 🐍 Python API - TODO

```python
import motionfx

# Single image blur
image = motionfx.load_image("input.jpg")
result = motionfx.avg_blur(image, motion_vector=(20, 5), num_frames=10)
motionfx.save_image(result, "output.jpg")

# Create dataset
generator = motionfx.MotionBlurDatasetGenerator("my_dataset")
generator.process_directory("./sharp_images/", num_variants=5)
generator.create_training_splits()

# Clinical hand movement dataset
motionfx.create_clinical_hand_dataset("./hand_images/", "./clinical_dataset/")
```

## Contributing

**Author**: Guan Huang  
**Email**: huangguan0220@gmail.com

If you use this tool in your research, please cite our paper:
https://doi.org/10.1016/j.eswa.2023.121093