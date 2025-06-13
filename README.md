# Lane and Obstacle Detection System

An advanced lane detection and obstacle recognition system for autonomous vehicles, implemented using YOLO algorithm and OpenCV image processing techniques.

## 📖 Research Paper

This project is based on the following research paper:
[Proposing Lane and Obstacle Detection Algorithm Using YOLO to Control Self-Driving Cars on Advanced Networks](https://www.researchgate.net/publication/360958173_Proposing_Lane_and_Obstacle_Detection_Algorithm_Using_YOLO_to_Control_Self-Driving_Cars_on_Advanced_Networks)

## ✨ Features

- **Lane Detection**: Sliding Window technique with advanced image processing
- **Obstacle Detection**: YOLO v4 implementation for multi-object detection
- **Curvature Calculation**: Precise road curvature radius computation
- **Real-time Display**: Live FPS and results visualization
- **Video & Camera Support**: Process video files or webcam input

## 🛠️ Installation & Setup

### Requirements

- Python 3.7+
- OpenCV
- NumPy
- Webcam or video file for testing

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv lane_detection_env

# Activate virtual environment
# Windows:
lane_detection_env\Scripts\activate
# Linux/macOS:
source lane_detection_env/bin/activate
```

### Install Dependencies

```bash
pip install opencv-python
pip install numpy
pip install argparse
```

### Download YOLO Model Files

For complete system functionality, download the following files and place them in the project root directory:

1. **Configuration file (cfg)**:
   ```bash
   wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg
   ```

2. **Weights file**:
   ```bash
   wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   ```

3. **Class names file**: The `coco.names` file should also be present in the root directory.

## 🚀 Usage

### Run with Webcam

```bash
python lane_detection.py --model_cfg yolov4.cfg --model_weights yolov4.weights --src 0
```

### Run with Video File

```bash
python lane_detection.py --model_cfg yolov4.cfg --model_weights yolov4.weights --video path/to/your/video.mp4
```

### Command Line Arguments

- `--model_cfg`: Path to YOLO configuration file
- `--model_weights`: Path to YOLO weights file  
- `--video`: Path to input video file (optional)
- `--src`: Camera source index (default: 0)
- `--output_dir`: Output directory path (optional)

## 📁 Project Structure

```
lane-detection/
│
├── lane_detection.py     # Main execution file
├── utils.py              # Image processing utility functions
├── yolov4.cfg           # YOLO configuration file
├── yolov4.weights       # YOLO model weights
├── coco.names           # COCO class names
├── cal_pickle.p         # Camera calibration file (optional)
└── README.md            # Project documentation
```

## 🔧 Configuration

### Processing Parameters

You can modify the following parameters in the code:

```python
FRAME_WIDTH = 640           # Frame width
FRAME_HEIGHT = 480          # Frame height
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold
NMS_THRESHOLD = 0.3         # Non-Maximum Suppression threshold
```

### Perspective Transform Adjustment

Use the "Trackbars" window to adjust perspective transform points:
- Width Top/Bottom: Top and bottom width adjustments
- Height Top/Bottom: Top and bottom height adjustments

## 📸 Display Windows

The system shows three display windows:

1. **Object Detection**: YOLO obstacle detection results
2. **Pipeline**: Various image processing stages
3. **Lane Detection Result**: Final lane detection output

<div align="center">
  <img src="https://github.com/SepehrNorouzi7/Lane-Detection/blob/main/screenshots/screenshot-1.jpg" alt="Pipeline Process" width="45%" />
  <img src="https://github.com/SepehrNorouzi7/Lane-Detection/blob/main/screenshots/screenshot-2.jpg" alt="Final Result" width="45%" />
</div>
<div align="center">
  <em>Left: Various stages of image processing pipeline | Right: Final lane and obstacle detection results</em>
</div>

## 🎯 Performance

- **Lane Detection Accuracy**: 90%+ under good lighting conditions
- **Processing Speed**: 15-30 FPS depending on hardware
- **Obstacle Detection**: Recognizes 80+ different object classes

## 🔍 Technical Details

### Lane Detection Algorithm

1. **Preprocessing**: Camera distortion removal, color filtering
2. **Perspective Transform**: Bird's eye view conversion
3. **Sliding Window**: Lane pixel detection
4. **Polynomial Fitting**: 2nd degree polynomial fitting
5. **Smoothing**: Moving average for noise reduction

### YOLO v4 Model

- **Architecture**: CSPDarknet53 backbone
- **Training Data**: COCO dataset
- **Detectable Classes**: 80 different classes
- **Accuracy**: mAP 43.5% on COCO dataset

## 🐛 Troubleshooting

### Common Issues

1. **"cal_pickle.p not found" error**: 
   - Camera calibration file is missing (this is normal)

2. **Low FPS**:
   - Reduce input resolution
   - Use GPU acceleration for YOLO

3. **Incorrect lane detection**:
   - Adjust perspective transform trackbars
   - Improve lighting conditions

---

**Note**: Make sure the files `yolov4.cfg`, `yolov4.weights`, and `coco.names` are present in the project root directory.
