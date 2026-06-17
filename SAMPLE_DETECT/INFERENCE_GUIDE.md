# YOLO Inference Guide - Parks Surveillance

Complete guide for running object detection inference on your parks surveillance dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Available Scripts](#available-scripts)
- [Getting Started](#getting-started)
- [Image Inference](#image-inference)
- [Video Inference](#video-inference)
- [Webcam Inference](#webcam-inference)
- [Advanced Features](#advanced-features)
- [Batch Processing](#batch-processing)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project includes three inference scripts with different capabilities:

| Script | Purpose | Best For |
|--------|---------|----------|
| `inference_simple.py` | Easy-to-use unified inference | Beginners, quick testing |
| `inference_comprehensive.py` | Advanced features & analytics | Production, detailed analysis |
| `batch_inference.py` | Process multiple files | Large datasets, automation |

**Legacy Scripts:**
- `Demo.py` - Basic image inference
- `detect_video.py` - Basic video inference

---

## Available Scripts

### 🎯 inference_simple.py

**Beginner-friendly unified inference script**

Features:
- ✅ Image inference
- ✅ Video inference  
- ✅ Webcam inference
- ✅ Auto-detection of input type
- ✅ Simple CLI interface
- ✅ Progress tracking

### 🚀 inference_comprehensive.py

**Advanced inference with professional features**

Features:
- ✅ Object tracking (ByteTrack)
- ✅ Zone detection
- ✅ Trajectory visualization
- ✅ Stationary object alerts
- ✅ Real-time analytics
- ✅ JSON/CSV export
- ✅ Multi-source support

### 📦 batch_inference.py

**Batch processing utility**

Features:
- ✅ Directory processing
- ✅ Recursive search
- ✅ Parallel processing
- ✅ Progress tracking
- ✅ Summary reports
- ✅ Error recovery

---

## Getting Started

### Prerequisites

Ensure you have the required packages installed:

```bash
pip install ultralytics opencv-python torch tqdm numpy
```

### Quick Test

Test with the sample image:

```bash
python inference_simple.py --source image.png --mode image
```

Test with the sample video:

```bash
python inference_simple.py --source dataset1.mp4 --mode video
```

---

## Image Inference

### Basic Image Inference

**Using inference_simple.py:**

```bash
# Auto-detect mode
python inference_simple.py --source image.png

# Explicit mode
python inference_simple.py --source image.png --mode image

# Custom confidence threshold
python inference_simple.py --source image.png --conf 0.4

# Don't display, just save
python inference_simple.py --source image.png --no-display
```

**Using inference_comprehensive.py:**

```bash
# Basic inference
python inference_comprehensive.py --source image.png

# With analytics
python inference_comprehensive.py --source image.png --analytics

# Export to JSON
python inference_comprehensive.py --source image.png --export-json
```

### Using Custom Models

Use your trained model:

```bash
# Use best model from training
python inference_simple.py --source image.png --model runs/train/parks_surveillance_training3/weights/best.pt

# Use specific checkpoint
python inference_simple.py --source image.png --model runs/detect/train/weights/best.pt
```

---

## Video Inference

### Basic Video Inference

**Using inference_simple.py:**

```bash
# Process video
python inference_simple.py --source dataset1.mp4

# Custom settings
python inference_simple.py --source dataset1.mp4 --conf 0.3 --no-display

# Save without displaying
python inference_simple.py --source dataset1.mp4 --no-display
```

### Advanced Video Inference

**Object Tracking:**

```bash
# Enable tracking
python inference_comprehensive.py --source dataset1.mp4 --track

# Tracking with trajectories
python inference_comprehensive.py --source dataset1.mp4 --track --save-trajectory

# Tracking with analytics
python inference_comprehensive.py --source dataset1.mp4 --track --analytics
```

**Zone Detection:**

Define zones as JSON arrays of polygon points `[[x1,y1,x2,y2,...]]`:

```bash
# Single zone (rectangular area)
python inference_comprehensive.py --source dataset1.mp4 --zones "[[100,100,500,100,500,400,100,400]]"

# Multiple zones
python inference_comprehensive.py --source dataset1.mp4 --zones "[[100,100,300,100,300,300,100,300],[400,200,600,200,600,400,400,400]]"
```

**Stationary Object Detection:**

```bash
# Alert for stationary objects
python inference_comprehensive.py --source dataset1.mp4 --track --stationary-alert
```

**Export Results:**

```bash
# Export to JSON
python inference_comprehensive.py --source dataset1.mp4 --track --export-json

# Export to CSV
python inference_comprehensive.py --source dataset1.mp4 --export-csv

# Both formats
python inference_comprehensive.py --source dataset1.mp4 --export-json --export-csv
```

---

## Webcam Inference

### Live Webcam Detection

**Using inference_simple.py:**

```bash
# Default camera (camera 0)
python inference_simple.py --mode webcam

# Specific camera
python inference_simple.py --mode webcam --camera 1

# Custom confidence
python inference_simple.py --mode webcam --conf 0.4
```

**Using inference_comprehensive.py:**

```bash
# Webcam with tracking
python inference_comprehensive.py --source 0 --track --analytics
```

> **Note:** Press 'q' to quit webcam mode

---

## Advanced Features

### Object Tracking

Track objects across video frames with unique IDs:

```bash
python inference_comprehensive.py --source dataset1.mp4 --track
```

**Features:**
- Unique ID for each object
- Maintains identity across frames
- Track statistics (first seen, last seen)

### Trajectory Visualization

Visualize object movement paths:

```bash
python inference_comprehensive.py --source dataset1.mp4 --track --save-trajectory
```

**Visualization:**
- Green dot: Start position
- Red dot: Current position
- Line: Movement path

### Zone Detection

Monitor specific areas in the frame:

```bash
# Define a rectangular zone
python inference_comprehensive.py --source dataset1.mp4 --zones "[[100,100,500,100,500,400,100,400]]"
```

**Use Cases:**
- Restricted area monitoring
- Entry/exit counting
- Zone-specific alerts

### Stationary Object Alerts

Detect objects that remain in one place:

```bash
python inference_comprehensive.py --source dataset1.mp4 --track --stationary-alert
```

**Use Cases:**
- Abandoned object detection
- Parking violation detection
- Loitering detection

### Analytics Dashboard

Real-time statistics overlay:

```bash
python inference_comprehensive.py --source dataset1.mp4 --analytics
```

**Displays:**
- Current frame number
- Objects in current frame
- Total detections
- Active tracks

### Export Capabilities

**JSON Export:**

```bash
python inference_comprehensive.py --source dataset1.mp4 --track --export-json
```

Output: `dataset1_results.json`

```json
{
  "source": "dataset1.mp4",
  "frames": 300,
  "statistics": {
    "total_detections": 1250,
    "class_counts": {
      "person": 800,
      "car": 350,
      "bicycle": 100
    }
  },
  "tracks": {
    "1": {
      "class": "person",
      "first_seen": 1,
      "last_seen": 150,
      "trajectory_length": 50
    }
  }
}
```

**CSV Export:**

```bash
python inference_comprehensive.py --source dataset1.mp4 --export-csv
```

Output: `dataset1_results.csv`

```csv
Frame,Detections
1,5
2,6
3,4
...
```

---

## Batch Processing

### Process Multiple Files

**Process all images in a directory:**

```bash
python batch_inference.py --input-dir ./images --type image
```

**Process all videos:**

```bash
python batch_inference.py --input-dir ./videos --type video
```

**Process with pattern:**

```bash
# Only MP4 files
python batch_inference.py --input-dir ./videos --pattern "*.mp4"

# Only JPG images
python batch_inference.py --input-dir ./images --pattern "*.jpg"
```

### Recursive Processing

Search in subdirectories:

```bash
python batch_inference.py --input-dir ./data --recursive
```

### Parallel Processing

Use multiple workers for faster processing:

```bash
# 4 parallel workers
python batch_inference.py --input-dir ./videos --workers 4

# 8 workers for large datasets
python batch_inference.py --input-dir ./data --workers 8 --recursive
```

> **Note:** Optimal workers = number of CPU cores

### Generate Reports

Create CSV summary report:

```bash
python batch_inference.py --input-dir ./videos --report summary.csv
```

Report includes:
- File path
- Processing status
- Number of detections
- Number of frames
- Output path
- Errors (if any)

### Custom Output Directory

```bash
python batch_inference.py --input-dir ./input --output-dir ./results
```

### Complete Example

```bash
# Process all videos recursively with 4 workers, custom model, and report
python batch_inference.py \
  --input-dir ./surveillance_footage \
  --output-dir ./processed_results \
  --pattern "*.mp4" \
  --recursive \
  --workers 4 \
  --model runs/train/parks_surveillance_training3/weights/best.pt \
  --conf 0.3 \
  --report batch_summary.csv
```

---

## Troubleshooting

### Common Issues

**1. "Model not found" error**

```bash
# Make sure model file exists
ls yolo11n.pt

# Or download it (will auto-download on first use)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

**2. "Video not found" error**

```bash
# Check file path
ls dataset1.mp4

# Use absolute path
python inference_simple.py --source "C:/Users/manth/Downloads/SAMPLE_DETECT/dataset1.mp4"
```

**3. Slow processing on CPU**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, consider:
# - Installing CUDA-enabled PyTorch
# - Using smaller model (yolo11n.pt)
# - Reducing video resolution
```

**4. Out of memory errors**

```bash
# Use smaller model
python inference_simple.py --source video.mp4 --model yolo11n.pt

# Reduce batch size (for batch processing)
python batch_inference.py --input-dir ./videos --workers 1
```

**5. Webcam not opening**

```bash
# Try different camera IDs
python inference_simple.py --mode webcam --camera 0
python inference_simple.py --mode webcam --camera 1

# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**6. Display window not showing**

```bash
# Make sure you're not using --no-display
python inference_simple.py --source image.png

# For remote servers, use --no-display and check saved files
python inference_simple.py --source image.png --no-display
```

### Performance Tips

**1. Use GPU if available**
- Install CUDA-enabled PyTorch
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**2. Choose appropriate model size**
- `yolo11n.pt` - Fastest, lowest accuracy
- `yolo11s.pt` - Balanced
- `yolo11m.pt` - Better accuracy, slower
- `yolo11l.pt` - High accuracy, much slower

**3. Optimize confidence threshold**
- Lower (0.1-0.2): More detections, more false positives
- Medium (0.25-0.4): Balanced
- Higher (0.5+): Fewer detections, higher precision

**4. Batch processing optimization**
- Use workers = CPU cores
- Process videos separately from images
- Use pattern matching to filter files

### Getting Help

**Check script help:**

```bash
python inference_simple.py --help
python inference_comprehensive.py --help
python batch_inference.py --help
```

**View results:**

```bash
# View training results
python view_results.py
```

---

## Next Steps

1. **Train custom model** - See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. **View training results** - Run `python view_results.py`
3. **Optimize performance** - Experiment with different models and thresholds
4. **Automate workflows** - Use batch processing for large datasets

---

## Quick Reference

See [INFERENCE_QUICK_REF.md](INFERENCE_QUICK_REF.md) for a condensed command reference.
