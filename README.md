# Parks Surveillance - YOLO Object Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parks-surveillance-yolo-object-detection-system.streamlit.app/)

A comprehensive YOLO-based object detection system for parks surveillance with training, inference, and batch processing capabilities.

## 🎯 Project Overview

This project provides a complete pipeline for:
- **Training** custom YOLO models on parks surveillance data
- **Inference** on images, videos, and webcam feeds
- **Batch processing** for multiple files
- **Advanced features** like object tracking, zone detection, and analytics

## 📁 Project Structure

```
SAMPLE_DETECT/
├── 🎓 Training Scripts
│   ├── train_simple.py              # Simple training script
│   └── train_comprehensive.py       # Advanced training with multiple modes
│
├── 🔍 Inference Scripts
│   ├── inference_simple.py          # Unified inference (image/video/webcam)
│   ├── inference_comprehensive.py   # Advanced inference with tracking
│   ├── batch_inference.py           # Batch processing utility
│   ├── Demo.py                      # Legacy image detection
│   └── detect_video.py              # Legacy video detection
│
├── 📊 Utilities
│   └── view_results.py              # View training results
│
├── 📚 Documentation
│   ├── QUICK_START.md               # Quick start guide
│   ├── TRAINING_GUIDE.md            # Training documentation
│   ├── TRAINING_QUICK_REF.md        # Training quick reference
│   ├── INFERENCE_GUIDE.md           # Inference documentation
│   └── INFERENCE_QUICK_REF.md       # Inference quick reference
│
├── 📁 Data & Results
│   ├── data/                        # Dataset (train/valid/test)
│   ├── runs/                        # Training & validation outputs
│   └── inference_results/           # Inference outputs
│
└── 📦 Models
    └── yolo11n.pt                   # Pre-trained YOLO model
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Simple training
python train_simple.py

# Advanced training with options
python train_comprehensive.py --mode standard --epochs 50
```

### 3. Inference

```bash
# Image inference
python inference_simple.py --source image.png

# Video inference
python inference_simple.py --source video.mp4

# Webcam inference
python inference_simple.py --mode webcam

# Advanced inference with tracking
python inference_comprehensive.py --source video.mp4 --track --analytics
```

### 4. Batch Processing

```bash
# Process all images in a directory
python batch_inference.py --input-dir ./images --type image

# Process videos with multiple workers
python batch_inference.py --input-dir ./videos --workers 4 --report summary.csv
```

## ✨ Features

### Training Features
- ✅ Simple beginner-friendly training
- ✅ Advanced training with multiple modes (quick/standard/high-accuracy)
- ✅ Resume training from checkpoints
- ✅ Fine-tuning capabilities
- ✅ Automatic model evaluation
- ✅ Training visualization and metrics

### Inference Features
- ✅ Image, video, and webcam inference
- ✅ Object tracking with unique IDs
- ✅ Zone-based detection
- ✅ Trajectory visualization
- ✅ Stationary object alerts
- ✅ Real-time analytics dashboard
- ✅ JSON/CSV export
- ✅ Batch processing with parallel workers

## 🎓 Dataset

**Classes (11 total):**
- Bag, Bench, Bicycle, Building, Cap
- Chalkpiece, Drawing, Tree
- car, person, road

**Dataset Split:**
- Training: 30 images
- Validation: 3 images
- Test: 6 images

## 📊 Model Performance

Best trained model: `runs/train/parks_surveillance_training3/weights/best.pt`

## 🎯 Usage Examples

### Training

```python
# Quick training for testing
python train_comprehensive.py --mode quick --epochs 10

# Standard training
python train_comprehensive.py --mode standard --epochs 50

# High accuracy training
python train_comprehensive.py --mode high-accuracy --epochs 100
```

### Inference

```python
# Simple inference
python inference_simple.py --source dataset1.mp4 --conf 0.3

# Comprehensive inference with all features
python inference_comprehensive.py \
  --source dataset1.mp4 \
  --track \
  --save-trajectory \
  --analytics \
  --export-json
```

### Batch Processing

```python
# Process entire directory
python batch_inference.py \
  --input-dir ./test_videos \
  --output-dir ./results \
  --workers 4 \
  --report summary.csv
```

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get started quickly
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training guide
- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete inference guide
- **[TRAINING_QUICK_REF.md](TRAINING_QUICK_REF.md)** - Training command reference
- **[INFERENCE_QUICK_REF.md](INFERENCE_QUICK_REF.md)** - Inference command reference

## 🔧 Configuration

### Confidence Threshold
- **0.1-0.2**: High recall, more false positives
- **0.25-0.4**: Balanced (recommended)
- **0.5+**: High precision, fewer detections

### Model Selection
- `yolo11n.pt` - Fastest, good for real-time
- `yolo11s.pt` - Balanced speed/accuracy
- `yolo11m.pt` - Better accuracy
- `best.pt` - Your custom trained model

## 📦 Requirements

```
ultralytics
opencv-python
torch
tqdm
numpy
```

## 🎬 Sample Results

**Video Inference:**
- Total detections: 2,507 (dataset1.mp4)
- Average: 11.88 objects per frame
- Classes detected: person, backpack, skateboard, handbag

**Object Tracking:**
- 52 unique tracked objects
- Maintained IDs across 211 frames

## 🔍 Advanced Features

### Object Tracking
```bash
python inference_comprehensive.py --source video.mp4 --track
```

### Zone Detection
```bash
python inference_comprehensive.py --source video.mp4 \
  --zones "[[100,100,500,100,500,400,100,400]]"
```

### Stationary Object Detection
```bash
python inference_comprehensive.py --source video.mp4 \
  --track --stationary-alert
```

### Export Results
```bash
python inference_comprehensive.py --source video.mp4 \
  --export-json --export-csv
```

## 🚨 Troubleshooting

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Webcam not opening:**
```bash
# Try different camera IDs
python inference_simple.py --mode webcam --camera 1
```

**Low detection rate:**
```bash
# Lower confidence threshold
python inference_simple.py --source video.mp4 --conf 0.1
```

## 📈 Performance Tips

1. **Use GPU** for faster training/inference
2. **Choose appropriate model size** based on speed/accuracy needs
3. **Adjust confidence threshold** for your use case
4. **Use batch processing** for multiple files
5. **Enable parallel workers** for faster batch processing

## 🤝 Contributing

This project is designed for parks surveillance but can be adapted for:
- Traffic monitoring
- Retail analytics
- Security systems
- Wildlife monitoring
- Sports analytics

## 📄 License

This project uses the Ultralytics YOLO model. See [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Dataset: CC BY 4.0 License

## 🙏 Acknowledgments

- **YOLO**: Ultralytics YOLO11
- **Dataset**: Roboflow Parks Surveillance Dataset
- **Framework**: PyTorch, OpenCV

## 📞 Support

For detailed documentation, see the guides in the project:
- Training issues → `TRAINING_GUIDE.md`
- Inference issues → `INFERENCE_GUIDE.md`
- Quick commands → `*_QUICK_REF.md` files

---

**Built with ❤️ for Parks Surveillance**
