# YOLO Inference Quick Reference

Quick command reference for YOLO inference scripts.

## 🎯 Simple Inference

### Image
```bash
# Basic
python inference_simple.py --source image.png

# Custom confidence
python inference_simple.py --source image.png --conf 0.4

# Custom model
python inference_simple.py --source image.png --model best.pt

# No display
python inference_simple.py --source image.png --no-display
```

### Video
```bash
# Basic
python inference_simple.py --source video.mp4

# Custom settings
python inference_simple.py --source video.mp4 --conf 0.3 --model best.pt

# Save only (no display)
python inference_simple.py --source video.mp4 --no-display
```

### Webcam
```bash
# Default camera
python inference_simple.py --mode webcam

# Specific camera
python inference_simple.py --mode webcam --camera 1

# Custom confidence
python inference_simple.py --mode webcam --conf 0.4
```

---

## 🚀 Comprehensive Inference

### Basic
```bash
# Image
python inference_comprehensive.py --source image.png

# Video
python inference_comprehensive.py --source video.mp4
```

### Tracking
```bash
# Enable tracking
python inference_comprehensive.py --source video.mp4 --track

# Tracking + trajectories
python inference_comprehensive.py --source video.mp4 --track --save-trajectory

# Tracking + analytics
python inference_comprehensive.py --source video.mp4 --track --analytics
```

### Zones
```bash
# Single zone
python inference_comprehensive.py --source video.mp4 --zones "[[100,100,500,100,500,400,100,400]]"

# Multiple zones
python inference_comprehensive.py --source video.mp4 --zones "[[100,100,300,100,300,300,100,300],[400,200,600,200,600,400,400,400]]"
```

### Stationary Detection
```bash
python inference_comprehensive.py --source video.mp4 --track --stationary-alert
```

### Export
```bash
# JSON
python inference_comprehensive.py --source video.mp4 --export-json

# CSV
python inference_comprehensive.py --source video.mp4 --export-csv

# Both
python inference_comprehensive.py --source video.mp4 --export-json --export-csv
```

### Full Features
```bash
python inference_comprehensive.py \
  --source video.mp4 \
  --track \
  --save-trajectory \
  --analytics \
  --stationary-alert \
  --export-json \
  --export-csv
```

---

## 📦 Batch Processing

### Basic
```bash
# Process directory
python batch_inference.py --input-dir ./images

# Specific type
python batch_inference.py --input-dir ./videos --type video

# Pattern matching
python batch_inference.py --input-dir ./data --pattern "*.mp4"
```

### Recursive
```bash
python batch_inference.py --input-dir ./data --recursive
```

### Parallel
```bash
# 4 workers
python batch_inference.py --input-dir ./videos --workers 4

# 8 workers
python batch_inference.py --input-dir ./data --workers 8 --recursive
```

### With Report
```bash
python batch_inference.py --input-dir ./videos --report summary.csv
```

### Custom Output
```bash
python batch_inference.py --input-dir ./input --output-dir ./results
```

### Complete Example
```bash
python batch_inference.py \
  --input-dir ./footage \
  --output-dir ./results \
  --pattern "*.mp4" \
  --recursive \
  --workers 4 \
  --model best.pt \
  --conf 0.3 \
  --report summary.csv
```

---

## 🔧 Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Input file path | Required |
| `--model` | YOLO model path | `yolo11n.pt` |
| `--conf` | Confidence threshold | `0.25` |
| `--no-save` | Don't save output | Save enabled |
| `--no-display` | Don't display | Display enabled |

---

## 📊 Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `yolo11n.pt` | ⚡⚡⚡ | ⭐⭐ | Real-time, CPU |
| `yolo11s.pt` | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `yolo11m.pt` | ⚡ | ⭐⭐⭐⭐ | High accuracy |
| `best.pt` | Varies | ⭐⭐⭐⭐⭐ | Custom trained |

---

## 🎯 Confidence Thresholds

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| `0.1-0.2` | More detections | High recall needed |
| `0.25-0.4` | Balanced | General use |
| `0.5+` | Fewer, confident | High precision |

---

## 💡 Quick Tips

**Check GPU:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Check cameras:**
```bash
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**Get help:**
```bash
python inference_simple.py --help
python inference_comprehensive.py --help
python batch_inference.py --help
```

**View training results:**
```bash
python view_results.py
```

---

## 📁 Output Files

### Simple Inference
- Images: `image_detected.png`
- Videos: `video_detected.mp4`

### Comprehensive Inference
- Videos: `video_comprehensive.mp4`
- JSON: `video_results.json`
- CSV: `video_results.csv`

### Batch Processing
- Output dir: `batch_output/` (default)
- Report: `summary.csv` (if specified)

---

## 🚨 Troubleshooting

**Model not found:**
```bash
# Auto-download
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

**Slow on CPU:**
```bash
# Use nano model
--model yolo11n.pt
```

**Out of memory:**
```bash
# Reduce workers
--workers 1
```

**Webcam not working:**
```bash
# Try different camera ID
--camera 0  # or 1, 2, etc.
```

---

## 📚 See Also

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Complete documentation
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training documentation
- [QUICK_START.md](QUICK_START.md) - Project quick start
