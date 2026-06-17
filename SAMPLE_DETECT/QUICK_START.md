# YOLO Parks Surveillance - Quick Start Guide

Get started with training and inference for your parks surveillance YOLO project in minutes!

## 📋 What You Have

### Training Scripts
- **`train_simple.py`** - Beginner-friendly training (recommended for first-time users)
- **`train_comprehensive.py`** - Advanced training with multiple modes

### Inference Scripts
- **`inference_simple.py`** - Easy inference for images/videos/webcam
- **`inference_comprehensive.py`** - Advanced inference with tracking & analytics
- **`batch_inference.py`** - Process multiple files at once

### Utilities
- **`view_results.py`** - View training results and metrics

### Documentation
- **`TRAINING_GUIDE.md`** - Complete training documentation
- **`INFERENCE_GUIDE.md`** - Complete inference documentation
- **`TRAINING_QUICK_REF.md`** - Training command reference
- **`INFERENCE_QUICK_REF.md`** - Inference command reference

---

## 🎯 Your Dataset

**Location:** `data/data.yaml`

**Classes (11 total):**
1. Bag
2. Bench
3. Bicycle
4. Building
5. Cap
6. Chalkpiece
7. Drawing
8. Tree
9. car
10. person
11. road

**Dataset Split:**
- Training: 30 images
- Validation: 3 images
- Test: 6 images

---

## 🚀 Quick Start - Training

### Step 1: Simple Training (Easiest)

```bash
python train_simple.py
```

This will:
- Train for 25 epochs
- Use YOLO11n (nano) model
- Save results to `runs/train/parks_surveillance_training/`
- Show progress and metrics

### Step 2: Advanced Training (More Control)

```bash
# Quick test (10 epochs)
python train_comprehensive.py --mode quick

# Standard training (50 epochs) - Recommended
python train_comprehensive.py --mode standard

# High accuracy (100 epochs)
python train_comprehensive.py --mode high-accuracy
```

### Step 3: View Results

```bash
python view_results.py
```

---

## 🔍 Quick Start - Inference

### Image Inference

```bash
# Basic
python inference_simple.py --source image.png

# With custom model
python inference_simple.py --source image.png --model runs/train/parks_surveillance_training3/weights/best.pt
```

### Video Inference

```bash
# Basic
python inference_simple.py --source dataset1.mp4

# With tracking
python inference_comprehensive.py --source dataset1.mp4 --track --analytics
```

### Webcam Inference

```bash
python inference_simple.py --mode webcam
```

### Batch Processing

```bash
# Process all images in a folder
python batch_inference.py --input-dir ./images --type image

# Process videos with 4 workers
python batch_inference.py --input-dir ./videos --workers 4 --report summary.csv
```

---

## 📁 Project Structure

```
SAMPLE_DETECT/
├── 🎓 Training
│   ├── train_simple.py
│   └── train_comprehensive.py
│
├── 🔍 Inference
│   ├── inference_simple.py
│   ├── inference_comprehensive.py
│   └── batch_inference.py
│
├── 📊 Data & Results
│   ├── data/                    # Your dataset
│   ├── runs/                    # Training & validation outputs
│   │   ├── train/              # Training runs
│   │   └── detect/             # Detection/validation runs
│   └── inference_results/       # Inference outputs
│
└── 📚 Documentation
    ├── README.md
    ├── TRAINING_GUIDE.md
    ├── INFERENCE_GUIDE.md
    └── *_QUICK_REF.md files
```

---

## 📊 After Training

### Find Your Model

After training, your best model will be at:
```
runs/train/parks_surveillance_training/weights/best.pt
```

### Use Your Model

```bash
# For inference
python inference_simple.py --source video.mp4 --model runs/train/parks_surveillance_training3/weights/best.pt

# For batch processing
python batch_inference.py --input-dir ./test --model runs/train/parks_surveillance_training3/weights/best.pt
```

---

## ⚙️ Configuration

### Model Sizes
- `yolo11n.pt` - Nano (fastest) ⚡
- `yolo11s.pt` - Small (balanced) ⚖️
- `yolo11m.pt` - Medium (better accuracy) 🎯
- `yolo11l.pt` - Large (high accuracy) 🔥

### Confidence Thresholds
- `0.1-0.2` - High recall (more detections)
- `0.25-0.4` - Balanced (recommended)
- `0.5+` - High precision (fewer, confident detections)

---

## 🔧 Common Commands

### Training
```bash
# Quick test
python train_simple.py

# Full training
python train_comprehensive.py --mode standard --epochs 50
```

### Inference
```bash
# Image
python inference_simple.py --source image.png

# Video
python inference_simple.py --source video.mp4

# Webcam
python inference_simple.py --mode webcam

# With tracking
python inference_comprehensive.py --source video.mp4 --track
```

### Batch Processing
```bash
# Process directory
python batch_inference.py --input-dir ./images --type image

# With report
python batch_inference.py --input-dir ./videos --report summary.csv
```

---

## 🚨 Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
- Reduce batch size in training scripts
- Use smaller model (`yolo11n.pt`)
- Reduce image size

### Slow Training
- Use GPU if available
- Use `yolo11n.pt` (nano model)
- Reduce epochs for testing

### Low Detection Rate
- Lower confidence threshold: `--conf 0.1`
- Use your trained model instead of pretrained
- Check if objects are in training classes

---

## 📖 Next Steps

1. **Start with simple training:**
   ```bash
   python train_simple.py
   ```

2. **Test inference:**
   ```bash
   python inference_simple.py --source dataset1.mp4
   ```

3. **Read detailed guides:**
   - Training: `TRAINING_GUIDE.md`
   - Inference: `INFERENCE_GUIDE.md`

4. **Process your data:**
   ```bash
   python batch_inference.py --input-dir ./your_data
   ```

---

## 💡 Recommended Workflow

**For Beginners:**
1. Run `python train_simple.py`
2. Wait for training to complete
3. Run `python inference_simple.py --source dataset1.mp4`
4. Check results in `inference_results/`

**For Advanced Users:**
1. Run `python train_comprehensive.py --mode standard`
2. Run `python inference_comprehensive.py --source video.mp4 --track --analytics`
3. Use `batch_inference.py` for multiple files
4. Export results with `--export-json` or `--export-csv`

---

## ✅ You're Ready!

Everything is set up and ready to use:
- ✅ Dataset properly formatted
- ✅ Training scripts ready
- ✅ Inference scripts ready
- ✅ Documentation complete

**Start training now:**
```bash
python train_simple.py
```

Good luck! 🚀

---

For detailed documentation, see:
- **Training:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Inference:** [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Quick Reference:** `*_QUICK_REF.md` files
