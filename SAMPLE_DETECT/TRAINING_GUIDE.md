# YOLO Training Guide - Parks Surveillance

## Quick Start (5 Minutes)

### Option 1: Simple Training (Recommended for Beginners)
```bash
python train_simple.py
```
This will train a YOLO model with default settings. Just run it and wait!

### Option 2: Comprehensive Training (More Control)
```bash
# Quick training (testing - 50 epochs)
python train_comprehensive.py --mode quick

# Standard training (recommended - 100 epochs)
python train_comprehensive.py --mode standard

# Production training (best accuracy - 200 epochs)
python train_comprehensive.py --mode production
```

## Training Scripts Overview

### 1. `train_simple.py` ⭐ **Start Here!**
- **Best for**: Beginners, quick start
- **Features**: Minimal configuration, just run and train
- **Training time**: ~1-2 hours (GPU) or 4-6 hours (CPU)

### 2. `train_comprehensive.py` 🚀 **Advanced**
- **Best for**: Advanced users, production use
- **Features**: 
  - Multiple training modes (quick/standard/production)
  - Model evaluation and export
  - Comprehensive reporting
  - Custom configuration support
- **Usage**:
  ```bash
  # See all options
  python train_comprehensive.py --help
  
  # Custom training
  python train_comprehensive.py --mode standard --epochs 150 --batch 8
  
  # Evaluate existing model
  python train_comprehensive.py --evaluate path/to/best.pt
  
  # Export model to ONNX
  python train_comprehensive.py --export path/to/best.pt --formats onnx
  ```

### 3. `train_yolo.py` 📝 **Original**
- Basic training script with good documentation
- Good for learning YOLO training concepts

### 4. `train_yolo_advanced.py` 🔧 **Feature-Rich**
- Multiple training modes with CLI support
- Fine-tuning and resume capabilities
- Interactive mode available

## Dataset Information

**Location**: `data/data.yaml`

**Classes (11 total)**:
- Bag, Bench, Bicycle, Building, Cap
- Chalkpiece, Drawing, Tree, car, person, road

**Structure**:
```
data/
├── train/images/    # Training images
├── train/labels/    # Training labels (YOLO format)
├── valid/images/    # Validation images
├── valid/labels/    # Validation labels
├── test/images/     # Test images
├── test/labels/     # Test labels
└── data.yaml        # Dataset configuration
```

## Training Modes Comparison

| Mode | Model | Epochs | Time (GPU) | Accuracy | Best For |
|------|-------|--------|------------|----------|----------|
| Quick | nano | 50 | ~30 min | Good | Testing, experimentation |
| Standard | small | 100 | ~1-2 hours | Better | General use, balanced |
| Production | medium | 200 | ~4-6 hours | Best | Deployment, high accuracy |

## Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n.pt | 5 MB | Fastest | Good | Real-time, CPU, mobile |
| yolo11s.pt | 18 MB | Fast | Better | Balanced performance |
| yolo11m.pt | 40 MB | Medium | High | GPU deployment |
| yolo11l.pt | 50 MB | Slow | Higher | High accuracy needed |
| yolo11x.pt | 114 MB | Slowest | Highest | Maximum accuracy |

## After Training

### 1. Find Your Trained Model
```
runs/train/
└── parks_surveillance_training/
    └── weights/
        ├── best.pt    # ⭐ Use this for detection!
        └── last.pt    # Last checkpoint (for resuming)
```

### 2. Use Your Model for Detection
Update `detect_video.py`:
```python
model_path = "runs/train/parks_surveillance_training3/weights/best.pt"
```

Then run:
```bash
python detect_video.py
```

### 3. View Training Results
Training generates several useful plots in the experiment directory:
- `results.png` - Training curves (loss, mAP, etc.)
- `confusion_matrix.png` - Confusion matrix
- `F1_curve.png` - F1 score curve
- `PR_curve.png` - Precision-Recall curve

## Common Issues & Solutions

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
python train_comprehensive.py --mode standard --batch 8
# or even smaller
python train_comprehensive.py --mode standard --batch 4
```

### Issue: Training Too Slow
**Solutions**:
1. Use smaller model: `--model yolo11n.pt`
2. Reduce epochs: `--epochs 50`
3. Use GPU if available
4. Reduce image size: `--imgsz 320`

### Issue: Poor Accuracy
**Solutions**:
1. Train longer: `--epochs 200`
2. Use larger model: `--model yolo11m.pt`
3. Increase image size: `--imgsz 1280`
4. Check dataset quality

## Hardware Requirements

**Minimum** (CPU Training):
- RAM: 8 GB
- Storage: 10 GB
- Time: 4-8 hours

**Recommended** (GPU Training):
- GPU: NVIDIA GPU with 6+ GB VRAM
- RAM: 16 GB
- Storage: 20 GB
- Time: 1-2 hours

**Optimal** (Fast GPU Training):
- GPU: NVIDIA GPU with 12+ GB VRAM (RTX 3080+)
- RAM: 32 GB
- Storage: 50 GB
- Time: 30-60 minutes

## Expected Results

After training, you should achieve:
- **mAP50-95**: 0.60-0.80 (good to excellent)
- **mAP50**: 0.80-0.95 (good to excellent)
- **Precision**: 0.70-0.90
- **Recall**: 0.65-0.85

Results vary based on:
- Dataset quality and size
- Model size
- Training duration
- Hardware capabilities

## Tips for Best Results

1. **Start with quick mode** to verify everything works
2. **Use standard mode** for production
3. **Monitor training** - loss should decrease, mAP should increase
4. **Be patient** - good models take time to train
5. **Save checkpoints** - training can be resumed if interrupted
6. **Validate results** - test on real videos before deployment

## Need Help?

1. Check training plots for issues
2. Review dataset labels and images
3. Try different model sizes
4. Adjust batch size for your hardware
5. Consult YOLO documentation: https://docs.ultralytics.com

---

**Ready to train? Start with:**
```bash
python train_simple.py
```

Good luck! 🚀
