# YOLO Training Quick Reference

## 🚀 Quick Start Commands

### Simplest Way (Beginners)
```bash
python train_simple.py
```

### Recommended Way
```bash
python train_comprehensive.py --mode standard
```

## 📋 All Training Options

### Simple Training
```bash
python train_simple.py
```
- ✅ No configuration needed
- ✅ Good defaults
- ⏱️ ~1-2 hours (GPU)

### Comprehensive Training

**Quick Mode** (Testing - 50 epochs)
```bash
python train_comprehensive.py --mode quick
```

**Standard Mode** (Recommended - 100 epochs)
```bash
python train_comprehensive.py --mode standard
```

**Production Mode** (Best - 200 epochs)
```bash
python train_comprehensive.py --mode production
```

**Custom Training**
```bash
python train_comprehensive.py --mode standard --epochs 150 --batch 8 --model yolo11m.pt
```

### Advanced Training (Original Scripts)

**Basic Training**
```bash
python train_yolo.py
```

**Advanced with Modes**
```bash
# Interactive mode
python train_yolo_advanced.py

# Command-line modes
python train_yolo_advanced.py --mode quick
python train_yolo_advanced.py --mode standard
python train_yolo_advanced.py --mode high_accuracy

# Resume training
python train_yolo_advanced.py --mode resume --checkpoint path/to/last.pt

# Fine-tune
python train_yolo_advanced.py --mode fine_tune --checkpoint path/to/model.pt
```

## 🔧 Common Customizations

### Change Model Size
```bash
python train_comprehensive.py --mode standard --model yolo11s.pt
```

Models: `yolo11n.pt` (fastest) → `yolo11s.pt` → `yolo11m.pt` → `yolo11l.pt` → `yolo11x.pt` (best)

### Adjust Training Duration
```bash
python train_comprehensive.py --mode standard --epochs 200
```

### Reduce Memory Usage
```bash
python train_comprehensive.py --mode standard --batch 4 --imgsz 320
```

### Increase Accuracy
```bash
python train_comprehensive.py --mode standard --epochs 200 --imgsz 1280 --model yolo11m.pt
```

## 📊 Evaluate & Export

### Evaluate Model
```bash
python train_comprehensive.py --evaluate path/to/best.pt
```

### Export to ONNX
```bash
python train_comprehensive.py --export path/to/best.pt --formats onnx
```

### Export Multiple Formats
```bash
python train_comprehensive.py --export path/to/best.pt --formats onnx torchscript
```

### Generate Report
```bash
python train_comprehensive.py --report path/to/best.pt
```

## 📁 Find Your Trained Model

After training, your model will be at:
```
runs/train/
└── [experiment_name]/
    └── weights/
        ├── best.pt    ⭐ Use this!
        └── last.pt    (for resuming)
```

## 🎯 Use Trained Model

Update `detect_video.py`:
```python
model_path = "runs/train/parks_surveillance_training3/weights/best.pt"
```

Then run:
```bash
python detect_video.py
```

## ⚡ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch 4` or `--imgsz 320` |
| Too slow | `--mode quick` or `--model yolo11n.pt` |
| Low accuracy | `--epochs 200` or `--model yolo11m.pt` |
| Want to resume | Use `train_yolo_advanced.py --mode resume` |

## 📈 Expected Training Time

| Hardware | Quick | Standard | Production |
|----------|-------|----------|------------|
| CPU | 2-3 hrs | 4-6 hrs | 8-12 hrs |
| GPU (6GB) | 30 min | 1-2 hrs | 4-6 hrs |
| GPU (12GB+) | 15 min | 30-60 min | 2-3 hrs |

## ✅ Training Checklist

- [ ] Dataset ready at `data/data.yaml`
- [ ] Choose training script (`train_simple.py` or `train_comprehensive.py`)
- [ ] Select mode (quick/standard/production)
- [ ] Adjust batch size if needed (for memory)
- [ ] Start training
- [ ] Monitor progress (loss ↓, mAP ↑)
- [ ] Find best model in `runs/train/`
- [ ] Evaluate model
- [ ] Use in `detect_video.py`

## 🎓 Recommended Workflow

1. **Test First**: `python train_comprehensive.py --mode quick`
2. **Full Training**: `python train_comprehensive.py --mode standard`
3. **Evaluate**: `python train_comprehensive.py --evaluate path/to/best.pt`
4. **Deploy**: Update `detect_video.py` with model path

---

**Need more details?** See `TRAINING_GUIDE.md`
