"""
Simple YOLO Training Script for Parks Surveillance
==================================================

This is a simplified, beginner-friendly training script.
Just run it and it will train a YOLO model on your dataset!

Usage:
    python train_simple.py
"""

from ultralytics import YOLO
import torch

def train_yolo():
    """Simple YOLO training function."""
    
    print("=" * 70)
    print("YOLO TRAINING - PARKS SURVEILLANCE")
    print("=" * 70)
    
    # Configuration
    data_yaml = "data/data.yaml"
    model_name = "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt
    epochs = 25
    image_size = 640
    batch_size = 16
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📊 Training Configuration:")
    print(f"   Dataset: {data_yaml}")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {image_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    if device == 'cpu':
        print(f"\n⚠️  WARNING: Training on CPU will be slow!")
        print(f"   Consider using a GPU for faster training.")
    
    # Load model
    print(f"\n📦 Loading YOLO model...")
    model = YOLO(model_name)
    
    # Start training
    print(f"\n🚀 Starting training...")
    print("=" * 70)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project="runs/train",
        name="parks_surveillance_training",
        patience=50,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    
    # Show results
    best_model = model.trainer.best
    print(f"\n📁 Results:")
    print(f"   Best model: {best_model}")
    print(f"   Training plots: {model.trainer.save_dir}")
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    metrics = model.val()
    
    print(f"\n📈 Performance:")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    print(f"\n✅ Training complete! Use your model:")
    print(f"   Update detect_video.py with: model_path = '{best_model}'")
    
    return best_model


if __name__ == "__main__":
    try:
        train_yolo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
