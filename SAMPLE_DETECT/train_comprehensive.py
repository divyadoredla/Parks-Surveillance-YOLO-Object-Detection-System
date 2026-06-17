"""
Comprehensive YOLO Training Script for Parks Surveillance
==========================================================

This script provides a complete training solution with:
- Multiple training modes (quick, standard, production)
- Automatic configuration management
- Real-time progress monitoring
- Comprehensive evaluation and reporting
- Model export capabilities

Usage:
    python train_comprehensive.py --mode standard
    python train_comprehensive.py --mode quick --epochs 30
    python train_comprehensive.py --mode production --model yolo11m.pt
"""

from ultralytics import YOLO
import os
import yaml
import torch
import argparse
from datetime import datetime
from pathlib import Path
import json


class ComprehensiveYOLOTrainer:
    """Comprehensive YOLO trainer with advanced features."""
    
    def __init__(self, data_yaml="data/data.yaml", output_dir="runs/train"):
        """
        Initialize the trainer.
        
        Args:
            data_yaml: Path to dataset configuration file
            output_dir: Directory to save training outputs
        """
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load dataset configuration
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
        
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Display system information
        self._display_system_info()
    
    def _display_system_info(self):
        """Display system and dataset information."""
        print("=" * 80)
        print("COMPREHENSIVE YOLO TRAINING - PARKS SURVEILLANCE")
        print("=" * 80)
        print(f"\n📊 Dataset Information:")
        print(f"   Configuration: {self.data_yaml}")
        print(f"   Classes: {self.data_config['nc']}")
        print(f"   Class names: {', '.join(self.data_config['names'])}")
        
        print(f"\n💻 System Information:")
        print(f"   Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"   ⚠️  Running on CPU - Training will be slower")
        print(f"   PyTorch version: {torch.__version__}")
    
    def get_training_config(self, mode="standard", **kwargs):
        """
        Get training configuration based on mode.
        
        Args:
            mode: Training mode (quick, standard, production, custom)
            **kwargs: Override parameters
        
        Returns:
            dict: Training configuration
        """
        # Predefined configurations
        configs = {
            'quick': {
                'model': 'yolo11n.pt',
                'epochs': 50,
                'imgsz': 640,
                'batch': 16,
                'patience': 20,
                'save_period': 10,
                'description': 'Quick training for testing (50 epochs, nano model)'
            },
            'standard': {
                'model': 'yolo11s.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'patience': 50,
                'save_period': 10,
                'description': 'Standard training for balanced performance (100 epochs, small model)'
            },
            'production': {
                'model': 'yolo11m.pt',
                'epochs': 200,
                'imgsz': 1280,
                'batch': 8,
                'patience': 100,
                'save_period': 20,
                'description': 'Production training for best accuracy (200 epochs, medium model)'
            }
        }
        
        if mode not in configs:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(configs.keys())}")
        
        config = configs[mode].copy()
        
        # Override with custom parameters
        config.update(kwargs)
        
        return config
    
    def train(self, mode="standard", **kwargs):
        """
        Train YOLO model with specified configuration.
        
        Args:
            mode: Training mode
            **kwargs: Override parameters
        
        Returns:
            YOLO: Trained model
        """
        # Get configuration
        config = self.get_training_config(mode, **kwargs)
        
        # Display training configuration
        print(f"\n🚀 Training Configuration:")
        print(f"   Mode: {mode}")
        print(f"   Description: {config['description']}")
        print(f"   Model: {config['model']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Image size: {config['imgsz']}")
        print(f"   Batch size: {config['batch']}")
        print(f"   Patience: {config['patience']}")
        print(f"   Save period: {config['save_period']} epochs")
        
        # Load model
        print(f"\n📦 Loading model: {config['model']}")
        model = YOLO(config['model'])
        
        # Prepare experiment name
        experiment_name = f"{mode}_{config['model'].replace('.pt', '')}_{self.timestamp}"
        
        print(f"\n🎯 Starting training...")
        print(f"   Experiment: {experiment_name}")
        print(f"   Output directory: {self.output_dir}/{experiment_name}")
        print("=" * 80)
        
        # Training parameters
        train_params = {
            'data': self.data_yaml,
            'epochs': config['epochs'],
            'imgsz': config['imgsz'],
            'batch': config['batch'],
            'device': self.device,
            'project': self.output_dir,
            'name': experiment_name,
            'patience': config['patience'],
            'save': True,
            'save_period': config['save_period'],
            'plots': True,
            'verbose': True,
            # Data augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 5.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
        }
        
        # Train the model
        results = model.train(**train_params)
        
        # Save training configuration
        config_path = Path(self.output_dir) / experiment_name / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\n📁 Results saved to: {model.trainer.save_dir}")
        print(f"   Best model: {model.trainer.best}")
        print(f"   Last checkpoint: {model.trainer.last}")
        
        return model
    
    def evaluate(self, model_path, split='val'):
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model weights
            split: Dataset split to evaluate on ('val' or 'test')
        
        Returns:
            Validation metrics
        """
        print("\n" + "=" * 80)
        print(f"📊 EVALUATING MODEL: {split.upper()} SET")
        print("=" * 80)
        
        model = YOLO(model_path)
        metrics = model.val(data=self.data_yaml, split=split)
        
        # Display metrics
        print(f"\n📈 Overall Performance:")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   mAP50:    {metrics.box.map50:.4f}")
        print(f"   mAP75:    {metrics.box.map75:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall:    {metrics.box.mr:.4f}")
        
        # Per-class metrics
        print(f"\n📋 Per-Class Performance:")
        for i, class_name in enumerate(self.data_config['names']):
            if i < len(metrics.box.maps):
                print(f"   {class_name:15s}: mAP50-95 = {metrics.box.maps[i]:.4f}")
        
        return metrics
    
    def export_model(self, model_path, formats=['onnx']):
        """
        Export model to different formats.
        
        Args:
            model_path: Path to model weights
            formats: List of export formats
        
        Returns:
            dict: Exported model paths
        """
        print("\n" + "=" * 80)
        print("📦 EXPORTING MODEL")
        print("=" * 80)
        
        model = YOLO(model_path)
        exported = {}
        
        for fmt in formats:
            print(f"\n   Exporting to {fmt.upper()}...")
            try:
                export_path = model.export(format=fmt)
                exported[fmt] = export_path
                print(f"   ✅ Success: {export_path}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return exported
    
    def generate_report(self, model_path):
        """
        Generate a comprehensive training report.
        
        Args:
            model_path: Path to trained model
        """
        print("\n" + "=" * 80)
        print("📄 GENERATING TRAINING REPORT")
        print("=" * 80)
        
        # Evaluate on validation set
        val_metrics = self.evaluate(model_path, split='val')
        
        # Evaluate on test set if available
        if 'test' in self.data_config:
            test_metrics = self.evaluate(model_path, split='test')
        
        # Model information
        model = YOLO(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"\n📊 Model Information:")
        print(f"   Model path: {model_path}")
        print(f"   Model size: {model_size:.2f} MB")
        
        print("\n" + "=" * 80)
        print("✅ REPORT GENERATION COMPLETE")
        print("=" * 80)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Comprehensive YOLO Training for Parks Surveillance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (testing)
  python train_comprehensive.py --mode quick
  
  # Standard training (recommended)
  python train_comprehensive.py --mode standard
  
  # Production training (best accuracy)
  python train_comprehensive.py --mode production
  
  # Custom training
  python train_comprehensive.py --mode standard --epochs 150 --batch 8
  
  # Evaluate existing model
  python train_comprehensive.py --evaluate path/to/best.pt
  
  # Export model
  python train_comprehensive.py --export path/to/best.pt --formats onnx torchscript
        """
    )
    
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'standard', 'production'],
                       help='Training mode')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--model', type=str, default=None,
                       help='Model size (overrides mode default)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides mode default)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (overrides mode default)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size (overrides mode default)')
    parser.add_argument('--evaluate', type=str, default=None,
                       help='Evaluate a trained model (provide path to weights)')
    parser.add_argument('--export', type=str, default=None,
                       help='Export a trained model (provide path to weights)')
    parser.add_argument('--formats', nargs='+', default=['onnx'],
                       help='Export formats (for --export)')
    parser.add_argument('--report', type=str, default=None,
                       help='Generate comprehensive report (provide path to weights)')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ComprehensiveYOLOTrainer(data_yaml=args.data)
        
        # Evaluation mode
        if args.evaluate:
            trainer.evaluate(args.evaluate)
            return
        
        # Export mode
        if args.export:
            trainer.export_model(args.export, formats=args.formats)
            return
        
        # Report mode
        if args.report:
            trainer.generate_report(args.report)
            return
        
        # Training mode
        kwargs = {}
        if args.model:
            kwargs['model'] = args.model
        if args.epochs:
            kwargs['epochs'] = args.epochs
        if args.batch:
            kwargs['batch'] = args.batch
        if args.imgsz:
            kwargs['imgsz'] = args.imgsz
        
        # Train the model
        model = trainer.train(mode=args.mode, **kwargs)
        
        # Get best model path
        best_model = model.trainer.best
        
        # Automatic evaluation
        print("\n" + "=" * 80)
        print("🔍 AUTOMATIC EVALUATION")
        print("=" * 80)
        trainer.evaluate(best_model)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n✅ Your trained model is ready!")
        print(f"   Best model: {best_model}")
        print(f"\n📝 Next steps:")
        print(f"   1. Use for detection: python detect_video.py")
        print(f"      (Update model_path to: {best_model})")
        print(f"   2. Export model: python train_comprehensive.py --export {best_model}")
        print(f"   3. Generate report: python train_comprehensive.py --report {best_model}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
