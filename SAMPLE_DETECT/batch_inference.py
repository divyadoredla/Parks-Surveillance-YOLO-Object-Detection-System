"""
Batch YOLO Inference Script for Parks Surveillance
==================================================

Process multiple images or videos in batch with parallel processing support.

Features:
- Process entire directories
- Recursive file search
- Parallel processing with multiple workers
- Progress tracking with ETA
- Summary reports (CSV)
- Error recovery
- File filtering by pattern

Usage Examples:
    # Process all images in a directory
    python batch_inference.py --input-dir ./images --type image
    
    # Process all MP4 videos with 4 workers
    python batch_inference.py --input-dir ./videos --pattern "*.mp4" --workers 4
    
    # Recursive search with custom model
    python batch_inference.py --input-dir ./data --recursive --model best.pt
    
    # Generate summary report
    python batch_inference.py --input-dir ./test --report summary.csv
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv
from datetime import datetime
import glob


def process_single_file(file_path, model_path, conf_threshold, output_dir, file_type, save_txt=False):
    """
    Process a single file.
    
    Args:
        file_path (str): Path to input file
        model_path (str): Path to YOLO model
        conf_threshold (float): Confidence threshold
        output_dir (str): Output directory
        file_type (str): 'image' or 'video'
    
    Returns:
        dict: Processing results
    """
    try:
        # Load model
        model = YOLO(model_path)
        
        # Create output path
        file_name = Path(file_path).stem
        file_ext = Path(file_path).suffix
        
        if file_type == 'image':
            output_path = os.path.join(output_dir, f"{file_name}_detected{file_ext}")
            
            # Run inference
            results = model(file_path, conf=conf_threshold, verbose=False)
            result = results[0]
            
            # Save annotated image
            annotated = result.plot()
            cv2.imwrite(output_path, annotated)
            
            # Get statistics
            num_detections = len(result.boxes)
            class_counts = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if save_txt:
                txt_path = os.path.join(output_dir, f"{file_name}.txt")
                with open(txt_path, 'w') as f:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xywhn = box.xywhn[0].tolist()  # normalized xywh
                        f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {conf:.6f}\n")

            return {
                'file': file_path,
                'status': 'success',
                'output': output_path,
                'detections': num_detections,
                'classes': class_counts,
                'frames': 1
            }
        
        else:  # video
            output_path = os.path.join(output_dir, f"{file_name}_detected.mp4")
            
            # Open video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {file_path}")
            
            # Get properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            frame_count = 0
            total_detections = 0
            class_counts = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = model(frame, conf=conf_threshold, verbose=False)
                annotated = results[0].plot()
                
                # Count detections
                for box in results[0].boxes:
                    total_detections += 1
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Write frame
                out.write(annotated)
            
            cap.release()
            out.release()
            
            return {
                'file': file_path,
                'status': 'success',
                'output': output_path,
                'detections': total_detections,
                'classes': class_counts,
                'frames': frame_count
            }
    
    except Exception as e:
        return {
            'file': file_path,
            'status': 'error',
            'error': str(e),
            'detections': 0,
            'classes': {},
            'frames': 0
        }


def find_files(input_dir, pattern='*', recursive=False, file_type='auto'):
    """
    Find files matching pattern.
    
    Args:
        input_dir (str): Input directory
        pattern (str): File pattern (glob)
        recursive (bool): Search recursively
        file_type (str): 'image', 'video', or 'auto'
    
    Returns:
        list: List of file paths
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    if recursive:
        search_pattern = os.path.join(input_dir, '**', pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)
    
    # Filter by type
    filtered_files = []
    for f in files:
        if not os.path.isfile(f):
            continue
        
        ext = Path(f).suffix.lower()
        
        if file_type == 'auto':
            if ext in image_exts or ext in video_exts:
                filtered_files.append(f)
        elif file_type == 'image':
            if ext in image_exts:
                filtered_files.append(f)
        elif file_type == 'video':
            if ext in video_exts:
                filtered_files.append(f)
    
    return filtered_files


def batch_inference(input_dir, model_path='yolo11n.pt', conf_threshold=0.25,
                   output_dir='batch_output', pattern='*', recursive=False,
                   file_type='auto', workers=1, report_path=None, save_txt=False):
    """
    Run batch inference on multiple files.
    
    Args:
        input_dir (str): Input directory
        model_path (str): Path to YOLO model
        conf_threshold (float): Confidence threshold
        output_dir (str): Output directory
        pattern (str): File pattern
        recursive (bool): Search recursively
        file_type (str): File type filter
        workers (int): Number of parallel workers
        report_path (str): Path to save CSV report
    
    Returns:
        dict: Batch processing results
    """
    print(f"\n{'='*70}")
    print(f"📦 BATCH YOLO INFERENCE")
    print(f"{'='*70}")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Model: {model_path}")
    print(f"🎯 Confidence: {conf_threshold}")
    print(f"🔍 Pattern: {pattern}")
    print(f"🔄 Recursive: {recursive}")
    print(f"👷 Workers: {workers}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files
    print(f"\n🔍 Searching for files...")
    files = find_files(input_dir, pattern, recursive, file_type)
    
    if not files:
        print(f"❌ No files found matching pattern: {pattern}")
        return
    
    print(f"✅ Found {len(files)} files to process")
    
    # Determine file types
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    file_types = []
    for f in files:
        ext = Path(f).suffix.lower()
        file_types.append('image' if ext in image_exts else 'video')
    
    # Process files
    print(f"\n🚀 Processing files...")
    print(f"{'='*70}")
    
    results = []
    
    if workers == 1:
        # Sequential processing
        for file_path, ftype in tqdm(zip(files, file_types), total=len(files), desc="Processing"):
            result = process_single_file(file_path, model_path, conf_threshold, output_dir, ftype, save_txt)
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_file, file_path, model_path, 
                              conf_threshold, output_dir, ftype, save_txt): file_path
                for file_path, ftype in zip(files, file_types)
            }
            
            with tqdm(total=len(files), desc="Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
    
    # Calculate statistics
    print(f"\n{'='*70}")
    print(f"✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    total_detections = sum(r['detections'] for r in results)
    total_frames = sum(r['frames'] for r in results)
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total detections: {total_detections}")
    print(f"   Total frames: {total_frames}")
    
    if total_frames > 0:
        print(f"   Avg detections/frame: {total_detections/total_frames:.2f}")
    
    # Aggregate class counts
    all_classes = {}
    for r in results:
        if r['status'] == 'success':
            for class_name, count in r['classes'].items():
                all_classes[class_name] = all_classes.get(class_name, 0) + count
    
    if all_classes:
        print(f"\n📦 Class Distribution:")
        for class_name, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {count}")
    
    # Show errors
    if failed > 0:
        print(f"\n❌ Errors:")
        for r in results:
            if r['status'] == 'error':
                print(f"   {Path(r['file']).name}: {r['error']}")
    
    # Save report
    if report_path:
        print(f"\n💾 Saving report to: {report_path}")
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Status', 'Detections', 'Frames', 'Output', 'Error'])
            for r in results:
                writer.writerow([
                    r['file'],
                    r['status'],
                    r['detections'],
                    r['frames'],
                    r.get('output', ''),
                    r.get('error', '')
                ])
        print(f"✅ Report saved")
    
    print(f"\n💾 Output files saved to: {output_dir}")
    
    return {
        'total': len(files),
        'successful': successful,
        'failed': failed,
        'results': results,
        'class_counts': all_classes
    }


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Batch YOLO Inference for Multiple Files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python batch_inference.py --input-dir ./images --type image
  
  # Process all MP4 videos with 4 workers
  python batch_inference.py --input-dir ./videos --pattern "*.mp4" --workers 4
  
  # Recursive search with custom model
  python batch_inference.py --input-dir ./data --recursive --model best.pt
  
  # Generate summary report
  python batch_inference.py --input-dir ./test --report summary.csv
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing files')
    parser.add_argument('--output-dir', type=str, default='batch_output',
                       help='Output directory (default: batch_output)')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model (default: yolo11n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--pattern', type=str, default='*',
                       help='File pattern to match (default: *)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively in subdirectories')
    parser.add_argument('--type', type=str, default='auto',
                       choices=['auto', 'image', 'video'],
                       help='File type filter (default: auto)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--report', type=str, default=None,
                       help='Path to save CSV report')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save detections to .txt files')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    if device == 'cpu' and args.workers > 1:
        print(f"⚠️  Running on CPU with multiple workers may not improve performance")
    
    # Run batch inference
    try:
        batch_inference(
            input_dir=args.input_dir,
            model_path=args.model,
            conf_threshold=args.conf,
            output_dir=args.output_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            file_type=args.type,
            workers=args.workers,
            report_path=args.report,
            save_txt=args.save_txt
        )
        
        print(f"\n✅ Batch processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
