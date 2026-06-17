"""
Simple YOLO Inference Script for Parks Surveillance
===================================================

A beginner-friendly, unified inference script for running YOLO object detection
on images, videos, and webcam feeds.

Usage Examples:
    # Image inference
    python inference_simple.py --source image.png --mode image
    
    # Video inference
    python inference_simple.py --source dataset1.mp4 --mode video
    
    # Webcam inference
    python inference_simple.py --mode webcam
    
    # Custom model and confidence
    python inference_simple.py --source video.mp4 --model best.pt --conf 0.4
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import torch


def infer_image(model, image_path, conf_threshold=0.25, save_output=True, display=True):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model instance
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold
        save_output (bool): Whether to save annotated image
        display (bool): Whether to display the result
    
    Returns:
        str: Path to output image if saved
    """
    print(f"\n{'='*70}")
    print(f"🖼️  IMAGE INFERENCE")
    print(f"{'='*70}")
    print(f"📁 Input: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Run inference
    print(f"🔍 Running detection...")
    results = model(image_path, conf=conf_threshold, verbose=False)
    
    # Process results
    result = results[0]
    num_detections = len(result.boxes)
    
    print(f"\n✅ Detected {num_detections} objects:")
    
    # Print detection details
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        print(f"  {i+1}. {class_name} (confidence: {confidence:.2f})")
        print(f"     Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    
    # Get annotated image
    annotated_img = result.plot()
    
    # Save output
    output_path = None
    if save_output:
        output_path = str(Path(image_path).stem) + "_detected" + str(Path(image_path).suffix)
        cv2.imwrite(output_path, annotated_img)
        print(f"\n💾 Saved to: {output_path}")
    
    # Display
    if display:
        cv2.imshow('YOLO Detection', annotated_img)
        print(f"\n👁️  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return output_path


def infer_video(model, video_path, conf_threshold=0.25, save_output=True, display=True):
    """
    Run inference on a video file.
    
    Args:
        model: YOLO model instance
        video_path (str): Path to input video
        conf_threshold (float): Confidence threshold
        save_output (bool): Whether to save annotated video
        display (bool): Whether to display the video
    
    Returns:
        str: Path to output video if saved
    """
    print(f"\n{'='*70}")
    print(f"🎥 VIDEO INFERENCE")
    print(f"{'='*70}")
    print(f"📁 Input: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Setup output video writer
    output_path = None
    out = None
    if save_output:
        output_path = str(Path(video_path).stem) + "_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Output: {output_path}")
    
    print(f"\n🔍 Processing video... (Press 'q' to quit)")
    print(f"{'='*70}")
    
    frame_count = 0
    total_detections = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # Count detections
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Objects: {num_detections}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress update
        if frame_count % 30 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {num_detections} objects")
        
        # Save frame
        if save_output and out is not None:
            out.write(annotated_frame)
        
        # Display frame
        if display:
            cv2.imshow('YOLO Video Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Stopped by user")
                break
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ VIDEO PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Summary:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections/frame: {total_detections/frame_count:.2f}")
    if save_output and output_path:
        print(f"   💾 Saved to: {output_path}")
    
    return output_path


def infer_webcam(model, conf_threshold=0.25, camera_id=0):
    """
    Run inference on webcam feed.
    
    Args:
        model: YOLO model instance
        conf_threshold (float): Confidence threshold
        camera_id (int): Camera device ID (usually 0)
    """
    print(f"\n{'='*70}")
    print(f"📹 WEBCAM INFERENCE")
    print(f"{'='*70}")
    print(f"🎥 Opening camera {camera_id}...")
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Error opening camera {camera_id}")
    
    print(f"✅ Camera opened successfully")
    print(f"\n🔍 Running live detection... (Press 'q' to quit)")
    print(f"{'='*70}\n")
    
    frame_count = 0
    
    # Process webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from camera")
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # Count detections
        num_detections = len(results[0].boxes)
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Objects: {num_detections} | Press 'q' to quit"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLO Webcam Detection', annotated_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n⏹️  Stopped by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"✅ WEBCAM INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total frames processed: {frame_count}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Simple YOLO Inference for Parks Surveillance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image inference
  python inference_simple.py --source image.png --mode image
  
  # Video inference
  python inference_simple.py --source dataset1.mp4 --mode video
  
  # Webcam inference
  python inference_simple.py --mode webcam
  
  # Custom settings
  python inference_simple.py --source video.mp4 --model best.pt --conf 0.4 --no-display
        """
    )
    
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image or video file')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['image', 'video', 'webcam', 'auto'],
                       help='Inference mode (default: auto-detect from source)')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model (default: yolo11n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam mode (default: 0)')
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'='*70}")
    print(f"🚀 YOLO INFERENCE - PARKS SURVEILLANCE")
    print(f"{'='*70}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device.upper()}")
    if device == 'cpu':
        print(f"⚠️  Running on CPU (GPU recommended for better performance)")
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Determine mode
    mode = args.mode
    if mode == 'auto' and args.source:
        ext = Path(args.source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            mode = 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            mode = 'video'
        else:
            print(f"❌ Cannot auto-detect mode for extension: {ext}")
            print(f"   Please specify --mode manually")
            return
    elif mode == 'auto':
        mode = 'webcam'
    
    # Run inference
    try:
        if mode == 'image':
            if not args.source:
                print(f"❌ Error: --source required for image mode")
                return
            infer_image(model, args.source, args.conf, 
                       not args.no_save, not args.no_display)
        
        elif mode == 'video':
            if not args.source:
                print(f"❌ Error: --source required for video mode")
                return
            infer_video(model, args.source, args.conf,
                       not args.no_save, not args.no_display)
        
        elif mode == 'webcam':
            infer_webcam(model, args.conf, args.camera)
        
        print(f"\n✅ Inference completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
