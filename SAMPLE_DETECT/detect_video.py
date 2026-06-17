from ultralytics import YOLO
import cv2
import os
from pathlib import Path


def detect_objects_video(video_path, model_path="yolo11n.pt", conf_threshold=0.25, 
                         save_output=True, display_video=True):
    """
    Detect objects in a video
     using YOLO model.
    
    Args:
        video_path (str): Path to the input video
        model_path (str): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
        save_output (bool): Whether to save the annotated video
        display_video (bool): Whether to display the video in real-time
    
    Returns:
        str: Path to the output video (if saved)
    """
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load the pretrained YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    
    # Prepare output video writer
    output_path = None
    out = None
    if save_output:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"  Output will be saved to: {output_path}")
    
    print(f"\nProcessing video with confidence threshold: {conf_threshold}")
    print("Press 'q' to quit early\n")
    
    frame_count = 0
    total_detections = 0
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO inference on the frame
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Count detections in this frame
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        # Add frame info to the video
        info_text = f"Frame: {frame_count}/{total_frames} | Objects: {num_detections}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display progress
        if frame_count % 30 == 0 or frame_count == 1:
            print(f"Processing frame {frame_count}/{total_frames} - Detected {num_detections} objects")
        
        # Save frame to output video
        if save_output and out is not None:
            out.write(annotated_frame)
        
        # Display the frame
        if display_video:
            cv2.imshow('Object Detection', annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    if save_output and output_path:
        print(f"Output saved to: {output_path}")
    
    return output_path


def detect_objects_video_detailed(video_path, model_path="yolo11n.pt", conf_threshold=0.25):
    """
    Detect objects in a video with detailed statistics and tracking.
    
    Args:
        video_path (str): Path to the input video
        model_path (str): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        dict: Dictionary containing detection statistics
    """
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load the pretrained YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nAnalyzing video: {video_path}")
    print(f"Total frames: {total_frames}")
    
    # Statistics tracking
    class_counts = {}
    frame_detections = []
    
    frame_count = 0
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        frame_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0])
            
            # Update class counts
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            
            frame_objects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': box.xyxy[0].tolist()
            })
        
        frame_detections.append({
            'frame': frame_count,
            'objects': frame_objects
        })
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Analyzed {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"DETECTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total frames analyzed: {frame_count}")
    print(f"\nObject Class Distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} detections")
    
    # Return statistics
    return {
        'total_frames': frame_count,
        'class_counts': class_counts,
        'frame_detections': frame_detections
    }


def main():
    """Main function to run video object detection."""
    # Example usage
    video_path = "dataset1.mp4"  # Change this to your video path
    
    # You can also specify custom parameters
    model_path = "yolo11n.pt"  # YOLO11 nano model
    conf_threshold = 0.25  # Confidence threshold
    
    try:
        print("="*60)
        print("VIDEO OBJECT DETECTION")
        print("="*60)
        
        # Option 1: Basic detection with visualization
        output_path = detect_objects_video(
            video_path=video_path,
            model_path=model_path,
            conf_threshold=conf_threshold,
            save_output=True,
            display_video=True
        )
        
        print("\n✓ Video object detection completed successfully!")
        
        # Option 2: Detailed analysis (uncomment to use)
        # print("\n\nRunning detailed analysis...")
        # stats = detect_objects_video_detailed(
        #     video_path=video_path,
        #     model_path=model_path,
        #     conf_threshold=conf_threshold
        # )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease provide a valid video path.")
        print("Usage: python detect_video.py")
        print("       (Make sure your video file exists or modify the video_path variable)")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
