from ultralytics import YOLO
import cv2
import os

def detect_objects(image_path, model_path="yolo11n.pt", conf_threshold=0.25):
    """
    Detect objects in an image using YOLO model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        results: YOLO detection results
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load the pretrained YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference on the image
    print(f"Running object detection on: {image_path}")
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    for result in results:
        # Get the number of detections
        num_detections = len(result.boxes)
        print(f"\nDetected {num_detections} objects:")
        
        # Print detection details
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            print(f"  {i+1}. {class_name} (confidence: {confidence:.2f})")
            print(f"     Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Save annotated image
        output_path = image_path.replace('.', '_detected.')
        annotated_img = result.plot()  # Draw boxes and labels
        cv2.imwrite(output_path, annotated_img)
        print(f"\nAnnotated image saved to: {output_path}")
    
    return results


def main():
    """Main function to run object detection."""
    # Example usage
    image_path = "image.png"  # Change this to your image path
    
    # You can also specify custom parameters
    model_path = "yolo11n.pt"  # YOLO11 nano model
    conf_threshold = 0.25  # Confidence threshold
    
    try:
        results = detect_objects(
            image_path=image_path,
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        
        print("\n✓ Object detection completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease provide a valid image path.")
        print("Usage: python detect_objects.py")
        print("       (Make sure 'test_image.jpg' exists or modify the image_path variable)")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()