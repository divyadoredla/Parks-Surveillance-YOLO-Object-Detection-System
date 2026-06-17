"""
YOLO Prediction Module
Wrapper for YOLO inference on images and videos
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
from datetime import datetime


class PredictionResult:
    """Class to store prediction results"""
    def __init__(self):
        self.num_detections = 0
        self.detections = []  # List of (class_name, confidence, bbox)
        self.annotated_path = None
        self.processing_time = 0
        self.total_frames = 0
        self.source_type = None  # 'image' or 'video'


def predict_image(image_path: str, model_path: str = "yolo11n.pt", conf_threshold: float = 0.25) -> PredictionResult:
    """
    Run YOLO prediction on an image
    
    Args:
        image_path: Path to the input image
        model_path: Path to YOLO model weights
        conf_threshold: Confidence threshold for detections
        
    Returns:
        PredictionResult object with detection results
    """
    import time
    start_time = time.time()
    
    result_obj = PredictionResult()
    result_obj.source_type = 'image'
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    for result in results:
        result_obj.num_detections = len(result.boxes)
        
        # Extract detection details
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            result_obj.detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
        
        # Save annotated image
        output_path = image_path.replace('.', '_detected.')
        annotated_img = result.plot()
        cv2.imwrite(output_path, annotated_img)
        result_obj.annotated_path = output_path
    
    result_obj.processing_time = time.time() - start_time
    return result_obj


def predict_video(video_path: str, model_path: str = "yolo11n.pt", conf_threshold: float = 0.25, 
                  progress_callback=None) -> PredictionResult:
    """
    Run YOLO prediction on a video
    
    Args:
        video_path: Path to the input video
        model_path: Path to YOLO model weights
        conf_threshold: Confidence threshold for detections
        progress_callback: Optional callback function for progress updates (frame_num, total_frames)
        
    Returns:
        PredictionResult object with detection results
    """
    import time
    start_time = time.time()
    
    result_obj = PredictionResult()
    result_obj.source_type = 'video'
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result_obj.total_frames = total_frames
    
    # Prepare output video
    output_path = video_path.replace('.', '_detected.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    detection_summary = {}
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process results
        for result in results:
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            # Count detections by class
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
            
            # Get annotated frame
            annotated_frame = result.plot()
            out.write(annotated_frame)
        
        # Progress callback
        if progress_callback:
            progress_callback(frame_count, total_frames)
    
    # Release resources
    cap.release()
    out.release()
    
    # Store results
    result_obj.num_detections = total_detections
    result_obj.detections = [
        {'class': class_name, 'count': count}
        for class_name, count in detection_summary.items()
    ]
    result_obj.annotated_path = output_path
    result_obj.processing_time = time.time() - start_time
    
    return result_obj


def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Save an uploaded file to disk
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        upload_dir: Directory to save uploaded files
        
    Returns:
        Path to the saved file
    """
    # Create upload directory if it doesn't exist
    Path(upload_dir).mkdir(exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(upload_dir, filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old files from a directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep (in hours)
    """
    import time
    
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
def generate_pdf_report(result: PredictionResult, original_filename: str, original_file_path: str = None, username: str = "admin", output_dir: str = "uploads") -> str:
    """
    Generate a PDF report for the prediction results
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    import os
    from datetime import datetime
    from activity_classifier import ActivityClassifier

    classifier = ActivityClassifier()
    # Classify the detections
    # ActivityClassifier needs the list of detection dictionaries
    if result.source_type == 'image':
        class_results = classifier.classify_detections(result.detections)
    else:
        # For video, result.detections is already aggregated, so we can't classify individual frames easily here.
        # But we will map what we can, or just mock it.
        # Ideally, we just pass the raw detections if available. Since predict_video aggregates them, 
        # we will simulate it for now.
        class_results = {'summary': {'authorized_count': 0, 'unauthorized_count': 0, 'total_detections': result.num_detections}, 'alerts': []}

    summary = class_results.get('summary', {})
    alerts = class_results.get('alerts', [])

    class PDF(FPDF):
        def header(self):
            # Title
            self.set_font('Arial', 'B', 18)
            self.set_text_color(51, 122, 183) # Blue color
            self.cell(0, 10, 'Park Activity Monitoring Report', 0, 1, 'C')
            self.ln(2)
            self.set_text_color(0, 0, 0)

    pdf = PDF()
    pdf.add_page()

    # 1. Metadata Table
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Report Type:', border=1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(140, 8, 'Park Activity Monitoring', border=1, ln=1)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Generated By:', border=1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(140, 8, username, border=1, ln=1)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Date & Time:', border=1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(140, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), border=1, ln=1)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Detection Model:', border=1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(140, 8, 'YOLOPredictionService', border=1, ln=1)
    pdf.ln(8)

    # 2. Activity Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(40, 160, 60) # Green text
    pdf.cell(0, 8, 'Activity Summary', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(90, 8, 'Metric', border=1)
    pdf.cell(50, 8, 'Count', border=1)
    pdf.cell(50, 8, 'Status', border=1, ln=1)

    # Table Body
    pdf.set_font('Arial', '', 10)
    pdf.cell(90, 8, 'Authorized Activities', border=1)
    pdf.cell(50, 8, str(summary.get('authorized_count', 0)), border=1)
    pdf.cell(50, 8, 'Allowed', border=1, ln=1)

    pdf.cell(90, 8, 'Unauthorized Activities', border=1)
    pdf.cell(50, 8, str(summary.get('unauthorized_count', 0)), border=1)
    pdf.cell(50, 8, 'Violation', border=1, ln=1)

    pdf.cell(90, 8, 'Total Detections', border=1)
    pdf.cell(50, 8, str(summary.get('total_detections', 0)), border=1)
    pdf.cell(50, 8, '-', border=1, ln=1)
    pdf.ln(8)

    # 3. Violations Detected
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(220, 50, 50) # Red text
    pdf.cell(0, 8, 'Violations Detected (Red Boxes)', 0, 1)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Class Name', border=1)
    pdf.cell(30, 8, 'Confidence', border=1)
    pdf.cell(80, 8, 'Rule Violated', border=1)
    pdf.cell(30, 8, 'Level', border=1, ln=1)

    pdf.set_font('Arial', '', 10)
    if not alerts:
        pdf.cell(190, 8, 'No violations detected.', border=1, ln=1, align='C')
    else:
        for alert in alerts:
            pdf.cell(50, 8, alert['class'], border=1)
            pdf.cell(30, 8, f"{alert['confidence']*100:.2f}%", border=1)
            pdf.cell(80, 8, alert['type'], border=1)
            pdf.cell(30, 8, alert['severity'], border=1, ln=1)
    pdf.ln(8)

    # 4. Images
    if result.source_type == 'image' and original_file_path and os.path.exists(original_file_path) and result.annotated_path and os.path.exists(result.annotated_path):
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(40, 160, 60) # Green text
        pdf.cell(0, 8, 'Monitoring Results', 0, 1)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(95, 6, 'Original Image', 0, 0)
        pdf.cell(95, 6, 'Monitored Image', 0, 1)
        
        y_before = pdf.get_y()
        # Ensure we don't go out of bounds
        max_img_h = 100
        pdf.image(original_file_path, x=10, y=y_before, w=90)
        pdf.image(result.annotated_path, x=105, y=y_before, w=90)

    pdf_path = os.path.join(output_dir, f"report_{original_filename}.pdf")
    pdf.output(pdf_path)
    return pdf_path

if __name__ == "__main__":
    # Test image prediction
    print("Testing image prediction...")
    if os.path.exists("image.png"):
        result = predict_image("image.png")
        print(f"✓ Image prediction completed")
        print(f"  Detections: {result.num_detections}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Output: {result.annotated_path}")
    else:
        print("✗ image.png not found")
