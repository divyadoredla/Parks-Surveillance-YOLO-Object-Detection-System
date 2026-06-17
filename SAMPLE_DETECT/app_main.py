"""
Parks Surveillance Web Application
Features: Authentication and Object Detection
"""

import streamlit as st
import cv2
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Import custom modules
from database import init_database, verify_user, create_user, get_user_count
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, APP_TITLE, APP_ICON,
    UPLOAD_DIR, RESULTS_DIR, SUPPORTED_IMAGE_FORMATS, SUPPORTED_VIDEO_FORMATS
)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database and session state
if 'db_initialized' not in st.session_state:
    init_database()
    st.session_state.db_initialized = True

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'


def login_page():
    """Professional login page"""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("### Secure Access - Activity Monitoring System")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("🔐 Login")
                
                if submit:
                    if username and password:
                        if verify_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.page = 'detection'
                            st.success("✓ Login successful!")
                            st.rerun()
                        else:
                            st.error("✗ Invalid credentials")
                    else:
                        st.warning("⚠ Please enter both username and password")
            
            st.info("💡 **Default:** `admin` / `admin123`")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Min 3 characters")
                new_password = st.text_input("Choose Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("✨ Create Account")
                
                if submit:
                    if not new_username or not new_password or not confirm_password:
                        st.warning("⚠ Please fill in all fields")
                    elif len(new_username) < 3:
                        st.error("✗ Username must be at least 3 characters")
                    elif len(new_password) < 6:
                        st.error("✗ Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("✗ Passwords do not match")
                    else:
                        if create_user(new_username, new_password):
                            st.success(f"✓ Account created! Welcome, {new_username}!")
                            st.session_state.authenticated = True
                            st.session_state.username = new_username
                            st.session_state.page = 'detection'
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("✗ Username already exists")


def save_uploaded_file(uploaded_file):
    """Save uploaded file and return path"""
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def detect_image(image_path, model, conf_threshold):
    """Detect objects in image"""
    results = model(image_path, conf=conf_threshold)[0]
    
    detections = []
    for box in results.boxes:
        detection = {
            'class': results.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].cpu().numpy().tolist()
        }
        detections.append(detection)
    
    annotated_image = results.plot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"detected_{timestamp}.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    
    return {
        'detections': detections,
        'annotated_path': str(output_path),
        'total_detections': len(detections)
    }


def detect_video(video_path, model, conf_threshold, progress_callback=None):
    """Detect objects in video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"detected_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    all_detections = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=conf_threshold)[0]
        
        for box in results.boxes:
            detection = {
                'class': results.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'frame': frame_num
            }
            all_detections.append(detection)
        
        annotated_frame = results.plot()
        out.write(annotated_frame)
        
        frame_num += 1
        if progress_callback:
            progress_callback(frame_num, total_frames)
    
    cap.release()
    out.release()
    
    return {
        'total_frames': total_frames,
        'total_detections': len(all_detections),
        'detections': all_detections,
        'annotated_path': str(output_path)
    }


def detection_page():
    """Main detection page"""
    st.title("🎯 Object Detection")
    st.markdown("### Upload images or videos for YOLO detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=SUPPORTED_IMAGE_FORMATS + SUPPORTED_VIDEO_FORMATS,
        help="Supported: JPG, PNG, MP4, AVI, MOV"
    )
    
    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        col1.metric("Filename", uploaded_file.name)
        col2.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        col3.metric("Type", uploaded_file.type)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_image = file_extension in SUPPORTED_IMAGE_FORMATS
        is_video = file_extension in SUPPORTED_VIDEO_FORMATS
        
        if is_image:
            st.markdown("#### 🖼️ Image Preview")
            st.image(uploaded_file, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=CONFIDENCE_THRESHOLD,
                step=0.05
            )
        with col2:
            st.info(f"Using model: **best.pt** (Custom trained)")
        
        if st.button("🎯 Run Detection", use_container_width=True):
            file_path = save_uploaded_file(uploaded_file)
            
            try:
                with st.spinner("Loading YOLO model..."):
                    model = YOLO(MODEL_PATH)
                
                if is_image:
                    with st.spinner("🔍 Detecting objects..."):
                        result = detect_image(file_path, model, conf_threshold)
                    
                    st.markdown("## 📊 Detection Results")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Detections", result['total_detections'])
                    col2.metric("Confidence Threshold", f"{conf_threshold:.0%}")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🖼️ Detected Objects")
                    st.image(result['annotated_path'], use_container_width=True)
                    
                    if result['detections']:
                        st.markdown("#### 📋 Detection Details")
                        for i, det in enumerate(result['detections'], 1):
                            st.markdown(f"{i}. **{det['class']}** - Confidence: {det['confidence']:.0%}")
                        
                        with open(result['annotated_path'], "rb") as file:
                            st.download_button(
                                label="⬇️ Download Annotated Image",
                                data=file,
                                file_name=f"detected_{uploaded_file.name}",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                    else:
                        st.info("No objects detected. Try lowering the confidence threshold.")
                
                elif is_video:
                    st.markdown("#### 🎬 Processing Video")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(frame_num, total_frames):
                        progress = frame_num / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_num}/{total_frames}")
                    
                    result = detect_video(file_path, model, conf_threshold, update_progress)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✓ Processing complete!")
                    
                    st.markdown("## 📊 Video Detection Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Frames", result['total_frames'])
                    col2.metric("Total Detections", result['total_detections'])
                    col3.metric("Avg per Frame", f"{result['total_detections']/result['total_frames']:.1f}")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🎥 Processed Video")
                    st.video(result['annotated_path'])
                    
                    if result['detections']:
                        st.markdown("#### � Detection Summary")
                        
                        class_counts = {}
                        for det in result['detections']:
                            class_name = det['class']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                            st.markdown(f"**{class_name}**: {count} detections")
                        
                        with open(result['annotated_path'], "rb") as file:
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=file,
                                file_name=f"detected_{uploaded_file.name}",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    else:
                        st.info("No objects detected in video. Try lowering the confidence threshold.")
            
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.exception(e)


def main():
    """Main application logic"""
    
    with st.sidebar:
        st.markdown("## 🎥 Navigation")
        
        if st.session_state.authenticated:
            st.success(f"✓ Logged in as **{st.session_state.username}**")
            st.markdown("---")
            
            if st.button("🎯 Detection", use_container_width=True):
                st.session_state.page = 'detection'
                st.rerun()
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = 'login'
                st.rerun()
        else:
            st.info("Please login to access the system")
    
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.page == 'detection':
            detection_page()


if __name__ == "__main__":
    main()
