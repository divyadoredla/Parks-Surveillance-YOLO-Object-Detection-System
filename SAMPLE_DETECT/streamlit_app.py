"""
Streamlit Web Application for YOLO Surveillance Platform
Features: Authentication, File Upload, and Real-time Object Detection
"""

import streamlit as st
import os
from pathlib import Path
from database import init_database, verify_user, get_user_count, create_user
from prediction_module import (
    predict_image, 
    predict_video, 
    save_uploaded_file,
    cleanup_old_files,
    generate_pdf_report
)

# Page configuration
st.set_page_config(
    page_title="YOLO Surveillance Platform",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
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
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        color: #333333;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database on first run
if 'db_initialized' not in st.session_state:
    init_database()
    st.session_state.db_initialized = True

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI Assistant for the Parks Surveillance Platform. How can I help you today?"}
    ]


def login_page():
    """Display login page"""
    st.title("🎥 YOLO Surveillance Platform")
    st.markdown("### Welcome! Please login to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🔐 Login")
            
            if submit:
                if username and password:
                    if verify_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.page = 'home'
                        st.success("✓ Login successful!")
                        st.rerun()
                    else:
                        st.error("✗ Invalid username or password")
                else:
                    st.warning("⚠ Please enter both username and password")
        
        st.markdown("---")
        
        # Sign-up link
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("💡 **Default:** `admin` / `admin123`")
        with col_b:
            if st.button("📝 Create New Account", use_container_width=True):
                st.session_state.page = 'signup'
                st.rerun()


def signup_page():
    """Display sign-up/registration page"""
    st.title("🎥 YOLO Surveillance Platform")
    st.markdown("### Create a New Account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        # Sign-up form
        with st.form("signup_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_password = st.text_input("Password", type="password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            submit = st.form_submit_button("✨ Create Account")
            
            if submit:
                # Validation
                if not new_username or not new_password or not confirm_password:
                    st.warning("⚠ Please fill in all fields")
                elif len(new_username) < 3:
                    st.error("✗ Username must be at least 3 characters long")
                elif len(new_password) < 6:
                    st.error("✗ Password must be at least 6 characters long")
                elif new_password != confirm_password:
                    st.error("✗ Passwords do not match")
                else:
                    # Attempt to create user
                    if create_user(new_username, new_password):
                        st.success(f"✓ Account created successfully! Welcome, {new_username}!")
                        st.info("You can now login with your new credentials")
                        # Auto-login after successful registration
                        st.session_state.authenticated = True
                        st.session_state.username = new_username
                        st.session_state.page = 'home'
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("✗ Username already exists. Please choose a different username.")
        
        st.markdown("---")
        
        # Back to login link
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
        
        st.markdown("")
        st.markdown("**Password Requirements:**")
        st.markdown("- At least 6 characters long")
        st.markdown("- Username must be at least 3 characters")


def home_page():
    """Display home/dashboard page"""
    st.title(f"👋 Welcome, {st.session_state.username}!")
    st.markdown("### Parks Surveillance - Object Detection Platform")
    
    # Stats cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="result-card">
                <h3>🎯 Detection Ready</h3>
                <p>Upload images or videos for real-time object detection using YOLO</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="result-card">
                <h3>📊 Model Info</h3>
                <p>Using YOLO11n - Fast and accurate object detection</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        user_count = get_user_count()
        st.markdown(f"""
            <div class="result-card">
                <h3>👥 Users</h3>
                <p>{user_count} registered user(s) in the system</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Upload & Predict", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
    
    with col2:
        if st.button("📖 View Documentation", use_container_width=True):
            st.info("Check the INFERENCE_GUIDE.md and TRAINING_GUIDE.md files for detailed documentation")
    
    st.markdown("---")
    
    # Information section
    st.markdown("### ℹ️ About This Platform")
    st.markdown("""
    This platform provides:
    - **Real-time Object Detection** using YOLO11 models
    - **Image Processing** with bounding box annotations
    - **Video Processing** with frame-by-frame detection
    - **Secure Authentication** with encrypted password storage
    - **Easy File Upload** with drag-and-drop interface
    
    Navigate to the **Upload & Predict** section to start detecting objects in your images and videos!
    """)


def upload_page():
    """Display upload and prediction page"""
    st.title("📤 Upload & Predict")
    st.markdown("### Upload images or videos for object detection")
    
    # Media type selector
    media_type = st.radio("Select Media Type:", ["🖼️ Image", "🎬 Video"], horizontal=True)
    
    if media_type == "🖼️ Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, PNG"
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        st.markdown("#### 📄 File Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filename", file_details["Filename"])
        col2.metric("Size", file_details["File Size"])
        col3.metric("Type", file_details["File Type"])
        
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_image = file_extension in ['jpg', 'jpeg', 'png']
        is_video = file_extension in ['mp4', 'avi', 'mov']
        
        # Preview for images
        if is_image:
            st.markdown("#### 🖼️ Image Preview")
            st.image(uploaded_file, use_container_width=True)
        
        st.markdown("---")
        
        # Model settings
        col1, col2 = st.columns(2)
        with col1:
            model_path = st.selectbox(
                "Select YOLO Model",
                ["yolo11n.pt", "runs/train/parks_surveillance_training3/weights/best.pt"],
                help="Choose between pretrained YOLO11n or your custom trained model"
            )
        with col2:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Minimum confidence for detections"
            )
        
        # Prediction button
        if st.button("🎯 Run Prediction", use_container_width=True):
            # Save uploaded file
            with st.spinner("Saving file..."):
                file_path = save_uploaded_file(uploaded_file)
            
            try:
                if is_image:
                    # Image prediction
                    with st.spinner("🔍 Running object detection on image..."):
                        result = predict_image(file_path, model_path, conf_threshold)
                    
                    # Display results
                    st.markdown("### 🎉 Prediction Results")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Detections", result.num_detections)
                    col2.metric("Processing Time", f"{result.processing_time:.2f}s")
                    col3.metric("Confidence", f"{conf_threshold:.0%}")
                    
                    # Detection details
                    if result.num_detections > 0:
                        st.markdown("#### 📋 Detection Details")
                        for i, detection in enumerate(result.detections, 1):
                            st.markdown(f"""
                                **{i}. {detection['class']}** - Confidence: {detection['confidence']:.2%}
                            """)
                        
                        # Display annotated image
                        st.markdown("#### 🖼️ Annotated Image")
                        st.image(result.annotated_path, use_container_width=True)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(result.annotated_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download Annotated Image",
                                    data=file,
                                    file_name=f"detected_{uploaded_file.name}",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        with col2:
                            pdf_path = generate_pdf_report(result, uploaded_file.name, original_file_path=file_path, username=st.session_state.username)
                            if pdf_path and os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📄 Download PDF Report",
                                        data=pdf_file,
                                        file_name=f"report_{uploaded_file.name}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            else:
                                st.info("PDF generation requires 'fpdf' package. Install it via 'pip install fpdf'.")
                        
                    else:
                        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
                
                elif is_video:
                    # Video prediction
                    st.markdown("#### 🎬 Processing Video")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(frame_num, total_frames):
                        progress = frame_num / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_num}/{total_frames}")
                    
                    result = predict_video(file_path, model_path, conf_threshold, update_progress)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✓ Processing complete!")
                    
                    # Display results
                    st.markdown("### 🎉 Prediction Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Frames", result.total_frames)
                    col2.metric("Total Detections", result.num_detections)
                    col3.metric("Processing Time", f"{result.processing_time:.1f}s")
                    col4.metric("FPS", f"{result.total_frames/result.processing_time:.1f}")
                    
                    # Detection summary
                    if result.num_detections > 0:
                        st.markdown("#### 📊 Detection Summary")
                        for detection in result.detections:
                            st.markdown(f"**{detection['class']}**: {detection['count']} detections")
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(result.annotated_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download Annotated Video",
                                    data=file,
                                    file_name=f"detected_{uploaded_file.name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        with col2:
                            pdf_path = generate_pdf_report(result, uploaded_file.name, original_file_path=file_path, username=st.session_state.username)
                            if pdf_path and os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📄 Download PDF Report",
                                        data=pdf_file,
                                        file_name=f"report_{uploaded_file.name}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            else:
                                st.info("PDF generation requires 'fpdf' package. Install it via 'pip install fpdf'.")
                        
                        
                        st.success(f"✓ Video saved to: {result.annotated_path}")
                    else:
                        st.warning("⚠️ No objects detected in the video. Try lowering the confidence threshold.")
                
                # Cleanup old files
                cleanup_old_files("uploads", max_age_hours=24)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)




def get_bot_response(user_input):
    """Simple mock/rule-based chatbot backend"""
    user_input = user_input.lower()
    
    # Rule-based matching
    if any(word in user_input for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your Parks Surveillance AI Assistant. How can I help you?"
    elif 'detect' in user_input or 'upload' in user_input:
        return "To detect objects, navigate to the 'Upload & Predict' page from the sidebar, upload an image or video, and click 'Run Prediction'."
    elif 'model' in user_input or 'yolo' in user_input:
        return "This platform uses YOLO11 (You Only Look Once), a state-of-the-art real-time object detection system. We currently support YOLO11n for fast and accurate inference."
    elif 'report' in user_input or 'pdf' in user_input:
        return "After running a prediction, you can download a detailed PDF report using the 'Download PDF Report' button that appears below the results."
    elif 'unauthorized' in user_input or 'violation' in user_input:
        return "The system automatically classifies detections as authorized or unauthorized based on predefined rules (e.g., vandalism, unauthorized vehicles, unattended items). The PDF report will highlight any violations in red."
    else:
        return "I'm still learning! While I might not have a specific answer for that yet, you can try asking me about how to use the platform, what models are used, or how to generate reports."

def chatbot_page():
    """Display the AI Chatbot interface"""
    st.title("💬 AI Assistant")
    st.markdown("### Ask me anything about the platform or object detection!")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = get_bot_response(prompt)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main application logic"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎥 Navigation")
        
        if st.session_state.authenticated:
            st.success(f"✓ Logged in as **{st.session_state.username}**")
            st.markdown("---")
            
            # Navigation buttons
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 'home'
                st.rerun()
            
            if st.button("📤 Upload & Predict", use_container_width=True):
                st.session_state.page = 'upload'
                st.rerun()
            
            if st.button("💬 AI Assistant", use_container_width=True):
                st.session_state.page = 'chatbot'
                st.rerun()
            
            st.markdown("---")
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = 'login'
                st.rerun()
        else:
            st.info("Please login to access the platform")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [YOLO Documentation](https://docs.ultralytics.com)
        - Inference Guide
        - Training Guide
        """)
    
    # Main content area
    if not st.session_state.authenticated:
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()
    else:
        if st.session_state.page == 'home':
            home_page()
        elif st.session_state.page == 'upload':
            upload_page()
        elif st.session_state.page == 'chatbot':
            chatbot_page()


if __name__ == "__main__":
    main()
