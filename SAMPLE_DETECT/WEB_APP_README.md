# 🎥 YOLO Surveillance Web Application

A modern Streamlit-based web application for object detection in images and videos using YOLO models.

## ✨ Features

- 🔐 **Secure Authentication** - Login/logout with encrypted password storage
- 📝 **User Registration** - Create new accounts with validation
- 📤 **File Upload** - Drag-and-drop interface for images and videos
- 🎯 **Real-time Detection** - YOLO-powered object detection with confidence scores
- 📊 **Visual Results** - Annotated images/videos with bounding boxes
- ⬇️ **Download Results** - Save processed files to your device
- 🎨 **Modern UI** - Beautiful, responsive design with progress indicators

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Database

The database will be automatically initialized on first run, or you can manually initialize it:

```bash
python database.py
```

This creates:
- SQLite database (`surveillance.db`)
- Default admin user (username: `admin`, password: `admin123`)

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Login

1. Navigate to `http://localhost:8501`
2. **Option 1 - Use Default Account:**
   - **Username**: `admin`
   - **Password**: `admin123`
3. **Option 2 - Create New Account:**
   - Click "📝 Create New Account" button
   - Enter username (min 3 characters)
   - Enter password (min 6 characters)
   - Confirm password
   - Click "✨ Create Account"
   - You'll be automatically logged in after successful registration

### Upload & Predict

1. After login, click "Upload & Predict" from the sidebar or home page
2. Upload an image (JPG, PNG) or video (MP4, AVI, MOV)
3. Select YOLO model:
   - `yolo11n.pt` - Pretrained YOLO11 nano (fast)
   - `runs/train/parks_surveillance_training3/weights/best.pt` - Your custom trained model
4. Adjust confidence threshold (0.1 - 1.0)
5. Click "Run Prediction"
6. View results and download annotated files

### Supported File Types

**Images:**
- JPG, JPEG, PNG

**Videos:**
- MP4, AVI, MOV

**File Size Limit:** 200MB

## 📁 Project Structure

```
SAMPLE_DETECT/
├── streamlit_app.py          # Main Streamlit application
├── database.py                # SQLite authentication module
├── prediction_module.py       # YOLO prediction wrapper
├── surveillance.db            # SQLite database (created on first run)
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── uploads/                   # Temporary uploaded files
├── results/                   # Processed results (optional)
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration

### Streamlit Settings

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload size limits
- Server port
- Browser behavior

### Model Settings

In `streamlit_app.py`, you can modify:
- Default model path
- Confidence threshold range
- Supported file types

## 🛠️ Modules

### `streamlit_app.py`

Main application with:
- Authentication flow
- Home dashboard
- Upload and prediction interface
- Session state management
- Custom CSS styling

### `database.py`

Database operations:
- User authentication
- Password hashing (bcrypt)
- SQLite connection management
- User creation and verification

### `prediction_module.py`

YOLO prediction wrapper:
- Image inference
- Video inference with progress tracking
- File upload handling
- Result formatting
- Automatic cleanup of old files

## 🔐 Security Features

- **Password Hashing**: bcrypt with salt
- **Session Management**: Streamlit session state
- **File Validation**: Type and size checking
- **Automatic Cleanup**: Old files removed after 24 hours

## 📊 Performance

**Image Processing:**
- Typical: 1-5 seconds
- Depends on: Image size, model, hardware

**Video Processing:**
- Typical: 30-60 FPS on GPU, 5-10 FPS on CPU
- Depends on: Video length, resolution, model, hardware

## 🎨 UI Features

- **Gradient Buttons**: Modern purple gradient design
- **Progress Indicators**: Real-time processing feedback
- **Responsive Layout**: Works on desktop, tablet, mobile
- **Result Cards**: Clean, organized result display
- **Download Buttons**: Easy file downloads

## 🐛 Troubleshooting

### Database Issues

```bash
# Reinitialize database
python database.py
```

### Port Already in Use

```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### Model Not Found

Ensure YOLO model files exist:
- `yolo11n.pt` (downloads automatically on first use)
- Custom model at specified path

### Upload Fails

- Check file size (< 200MB)
- Verify file type is supported
- Ensure sufficient disk space

## 📝 Default Credentials

**Username:** `admin`  
**Password:** `admin123`

⚠️ **Important:** Change default credentials in production!

## 🔄 Adding New Users

Currently, users can be added programmatically:

```python
from database import create_user

# Create new user
success = create_user("newuser", "password123")
if success:
    print("User created successfully")
else:
    print("Username already exists")
```

## 📚 Related Documentation

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Detailed inference documentation
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Model training guide
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [Streamlit Documentation](https://docs.streamlit.io)
- [YOLO Documentation](https://docs.ultralytics.com)

## 🎯 Features Roadmap

Future enhancements:
- [ ] User registration page
- [ ] Password reset functionality
- [ ] Prediction history
- [ ] Batch file processing
- [ ] Real-time webcam detection
- [ ] Export results to CSV/JSON
- [ ] User roles and permissions

## 💡 Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
2. **Model Selection**: Use `yolo11n.pt` for speed, larger models for accuracy
3. **Confidence Threshold**: Lower for more detections, higher for precision
4. **File Cleanup**: Old uploads auto-delete after 24 hours

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review related documentation
3. Check YOLO/Streamlit documentation

---

**Built with:** Streamlit, YOLO11, SQLite, bcrypt  
**Version:** 1.0.0  
**Last Updated:** December 2025
