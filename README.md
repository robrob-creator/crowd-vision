# CrowdVision - Complete Streamlit Deployment

A complete, production-ready Streamlit deployment of CrowdVision with Firebase integration, source management, and real-time detection capabilities.

## 🚀 Features

- **🔐 Authentication**: Secure login system with user management
- **📷 Quick Detection**: Upload images/videos for instant garbage detection
- **🎛️ Source Management**: Add, configure, and manage detection sources
- **📊 Live Metrics**: Real-time monitoring and analytics
- **🔥 Firebase Integration**: Cloud storage for sources and metrics
- **🤖 AI Detection**: YOLOv8 + Detectron2 for people and garbage detection
- **� YouTube Support**: Direct YouTube URL processing with yt-dlp integration
- **�📱 Responsive UI**: Modern Streamlit interface with tabs and controls

## � Project Structure

```
streamlit-deploy/
├── app.py                    # Main Streamlit application
├── detectron_detector.py     # Detection engine (people + garbage)
├── firebase_utils.py         # Firebase utilities
├── firebase_credentials.json # Firebase service account (ADD YOURS)
├── requirements.txt          # Python dependencies
├── models/
│   └── garbage_detector.pt   # Trained YOLO model (6.5MB)
├── packages.txt             # System packages for Streamlit Cloud
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🛠️ Setup & Installation

### 1. Environment Setup

```bash
# Clone or download this folder
cd streamlit-deploy/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Firebase Configuration

**For Local Development:**

1. Download your Firebase service account JSON from Firebase Console
2. Save it as `firebase_credentials.json` in the project root

**For Streamlit Cloud Deployment:**

- Use Streamlit secrets (see deployment section below)
- The app will automatically detect and use Streamlit secrets when deployed

**Environment Variables** (Optional):

```bash
export FIREBASE_DB_URL="https://your-project.firebaseio.com"
```

### 3. Model Setup

The trained garbage detection model is included. If you want to use your own:

```bash
# Place your model in models/garbage_detector.pt
cp /path/to/your/model.pt models/garbage_detector.pt
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:

   ```bash
   git init
   git add .
   git commit -m "CrowdVision Streamlit deployment"
   git remote add origin https://github.com/yourusername/crowdvision.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path to: `app.py`
   - Add secrets (see below)
   - Click **Deploy**!

3. **Streamlit Secrets** (in Streamlit Cloud dashboard):

   ```toml
   [firebase]
   # Copy your entire firebase_credentials.json content here
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = """
   -----BEGIN PRIVATE KEY-----
   YOUR_PRIVATE_KEY_HERE
   -----END PRIVATE KEY-----
   """
   client_email = "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project.iam.gserviceaccount.com"
   universe_domain = "googleapis.com"

   # Optional: Override default Firebase DB URL
   firebase_db_url = "https://your-project.firebaseio.com"
   ```

   **How to configure secrets in Streamlit Cloud:**

   - Go to your app dashboard
   - Click "⋮" → "Settings" → "Secrets"
   - Copy the above format and paste your actual Firebase credentials
   - Save and redeploy

### Option 2: Local Development

```bash
# Run locally
streamlit run app.py

# Access at http://localhost:8501
```

### Option 3: Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🎯 How to Use

### 1. Login

- **Default Users**:
  - Username: `admin`, Password: `changeme`
  - Username: `user`, Password: `user123`

### 2. Quick Detection Tab

- Upload images or videos for instant detection
- Adjust confidence threshold for sensitivity
- View detailed results and analytics

### 3. Add Source Tab

- Configure detection sources (cameras, streams, files)
- Set Firebase integration for data storage
- Configure update intervals and privacy settings

### 4. Active Sources Tab

- View all configured sources
- Start/stop detection processes
- Monitor real-time status and metrics

### 5. Live Metrics Tab

- View aggregated metrics across all sources
- Real-time people and garbage counts
- Last sync timestamps

## � Configuration

### Detection Parameters

- **Garbage Confidence**: 0.25 (adjustable)
- **People Detection**: Detectron2 COCO model (optional)
- **Processing Interval**: Configurable per source
- **Frame Skipping**: Automatic for performance

**Note**: The app works perfectly with YOLO-only detection if Detectron2 is not available. People detection will be disabled but garbage detection remains fully functional.

### Supported Source Types

The application supports multiple video source types:

- **Webcam**: Local camera access (index 0, 1, etc.)
- **File**: Local video files (MP4, AVI, MOV)
- **Live Stream**: RTSP/HTTP streaming URLs
- **YouTube**: Direct YouTube video URLs (automatically resolved using yt-dlp)

**YouTube Support**: Simply paste any YouTube URL (youtu.be or youtube.com) and the system will automatically resolve it to a direct video stream for processing.

### Firebase Structure

```
Firebase Realtime Database:
/locations/{source_name}/
  ├── currentNumberOfPeople: number
  ├── currentGarbage: number
  ├── private: boolean
  └── lastUpdate: timestamp

/sources/{source_id}/
  ├── location: string
  ├── source_type: string
  ├── status: "running"|"stopped"
  └── config: {...}
```

## 🐛 Troubleshooting

### Common Issues

1. **Firebase Connection Failed**:

   - Check `firebase_credentials.json` is valid
   - Verify Firebase project permissions
   - Check network connectivity

2. **Model Loading Failed**:

   - Ensure `models/garbage_detector.pt` exists
   - Check ultralytics version compatibility
   - App falls back to YOLOv8n if needed

3. **Detection Not Working**:

   - Check source URLs are accessible
   - Verify camera permissions
   - Check processing intervals

4. **YouTube URL Issues**:

   - Ensure yt-dlp is installed (`pip install yt-dlp`)
   - Check YouTube URL is valid and publicly accessible
   - Some private/restricted videos may not work
   - Network connectivity required for URL resolution

5. **Streamlit Cloud Issues**:
   - Ensure all files are committed to Git
   - Check package limits (model size ~7MB)
   - Monitor logs in Streamlit Cloud dashboard

### Performance Optimization

- **Video Processing**: Skips frames for speed
- **Model Caching**: Uses Streamlit's caching
- **Process Management**: Isolated detection processes
- **Memory Management**: Automatic cleanup

## 📊 System Requirements

### Minimum

- Python 3.8+
- 4GB RAM
- 2GB storage

### Recommended

- Python 3.9+
- 8GB RAM
- Stable internet for Firebase

## 🔒 Security Notes

- Change default passwords in production
- Use environment variables for sensitive data
- Firebase credentials should be kept secure
- Consider user authentication for multi-user deployments

## 📞 Support & Development

### Local Testing

```bash
# Test deployment
python test_deployment.py

# Run with debug
streamlit run app.py --logger.level=debug
```

### Adding New Features

- Detection logic in `detectron_detector.py`
- UI components in `app.py`
- Firebase operations in `firebase_utils.py`

## 📈 Monitoring & Analytics

- **Real-time Metrics**: Firebase integration
- **Detection Statistics**: Class-wise counts
- **Performance Monitoring**: Processing times
- **Error Logging**: Streamlit logs

## 🎉 Success Metrics

✅ **Complete CrowdVision functionality**
✅ **Firebase integration working**
✅ **Real-time detection processes**
✅ **Responsive web interface**
✅ **Production-ready deployment**

---

**Ready to deploy?** Push to GitHub and launch on Streamlit Cloud! 🚀

- **Real-time Detection**: Uses YOLOv8 for accurate garbage detection
- **Configurable Thresholds**: Adjust confidence levels for detection
- **Detailed Analytics**: View detection results and statistics

## 🔧 Configuration

### Model Configuration

The app automatically loads the trained model from `models/garbage_detector.pt`. If the model file is not found, it falls back to the base YOLOv8n model.

### Detection Parameters

- **Confidence Threshold**: Adjustable from 0.1 to 1.0 (default: 0.25)
- Lower values detect more objects but may include false positives
- Higher values are more precise but may miss detections

## 🗑️ Garbage Categories

The model is trained to detect 60+ types of garbage including:

- Plastic bottles and caps
- Aluminum foil and cans
- Cardboard and paper products
- Food containers and packaging
- Various plastic items
- And many more...

## 🚀 Deployment Notes

### File Size Considerations

- The model file (`garbage_detector.pt`) is approximately 50MB
- Consider using Streamlit Cloud's paid plans for larger deployments
- You can reduce model size by using quantization or smaller YOLO variants

### Performance Optimization

- Video processing skips frames for better performance
- Images are processed in real-time
- Consider using GPU instances for better performance

### Security Considerations

- This demo version doesn't include authentication
- For production use, consider adding user authentication
- Be mindful of uploaded file sizes and processing limits

## 🐛 Troubleshooting

### Model Loading Issues

If the model fails to load:

1. Ensure `models/garbage_detector.pt` exists
2. Check that ultralytics is properly installed
3. The app will fallback to YOLOv8n if needed

### Video Processing Issues

For video upload problems:

1. Ensure videos are in supported formats (MP4, AVI, MOV)
2. Check file size limits (Streamlit has upload limits)
3. Video processing may be slow for long videos

### Performance Issues

If the app is slow:

1. Reduce video frame processing frequency
2. Use smaller images
3. Consider using a more powerful deployment platform

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review Streamlit Cloud documentation
3. Ensure all dependencies are correctly installed

## 📄 License

This project is part of CrowdVision - an open-source garbage detection system.
