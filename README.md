# CrowdVision - Streamlit Deployment

A simplified, deployable version of CrowdVision for garbage detection using Streamlit.

## 🚀 Quick Start

### Local Development

1. **Clone or copy this folder** to your local machine

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push this folder** to a GitHub repository

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select this folder as the main file path
   - Set `app.py` as the main file
   - Click Deploy!

## 📁 Project Structure

```
streamlit-deploy/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/
│   └── garbage_detector.pt # Trained YOLOv8 model
└── README.md             # This file
```

## 🎯 Features

- **Image Upload**: Upload images to detect garbage
- **Video Processing**: Upload videos for batch processing
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
