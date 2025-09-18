import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="CrowdVision - Garbage Detection",
    page_icon="🗑️",
    layout="wide"
)

# Load YOLO model
@st.cache_resource
def load_garbage_model():
    """Load the garbage detection model"""
    try:
        # Try to load the trained model first
        model_path = "models/garbage_detector.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            st.success("✅ Loaded trained garbage detection model")
        else:
            # Fallback to base YOLOv8n
            model = YOLO("yolov8n.pt")
            st.warning("⚠️ Using base YOLOv8n model (limited garbage detection)")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

def detect_garbage(frame, model, conf_threshold=0.25):
    """Detect garbage in a frame"""
    if model is None:
        return frame, []

    try:
        # Run inference
        results = model(frame, conf=conf_threshold)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                           (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame, detections

    except Exception as e:
        st.error(f"Detection error: {e}")
        return frame, []

def main():
    st.title("🗑️ CrowdVision - Garbage Detection System")
    st.markdown("---")

    # Load model
    model = load_garbage_model()

    if model is None:
        st.error("Cannot proceed without a detection model")
        return

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects but may include false positives"
    )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📹 Input Source")

        input_type = st.radio(
            "Select input type:",
            ["Upload Image", "Webcam", "Video File"],
            help="Choose how you want to provide input for garbage detection"
        )

        frame = None
        detections = []

        if input_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to detect garbage in"
            )

            if uploaded_file is not None:
                # Convert uploaded file to numpy array
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Detect garbage
                    processed_frame, detections = detect_garbage(frame, model, conf_threshold)

                    # Display results
                    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                            caption="Detection Results",
                            use_column_width=True)

        elif input_type == "Webcam":
            st.write("Webcam detection would require additional setup for Streamlit deployment")
            st.info("💡 For production deployment, consider using a video stream URL or uploaded files")

        elif input_type == "Video File":
            uploaded_video = st.file_uploader(
                "Choose a video...",
                type=['mp4', 'avi', 'mov'],
                help="Upload a video to detect garbage in"
            )

            if uploaded_video is not None:
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name

                # Process video
                cap = cv2.VideoCapture(video_path)

                if cap.isOpened():
                    st.write("Processing video...")

                    # Get video info
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Progress bar
                    progress_bar = st.progress(0)
                    frame_count = 0

                    # Process every 30th frame for demo
                    frame_skip = 30

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        # Skip frames for performance
                        if frame_count % frame_skip != 0:
                            continue

                        # Detect garbage
                        processed_frame, frame_detections = detect_garbage(frame, model, conf_threshold)
                        detections.extend(frame_detections)

                        # Update progress
                        progress = min(frame_count / total_frames, 1.0)
                        progress_bar.progress(progress)

                        # Show current frame
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_count}",
                                use_column_width=True)

                        # Small delay to prevent overwhelming the app
                        time.sleep(0.1)

                    cap.release()
                    os.unlink(video_path)
                    progress_bar.empty()

                else:
                    st.error("Could not open video file")

    with col2:
        st.header("📊 Detection Results")

        if detections:
            st.success(f"🎯 Found {len(detections)} garbage items!")

            # Summary statistics
            class_counts = {}
            for det in detections:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            st.subheader("📈 Detection Summary")
            for class_name, count in class_counts.items():
                st.write(f"**{class_name}**: {count} items")

            # Detailed results
            st.subheader("📋 Detailed Detections")
            for i, det in enumerate(detections[:20]):  # Show first 20 detections
                with st.expander(f"Detection {i+1}: {det['class']}"):
                    st.write(f"**Class**: {det['class']}")
                    st.write(".3f")
                    st.write(f"**Bounding Box**: {det['bbox']}")

            if len(detections) > 20:
                st.info(f"Showing first 20 of {len(detections)} detections")

        else:
            st.info("No garbage detected yet. Try uploading an image or video!")

    # Footer
    st.markdown("---")
    st.markdown("### 🏗️ About CrowdVision")
    st.markdown("""
    CrowdVision is an AI-powered garbage detection system that helps monitor and track waste in public spaces.

    **Features:**
    - Real-time garbage detection using YOLOv8
    - Support for images and videos
    - Configurable confidence thresholds
    - Detailed detection analytics

    **Model**: Trained on TACO dataset with 60+ garbage categories
    """)

if __name__ == "__main__":
    main()