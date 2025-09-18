import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, db
from multiprocessing import Process
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Try to import streamlit-authenticator
try:
    import streamlit_authenticator as stauth
except ImportError:
    stauth = None

# Page configuration
st.set_page_config(
    page_title="CrowdVision - Garbage Detection",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Firebase configuration
FIREBASE_DB_URL = os.environ.get('FIREBASE_DB_URL', 'https://inventi-fc7cc-default-rtdb.firebaseio.com')

class CrowdVisionApp:
    def __init__(self):
        self.authenticator = self._init_authenticator()
        self._init_firebase()
        self._init_session_state()

    def _init_firebase(self):
        """Initialize Firebase connection"""
        try:
            firebase_admin.get_app()
        except ValueError:
            try:
                # Try Streamlit secrets first (for deployment)
                if hasattr(st, 'secrets') and 'firebase' in st.secrets:
                    import json
                    # Extract Firebase credentials, excluding non-credential fields
                    secrets_dict = dict(st.secrets['firebase'])
                    firebase_db_url = secrets_dict.pop('firebase_db_url', FIREBASE_DB_URL)

                    # Create clean credentials dict with only Firebase service account fields
                    cred_dict = {k: v for k, v in secrets_dict.items()
                               if k in ['type', 'project_id', 'private_key_id', 'private_key',
                                       'client_email', 'client_id', 'auth_uri', 'token_uri',
                                       'auth_provider_x509_cert_url', 'client_x509_cert_url', 'universe_domain']}

                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred, {'databaseURL': firebase_db_url})
                # Fallback to file-based credentials (for local development)
                elif os.path.exists('firebase_credentials.json'):
                    cred = credentials.Certificate('firebase_credentials.json')
                    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
                elif os.path.exists('config/firebase_credentials.json'):
                    cred = credentials.Certificate('config/firebase_credentials.json')
                    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
                else:
                    st.warning("Firebase credentials not found. Some features may not work.")
                    return
            except Exception as e:
                st.error(f"Failed to initialize Firebase: {e}")
                return

        self.fs_client = firestore.client()
        self.rt_db = db

    def _init_authenticator(self):
        """Initialize authentication"""
        if stauth is None:
            return None

        # Default credentials
        default_password = 'changeme'
        hasher = stauth.Hasher()
        credentials_data = {
            'usernames': {
                'admin': {
                    'name': 'Administrator',
                    'email': 'admin@crowdvision.com',
                    'password': hasher.hash(default_password),
                },
                'user': {
                    'name': 'User',
                    'email': 'user@crowdvision.com',
                    'password': hasher.hash('user123'),
                }
            }
        }
        return credentials_data

    def _init_session_state(self):
        """Initialize session state variables"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'sources' not in st.session_state:
            st.session_state.sources = self.load_sources_from_firebase()
        if 'processes' not in st.session_state:
            st.session_state.processes = {}
        if 'firebase_apps' not in st.session_state:
            st.session_state.firebase_apps = {}

    def login(self):
        """Login page"""
        st.title("🔐 CrowdVision Login")

        if not self.authenticator:
            st.warning("Authentication not available. Using default access.")
            if st.button("Continue as Guest"):
                st.session_state.logged_in = True
                st.session_state.current_user = "Guest User"
                st.rerun()
            return

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in self.authenticator['usernames']:
                stored_password = self.authenticator['usernames'][username]['password']
                hasher = stauth.Hasher()
                if hasher.check_pw(password, stored_password):
                    st.session_state.logged_in = True
                    st.session_state.current_user = self.authenticator['usernames'][username]['name']
                    st.success(f"Welcome, {st.session_state.current_user}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Invalid username or password")

    def main_ui(self):
        """Main application UI"""
        st.title("👥 CrowdVision - People & Garbage Detection")

        # User info in sidebar
        if st.session_state.current_user:
            st.sidebar.success(f"Logged in as: {st.session_state.current_user}")

        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📷 Quick Detection", "➕ Add Source", "📋 Active Sources", "📊 Live Metrics"])

        with tab1:
            self.quick_detection_tab()

        with tab2:
            self.add_source_tab()

        with tab3:
            self.sources_tab()

        with tab4:
            self.metrics_tab()

    def quick_detection_tab(self):
        """Quick detection tab for image/video upload"""
        st.header("🗑️ Quick Garbage Detection")

        # Load model
        model = load_garbage_model()

        if model is None:
            st.error("Cannot load detection model")
            return

        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        with col2:
            input_type = st.radio("Input Type", ["Image", "Video"], horizontal=True)

        if input_type == "Image":
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                # Process image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    processed_frame, detections = detect_garbage(frame, model, conf_threshold)
                    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                            caption="Detection Results")

                    if detections:
                        st.success(f"Found {len(detections)} garbage items!")
                        # Show detections table
                        det_data = []
                        for det in detections:
                            det_data.append({
                                "Class": det['class'],
                                "Confidence": ".3f",
                                "Location": f"({det['bbox'][0]}, {det['bbox'][1]})"
                            })
                        st.table(det_data)
                    else:
                        st.info("No garbage detected in the image.")

        else:  # Video
            uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            if uploaded_video:
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name

                # Process video
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    st.write("Processing video...")
                    progress_bar = st.progress(0)
                    frame_count = 0
                    detections = []

                    # Process every 30th frame
                    frame_skip = 30
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % frame_skip != 0:
                            continue

                        # Detect garbage
                        _, frame_detections = detect_garbage(frame, model, conf_threshold)
                        detections.extend(frame_detections)

                        # Update progress
                        progress = min(frame_count / total_frames, 1.0)
                        progress_bar.progress(progress)

                    cap.release()
                    os.unlink(video_path)
                    progress_bar.empty()

                    if detections:
                        st.success(f"Found {len(detections)} garbage items in video!")

                        # Group by class
                        class_counts = {}
                        for det in detections:
                            class_counts[det['class']] = class_counts.get(det['class'], 0) + 1

                        st.subheader("Detection Summary")
                        for cls, count in class_counts.items():
                            st.write(f"**{cls}**: {count} items")
                    else:
                        st.info("No garbage detected in the video.")
                else:
                    st.error("Could not open video file")

    def add_source_tab(self):
        """Add new detection source tab"""
        st.header("➕ Add New Detection Source")

        with st.form("add_source_form"):
            col1, col2 = st.columns(2)

            with col1:
                location = st.text_input("Location Name")
                source_type = st.selectbox("Source Type", ["Webcam", "File", "Live Stream", "YouTube"])
                source_link = st.text_input("Source Link")

            with col2:
                stream_link = st.text_input("Stream Link (Optional)")
                interval = st.slider("Update Interval (seconds)", 30, 3600, 180)
                privacy = st.checkbox("Private Location")

            # Firebase settings
            st.subheader("Firebase Settings")
            firebase_method = st.selectbox("Firebase Method", ["realtime", "firestore"])
            firebase_path = st.text_input("Firebase Path", "/locations")
            firebase_db_url = st.text_input("Firebase DB URL", FIREBASE_DB_URL)

            submitted = st.form_submit_button("Add Source")

            if submitted:
                if not location or not source_link:
                    st.error("Please fill in location and source link")
                    return

                source_id = self.add_source_to_data(
                    location, source_type, source_link, firebase_method,
                    'firebase_credentials.json', firebase_path, firebase_db_url,
                    stream_link, privacy, interval
                )

                st.success(f"Source added with ID: {source_id}")
                st.rerun()

    def sources_tab(self):
        """Active sources management tab"""
        st.header("📋 Active Detection Sources")

        if not st.session_state.sources:
            st.info("No sources added yet. Add a source in the 'Add Source' tab.")
            return

        # Display sources
        for source_id, source in st.session_state.sources.items():
            with st.expander(f"📍 {source.get('location', 'Unknown')} - {source.get('source_type', 'Unknown')}"):
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    status = source.get('status', 'stopped')
                    if status == 'running':
                        st.success("Running")
                    else:
                        st.error("Stopped")

                with col2:
                    if source.get('status') == 'running':
                        people, garbage, priv, last_sync = self.get_metrics(source_id)
                        st.metric("People", people)
                    else:
                        st.metric("People", "N/A")

                with col3:
                    if source.get('status') == 'running':
                        people, garbage, priv, last_sync = self.get_metrics(source_id)
                        st.metric("Garbage", garbage)
                    else:
                        st.metric("Garbage", "N/A")

                with col4:
                    if source.get('status') == 'running':
                        if st.button("⏹️ Stop", key=f"stop_{source_id}"):
                            self.stop_detection(source_id)
                            st.rerun()
                    else:
                        if st.button("▶️ Start", key=f"start_{source_id}"):
                            self.start_detection(source_id)
                            st.rerun()

                with col5:
                    if st.button("🗑️ Delete", key=f"delete_{source_id}"):
                        self.delete_source(source_id)
                        st.rerun()

        # Clear all sources
        if st.button("🗑️ Clear All Sources"):
            if st.warning("Are you sure you want to clear all sources?") and st.button("Confirm Clear"):
                # Stop all running processes
                for source_id in list(st.session_state.processes.keys()):
                    self.stop_detection(source_id)

                st.session_state.sources = {}
                st.session_state.processes = {}
                st.session_state.firebase_apps = {}

                # Clear from Firebase
                try:
                    docs = self.fs_client.collection('sources').stream()
                    for doc in docs:
                        doc.reference.delete()
                    st.success("All sources cleared!")
                except Exception as e:
                    st.error(f"Error clearing sources from Firebase: {e}")

    def metrics_tab(self):
        """Live metrics tab"""
        st.header("📊 Live Metrics")

        if not st.session_state.sources:
            st.info("No sources available for metrics.")
            return

        metrics_text = "Live Metrics:\n\n"
        total_people = 0
        total_garbage = 0

        for source_id, source in st.session_state.sources.items():
            if source.get('status') == 'running':
                people, garbage, priv, last_sync = self.get_metrics(source_id)
                total_people += people
                total_garbage += garbage
                metrics_text += f"📍 {source['location']}:\n"
                metrics_text += f"  👥 People: {people}\n"
                metrics_text += f"  🗑️ Garbage: {garbage}\n"
                metrics_text += f"  ⏰ Last Sync: {last_sync or 'N/A'}\n\n"
            else:
                metrics_text += f"📍 {source['location']}: ⏸️ Stopped\n\n"

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sources", len(st.session_state.sources))
        with col2:
            st.metric("Total People Detected", total_people)
        with col3:
            st.metric("Total Garbage Detected", total_garbage)

        st.code(metrics_text, language="text")

        if st.button("🔄 Refresh Metrics"):
            st.rerun()

    def add_source_to_data(self, location, source_type, source_link, firebase_method,
                          firebase_credentials, firebase_path, firebase_db_url,
                          stream_link, privacy, interval):
        """Add source to session state and Firebase"""
        source_id = str(uuid.uuid4())
        source = {
            'id': source_id,
            'location': location,
            'source_type': source_type,
            'source_link': source_link,
            'firebase_method': firebase_method,
            'firebase_credentials': firebase_credentials,
            'firebase_path': firebase_path,
            'firebase_db_url': firebase_db_url,
            'stream_link': stream_link,
            'privacy': privacy,
            'interval': interval,
            'status': 'stopped',
            'current_people': 0,
            'current_garbage': 0,
            'last_sync': None
        }
        st.session_state.sources[source_id] = source
        self.save_source_to_firebase(source_id, source)
        return source_id

    def start_detection(self, source_id):
        """Start detection process for a source"""
        source = st.session_state.sources[source_id]

        # Determine source
        if source['source_type'] == 'Webcam':
            src = source['source_link']
        else:
            src = source['source_link']

        args_dict = {
            'source': src,
            'model': 'yolov8n.pt',
            'garbage_model': 'models/garbage_detector.pt',
            'location': source.get('location', 'Unknown'),
            'interval': source.get('interval', 180),
            'garbage_conf': 0.25,
            'person_conf': 0.5,
            'firebase_method': source.get('firebase_method', ''),
            'firebase_credentials': source.get('firebase_credentials', ''),
            'firebase_path': source.get('firebase_path', ''),
            'firebase_db_url': source.get('firebase_db_url', ''),
            'log_metrics': True,
            'test_firebase': True,
            'privacy': source['privacy'],
            'process_every': 8  # Process every 8th frame for performance
        }

        # Import detection function
        try:
            from detectron_detector import run_detection
        except ImportError:
            st.error("Detection module not available. Please ensure detectron_detector.py is in the same directory.")
            return

        # Start process
        p = Process(target=run_detection, args=(args_dict,))
        p.start()
        st.session_state.processes[source_id] = p
        st.session_state.sources[source_id]['status'] = 'running'
        st.success(f"Started detection for {source['location']}")

    def stop_detection(self, source_id):
        """Stop detection process"""
        if source_id in st.session_state.processes:
            p = st.session_state.processes[source_id]
            p.terminate()
            p.join()
            del st.session_state.processes[source_id]
        st.session_state.sources[source_id]['status'] = 'stopped'
        st.success(f"Stopped detection for {st.session_state.sources[source_id]['location']}")

    def delete_source(self, source_id):
        """Delete a source"""
        if source_id in st.session_state.sources:
            del st.session_state.sources[source_id]
            self.delete_source_from_firebase(source_id)
            if source_id in st.session_state.processes:
                self.stop_detection(source_id)
            st.success("Source deleted!")

    def get_metrics(self, source_id):
        """Get metrics for a source from Firebase"""
        source = st.session_state.sources[source_id]

        try:
            if source['firebase_method'] == 'firestore':
                if source['firebase_credentials'] not in st.session_state.firebase_apps:
                    cred = credentials.Certificate(source['firebase_credentials'])
                    app = firebase_admin.initialize_app(cred, name=source_id)
                    st.session_state.firebase_apps[source['firebase_credentials']] = app
                else:
                    app = st.session_state.firebase_apps[source['firebase_credentials']]

                fs_client_app = firestore.client(app=app)
                doc = fs_client_app.collection(source['firebase_path'].strip('/')).document(
                    source['location'].replace(' ', '_')).get()

                if doc.exists:
                    data = doc.to_dict()
                    return (data.get('currentNumberOfPeople', 0),
                           data.get('currentGarbage', 0),
                           data.get('private', False),
                           datetime.now())

            elif source['firebase_method'] == 'realtime':
                if source['firebase_credentials'] not in st.session_state.firebase_apps:
                    cred = credentials.Certificate(source['firebase_credentials'])
                    app = firebase_admin.initialize_app(cred,
                                                      {'databaseURL': source['firebase_db_url']},
                                                      name=source_id)
                    st.session_state.firebase_apps[source['firebase_credentials']] = app
                else:
                    app = st.session_state.firebase_apps[source['firebase_credentials']]

                ref = db.reference(f"{source['firebase_path'].strip('/')}/{source['location'].replace(' ', '_')}",
                                 app=app)
                data = ref.get()
                if data:
                    return (data.get('currentNumberOfPeople', 0),
                           data.get('currentGarbage', 0),
                           data.get('private', False),
                           datetime.now())

        except Exception as e:
            st.error(f"Error fetching metrics for {source['location']}: {e}")

        return 0, 0, source['privacy'], None

    # Firebase methods
    def load_sources_from_firebase(self):
        """Load sources from Firebase"""
        sources = {}
        try:
            if hasattr(self, 'fs_client'):
                docs = self.fs_client.collection('sources').stream()
                for doc in docs:
                    data = doc.to_dict()
                    # Add defaults for missing keys
                    data.setdefault('id', doc.id)
                    data.setdefault('location', 'Unknown Location')
                    data.setdefault('source_type', 'Unknown')
                    data.setdefault('source_link', '')
                    data.setdefault('firebase_method', '')
                    data.setdefault('firebase_credentials', '')
                    data.setdefault('firebase_path', '')
                    data.setdefault('firebase_db_url', '')
                    data.setdefault('stream_link', '')
                    data.setdefault('privacy', False)
                    data.setdefault('interval', 180)
                    data.setdefault('status', 'stopped')
                    data.setdefault('current_people', 0)
                    data.setdefault('current_garbage', 0)
                    data.setdefault('last_sync', None)
                    sources[doc.id] = data
        except Exception as e:
            st.error(f"Error loading sources from Firebase: {e}")
        return sources

    def save_source_to_firebase(self, source_id, source):
        """Save source to Firebase"""
        try:
            if hasattr(self, 'fs_client'):
                self.fs_client.collection('sources').document(source_id).set(source)
        except Exception as e:
            st.error(f"Error saving source to Firebase: {e}")

    def delete_source_from_firebase(self, source_id):
        """Delete source from Firebase"""
        try:
            if hasattr(self, 'fs_client'):
                self.fs_client.collection('sources').document(source_id).delete()
        except Exception as e:
            st.error(f"Error deleting source from Firebase: {e}")

# Load YOLO model
@st.cache_resource
def load_garbage_model():
    """Load the garbage detection model"""
    try:
        model_path = "models/garbage_detector.pt"
        if os.path.exists(model_path):
            from ultralytics import YOLO
            model = YOLO(model_path)
            st.success("✅ Loaded trained garbage detection model")
        else:
            from ultralytics import YOLO
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
    app = CrowdVisionApp()

    if not st.session_state.logged_in:
        app.login()
    else:
        app.main_ui()

if __name__ == "__main__":
    main()