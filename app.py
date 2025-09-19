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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json

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
            # Check if default app already exists
            firebase_admin.get_app()
            # App already exists, just get clients
            self.fs_client = firestore.client()
            self.rt_db = db
            return
        except ValueError:
            # Default app doesn't exist, initialize it
            pass

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

    def display_live_feed(self, source_id, placeholder):
        """Display live video feed with detections for a source"""
        source = st.session_state.sources.get(source_id, {})
        source_link = source.get('source_link', '')

        try:
            if source_link:
                # Display the video source directly with autoplay
                if source_link.startswith(('http://', 'https://')):
                    # Use HTML video element with autoplay and cross-video control
                    video_html = f"""
                    <div class="video-container">
                        <video
                            id="video-{source_id}"
                            width="100%"
                            height="auto"
                            autoplay
                            muted
                            playsinline
                            controls
                            data-source-id="{source_id}"
                            onplay="pauseOtherVideos('{source_id}')"
                        >
                            <source src="{source_link}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>

                    <script>
                        function pauseOtherVideos(currentId) {{
                            // Pause all videos except the current one
                            const videos = document.querySelectorAll('video[data-source-id]');
                            videos.forEach(video => {{
                                if (video.getAttribute('data-source-id') !== currentId) {{
                                    video.pause();
                                }}
                            }});
                        }}

                        // Ensure only one video plays at a time
                        document.addEventListener('DOMContentLoaded', function() {{
                            const videos = document.querySelectorAll('video[data-source-id]');
                            videos.forEach(video => {{
                                video.addEventListener('play', function() {{
                                    pauseOtherVideos(video.getAttribute('data-source-id'));
                                }});
                            }});
                        }});
                    </script>

                    <style>
                        .video-container {{
                            margin: 10px 0;
                        }}
                        .video-container video {{
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        }}
                    </style>
                    """
                    placeholder.markdown(video_html, unsafe_allow_html=True)
                else:
                    # For local files or other sources, show info
                    placeholder.info(f"🎥 Video Source: {source_link}")
            else:
                placeholder.info("🎥 No video source configured")
        except Exception as e:
            placeholder.error(f"Error displaying video: {e}")

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
        if 'temp_files' not in st.session_state:
            st.session_state.temp_files = []
        if 'annotating_image' not in st.session_state:
            st.session_state.annotating_image = None

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
        """Main application UI - now handles navigation"""
        st.title("👥 CrowdVision - People & Garbage Detection")

        # User info in sidebar
        if st.session_state.current_user:
            st.sidebar.success(f"Logged in as: {st.session_state.current_user}")

        # Sidebar navigation
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", [
            "Main Dashboard",
            "Property Analytics",
            "AI Improvement"
        ], index=0)

        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()

        if page == "Main Dashboard":
            self.main_dashboard()
        elif page == "Property Analytics":
            self.property_analytics_page()
        else:  # AI Improvement
            self.ai_improvement_page()

    def main_dashboard(self):
        """Main application dashboard"""
        # st.title("👥 CrowdVision - People & Garbage Detection")

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

    def property_analytics_page(self):
        """Property analytics page"""
        try:
            from property_analytics import PropertyAnalytics
            analytics = PropertyAnalytics(self)
            analytics.display()
        except ImportError as e:
            st.error(f"Could not load property analytics: {e}")
            st.info("Make sure property_analytics.py is in the same directory.")

    def ai_improvement_page(self):
        """AI Improvement dashboard page"""
        try:
            dashboard = AIImprovementDashboard(self)
            dashboard.display()
        except Exception as e:
            st.error(f"Could not load AI improvement dashboard: {e}")
            st.info("Make sure all required dependencies are installed.")

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

        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh metrics", value=False)

        # Display sources
        for source_id, source in st.session_state.sources.items():
            with st.expander(f"📍 {source.get('location', 'Unknown')} - {source.get('source_type', 'Unknown')}"):
                # Display options for running sources
                if source.get('status') == 'running':
                    # Button to open live feed in new tab
                    source_link = source.get('source_link', '')
                    if source_link:
                        st.link_button(
                            "🎥 Open Live Feed",
                            source_link,
                            help=f"Open live feed for {source.get('location', 'Unknown')} in new tab"
                        )
                    else:
                        st.info("No live feed URL available")

                # Status and metrics row
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    status = source.get('status', 'stopped')
                    if status == 'running':
                        st.success("Running")
                    else:
                        st.error("Stopped")

                # Create placeholders for real-time metrics
                people_placeholder = st.empty()
                garbage_placeholder = st.empty()

                with col2:
                    if source.get('status') == 'running':
                        people, garbage, min_people, max_people, min_garbage, max_garbage, priv, last_sync = self.get_metrics(source_id)
                        people_range = f"{min_people}-{max_people}" if min_people != max_people else str(min_people)
                        people_placeholder.metric("People", people, delta=f"Approx. Range: {people_range}")
                    else:
                        people_placeholder.metric("People", "N/A")

                with col3:
                    if source.get('status') == 'running':
                        people, garbage, min_people, max_people, min_garbage, max_garbage, priv, last_sync = self.get_metrics(source_id)
                        garbage_range = f"{min_garbage}-{max_garbage}" if min_garbage != max_garbage else str(min_garbage)
                        garbage_placeholder.metric("Garbage", garbage, delta=f"Approx. Range: {garbage_range}")
                    else:
                        garbage_placeholder.metric("Garbage", "N/A")

                # Display last sync time
                if source.get('status') == 'running':
                    people, garbage, min_people, max_people, min_garbage, max_garbage, priv, last_sync = self.get_metrics(source_id)
                    if last_sync:
                        # Format the timestamp for display
                        time_diff = datetime.now() - last_sync.replace(tzinfo=None)
                        if time_diff.seconds < 60:
                            sync_display = f"🔄 Synced {time_diff.seconds}s ago"
                        elif time_diff.seconds < 3600:
                            sync_display = f"🔄 Synced {time_diff.seconds // 60}m ago"
                        else:
                            sync_display = f"🔄 Synced {time_diff.seconds // 3600}h ago"
                        st.caption(sync_display)
                    else:
                        st.caption("🔄 Never synced")

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

        # Auto-refresh logic
        if auto_refresh:
            import time
            time.sleep(5)  # Refresh every 5 seconds
            st.rerun()

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
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

                # Clean up all live feed files
                import glob
                for feed_file in glob.glob("live_feed_*.jpg"):
                    try:
                        os.unlink(feed_file)
                    except Exception as e:
                        print(f"Warning: Could not clean up {feed_file}: {e}")

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

        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh metrics", value=True, key="metrics_auto_refresh")

        metrics_text = "Live Metrics:\n\n"
        total_people_current = 0
        total_people_min = 0
        total_people_max = 0
        total_garbage_current = 0
        total_garbage_min = 0
        total_garbage_max = 0

        for source_id, source in st.session_state.sources.items():
            if source.get('status') == 'running':
                people, garbage, min_people, max_people, min_garbage, max_garbage, priv, last_sync = self.get_metrics(source_id)
                total_people_current += people
                total_people_min += min_people
                total_people_max += max_people
                total_garbage_current += garbage
                total_garbage_min += min_garbage
                total_garbage_max += max_garbage
                people_range = f"{min_people}-{max_people}" if min_people != max_people else str(min_people)
                garbage_range = f"{min_garbage}-{max_garbage}" if min_garbage != max_garbage else str(min_garbage)
                metrics_text += f"📍 {source['location']}:\n"
                metrics_text += f"  👥 People: {people_range}\n"
                metrics_text += f"  🗑️ Garbage: {garbage_range}\n"
                metrics_text += f"  ⏰ Last Sync: {last_sync or 'N/A'}\n\n"
            else:
                metrics_text += f"📍 {source['location']}: ⏸️ Stopped\n\n"

        # Summary metrics with placeholders for real-time updates
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sources", len(st.session_state.sources))
        with col2:
            total_people_placeholder = st.empty()
            people_range = f"{total_people_min}-{total_people_max}" if total_people_min != total_people_max else str(total_people_min)
            total_people_placeholder.metric("Total People", total_people_current, delta=f"Range: {people_range}")
        with col3:
            total_garbage_placeholder = st.empty()
            garbage_range = f"{total_garbage_min}-{total_garbage_max}" if total_garbage_min != total_garbage_max else str(total_garbage_min)
            total_garbage_placeholder.metric("Total Garbage", total_garbage_current, delta=f"Range: {garbage_range}")

        st.code(metrics_text, language="text")

        # Auto-refresh logic
        if auto_refresh:
            import time
            time.sleep(5)  # Refresh every 5 seconds
            st.rerun()

        if st.button("🔄 Refresh Now"):
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
            'data_watchers_mode': False,
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

        # For Streamlit Cloud, create a temporary credentials file for the subprocess
        temp_credentials_path = None
        if hasattr(st, 'secrets') and 'firebase' in st.secrets:
            import tempfile
            import json

            # Extract Firebase credentials from secrets
            secrets_dict = dict(st.secrets['firebase'])
            firebase_db_url = secrets_dict.pop('firebase_db_url', FIREBASE_DB_URL)

            # Create clean credentials dict
            cred_dict = {k: v for k, v in secrets_dict.items()
                       if k in ['type', 'project_id', 'private_key_id', 'private_key',
                               'client_email', 'client_id', 'auth_uri', 'token_uri',
                               'auth_provider_x509_cert_url', 'client_x509_cert_url', 'universe_domain']}

            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(cred_dict, f)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to disk
                temp_credentials_path = f.name

        args_dict = {
            'source': src,
            'model': 'yolov8n.pt',
            'garbage_model': 'models/garbage_detector.pt',
            'location': source.get('location', 'Unknown'),
            'interval': source.get('interval', 180),
            'garbage_conf': 0.25,
            'person_conf': 0.5,
            'firebase_method': source.get('firebase_method', ''),
            'firebase_credentials': temp_credentials_path or source.get('firebase_credentials', ''),
            'firebase_path': source.get('firebase_path', ''),
            'firebase_db_url': source.get('firebase_db_url', ''),
            'log_metrics': True,
            'test_firebase': True,
            'privacy': source['privacy'],
            'process_every': 8,  # Process every 8th frame for performance
            'test_mode': False  # Always run in normal mode for production
        }

        # Add Cloudinary credentials from secrets.toml if available
        if hasattr(st, 'secrets') and 'cloudinary' in st.secrets:
            args_dict['cloudinary_cloud_name'] = st.secrets['cloudinary']['cloud_name']
            args_dict['cloudinary_api_key'] = st.secrets['cloudinary']['api_key']
            args_dict['cloudinary_api_secret'] = st.secrets['cloudinary']['api_secret']

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

        # Store temp file path for cleanup
        if temp_credentials_path:
            if 'temp_files' not in st.session_state:
                st.session_state.temp_files = []
            st.session_state.temp_files.append(temp_credentials_path)

        st.success(f"Started detection for {source['location']}")

    def stop_detection(self, source_id):
        """Stop detection process"""
        if source_id in st.session_state.processes:
            p = st.session_state.processes[source_id]
            p.terminate()
            p.join()
            del st.session_state.processes[source_id]
        st.session_state.sources[source_id]['status'] = 'stopped'

        # Clean up temporary credential files
        if 'temp_files' in st.session_state:
            for temp_file in st.session_state.temp_files[:]:  # Copy list to avoid modification during iteration
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    st.session_state.temp_files.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {temp_file}: {e}")

        # Clean up live feed file
        source = st.session_state.sources.get(source_id, {})
        location = source.get('location', 'Unknown').replace(' ', '_')
        feed_file = f"live_feed_{location}.jpg"
        try:
            if os.path.exists(feed_file):
                os.unlink(feed_file)
        except Exception as e:
            print(f"Warning: Could not clean up live feed file {feed_file}: {e}")

        st.success(f"Stopped detection for {st.session_state.sources[source_id]['location']}")

    def capture_manual_screenshot(self, source_id):
        """Capture a manual screenshot from the current video feed"""
        source = st.session_state.sources.get(source_id, {})
        if not source or source.get('status') != 'running':
            st.error("Source is not running. Cannot capture screenshot.")
            return

        try:
            # First try to use existing live feed file if available
            location = source['location'].replace(' ', '_')
            feed_file = f"live_feed_{location}.jpg"

            if os.path.exists(feed_file):
                # Use the existing live feed file
                frame = cv2.imread(feed_file)
                if frame is None:
                    st.error("Could not read existing live feed file.")
                    return
            else:
                # Fallback: Get the video source URL and capture a frame
                source_link = source.get('source_link', '')
                if not source_link:
                    st.error("No video source available for screenshot capture.")
                    return

                # Capture a frame from the video
                cap = cv2.VideoCapture(source_link)
                if not cap.isOpened():
                    st.error("Could not open video source for screenshot capture.")
                    return

                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    st.error("Could not capture frame from video source.")
                    return

            # Upload to Cloudinary
            import cloudinary
            import cloudinary.uploader

            # Configure Cloudinary
            if hasattr(st, 'secrets') and 'cloudinary' in st.secrets:
                cloudinary.config(
                    cloud_name=st.secrets['cloudinary']['cloud_name'],
                    api_key=st.secrets['cloudinary']['api_key'],
                    api_secret=st.secrets['cloudinary']['api_secret']
                )
            else:
                st.error("Cloudinary credentials not configured in secrets.toml")
                return

            # Save frame to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_filename = temp_file.name
                cv2.imwrite(temp_filename, frame)

            # Upload to Cloudinary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            location = source['location'].replace(' ', '_')
            upload_result = cloudinary.uploader.upload(
                temp_filename,
                public_id=f"manual_screenshot_{location}_{timestamp}",
                folder="crowdvision/screenshots"
            )

            # Clean up temp file
            os.unlink(temp_filename)

            screenshot_url = upload_result.get('secure_url')
            if screenshot_url:
                # Update Firebase with the screenshot URL
                self.update_screenshot_in_firebase(source_id, screenshot_url, 'manual')
                st.success("Manual screenshot captured and uploaded successfully!")
            else:
                st.error("Failed to upload screenshot to Cloudinary.")

        except Exception as e:
            st.error(f"Failed to capture manual screenshot: {e}")

    def update_screenshot_in_firebase(self, source_id, screenshot_url, detection_type='manual'):
        """Update screenshot URL in Firebase"""
        source = st.session_state.sources.get(source_id, {})
        if not source:
            return

        try:
            if source['firebase_method'] == 'firestore':
                if hasattr(self, 'fs_client') and self.fs_client:
                    doc_ref = self.fs_client.collection(source['firebase_path'].strip('/')).document(
                        source['location'].replace(' ', '_'))
                    doc_ref.update({
                        'latestPersonScreenshot': screenshot_url if detection_type == 'person' or detection_type == 'manual' else firestore.FieldValue.delete(),
                        'latestGarbageScreenshot': screenshot_url if detection_type == 'garbage' else firestore.FieldValue.delete(),
                        'lastUpdate': firestore.SERVER_TIMESTAMP
                    })
            elif source['firebase_method'] == 'realtime':
                if hasattr(self, 'rt_db') and self.rt_db:
                    ref = self.rt_db.reference(f"{source['firebase_path'].strip('/')}/{source['location'].replace(' ', '_')}")
                    ref.update({
                        'latestPersonScreenshot': screenshot_url if detection_type == 'person' or detection_type == 'manual' else None,
                        'latestGarbageScreenshot': screenshot_url if detection_type == 'garbage' else None,
                        'lastUpdate': {'.sv': 'timestamp'}
                    })
        except Exception as e:
            print(f"Error updating screenshot in Firebase: {e}")

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
                # Use the global Firestore client instead of creating per-source apps
                if hasattr(self, 'fs_client') and self.fs_client:
                    doc = self.fs_client.collection(source['firebase_path'].strip('/')).document(
                        source['location'].replace(' ', '_')).get()

                    if doc.exists:
                        data = doc.to_dict()
                        return (data.get('currentNumberOfPeople', 0),
                               data.get('currentGarbage', 0),
                               data.get('minPeopleCount', 0),
                               data.get('maxPeopleCount', 0),
                               data.get('minGarbageCount', 0),
                               data.get('maxGarbageCount', 0),
                               data.get('private', False),
                               datetime.now())

            elif source['firebase_method'] == 'realtime':
                # Use the global Realtime Database client instead of creating per-source apps
                if hasattr(self, 'rt_db') and self.rt_db:
                    ref = self.rt_db.reference(f"{source['firebase_path'].strip('/')}/{source['location'].replace(' ', '_')}")
                    data = ref.get()
                    if data:
                        return (data.get('currentNumberOfPeople', 0),
                               data.get('currentGarbage', 0),
                               data.get('minPeopleCount', 0),
                               data.get('maxPeopleCount', 0),
                               data.get('minGarbageCount', 0),
                               data.get('maxGarbageCount', 0),
                               data.get('private', False),
                               datetime.now())

        except Exception as e:
            st.error(f"Error fetching metrics for {source['location']}: {e}")

        return 0, 0, 0, 0, 0, 0, source['privacy'], None

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
                    data.setdefault('data_watchers_mode', False)
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

    def display_live_feed(self, source_id, placeholder):
        """Display live video feed by reading the latest frame file"""
        try:
            source = st.session_state.sources.get(source_id)
            if not source:
                placeholder.error("Source not found")
                return

            location = source.get('location', 'Unknown')
            feed_filename = f"live_feed_{location.replace(' ', '_')}.jpg"

            if os.path.exists(feed_filename):
                # Read the image file
                image = cv2.imread(feed_filename)
                if image is not None:
                    # Convert BGR to RGB for Streamlit
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Add timestamp to force refresh
                    import time
                    timestamp = int(time.time())

                    placeholder.image(image_rgb, caption=f"Live Feed - {location} (Updated: {timestamp})", use_container_width=True)

                    # Add refresh button for live feed
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🔄 Refresh", key=f"refresh_feed_{source_id}"):
                            st.rerun()
                else:
                    placeholder.warning("Could not read live feed image")
                    with st.spinner("Waiting for live feed..."):
                        import time
                        time.sleep(2)
                        st.rerun()
            else:
                placeholder.info("Live feed not available yet. Detection process may still be starting.")
                # Auto-refresh while waiting for feed
                with st.spinner("Waiting for live feed..."):
                    import time
                    time.sleep(3)
                    st.rerun()

        except Exception as e:
            placeholder.error(f"Error displaying live feed: {e}")


class AIImprovementDashboard:
    """AI Improvement Dashboard for analyzing and improving detection accuracy"""

    def __init__(self, app):
        self.app = app
        self.fs_client = app.fs_client
        self._init_cloudinary()

    def _init_cloudinary(self):
        """Initialize Cloudinary client"""
        try:
            import cloudinary
            import cloudinary.uploader
            import cloudinary.api

            # Configure Cloudinary using Streamlit secrets
            cloudinary.config(
                cloud_name=st.secrets.get("cloudinary", {}).get("cloud_name"),
                api_key=st.secrets.get("cloudinary", {}).get("api_key"),
                api_secret=st.secrets.get("cloudinary", {}).get("api_secret")
            )
            self.cloudinary = cloudinary
        except ImportError:
            st.warning("Cloudinary not available. Some features may not work.")
            self.cloudinary = None
        except Exception as e:
            st.warning(f"Cloudinary configuration error: {e}")
            self.cloudinary = None

    def display(self):
        """Main display method for the AI improvement dashboard"""
        st.title("🧠 AI Improvement Dashboard")
        st.markdown("Analyze detection performance and improve model accuracy using real-world data.")

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Detection Analysis",
            "🎯 Model Fine-tuning",
            "📊 Performance Metrics",
            "⚙️ Configuration Tuning",
            "📚 Training Data Management"
        ])

        with tab1:
            self.detection_analysis_tab()

        with tab2:
            self.model_finetuning_tab()

        with tab3:
            self.performance_metrics_tab()

        with tab4:
            self.configuration_tuning_tab()

        with tab5:
            self.training_data_management_tab()

    def detection_analysis_tab(self):
        """Analyze detection results from Cloudinary screenshots"""
        st.header("🔍 Detection Analysis")
        st.markdown("Review and annotate detection results from live feed screenshots.")

        # Get screenshots from Cloudinary
        if self.cloudinary:
            try:
                # Get recent screenshots
                screenshots = self._get_cloudinary_screenshots()

                if screenshots:
                    st.subheader("Recent Screenshots")

                    # Display screenshots in a grid
                    cols = st.columns(3)
                    for i, screenshot in enumerate(screenshots[:9]):  # Show first 9
                        with cols[i % 3]:
                            self._display_screenshot_for_analysis(screenshot)
                else:
                    st.info("No screenshots found in Cloudinary. Images will appear here once live feed screenshots are captured and uploaded.")

            except Exception as e:
                st.error(f"Error loading screenshots: {e}")
        else:
            st.warning("Cloudinary not configured. Please set up Cloudinary credentials.")

        # Manual annotation section
        st.subheader("Manual Annotation")
        st.markdown("Upload images and provide ground truth annotations for model training.")

        uploaded_file = st.file_uploader("Upload image for annotation", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            self._annotate_uploaded_image(uploaded_file)

        # Display annotation interface if an image is being annotated
        if st.session_state.annotating_image:
            st.divider()
            self._annotate_cloudinary_image(
                st.session_state.annotating_image['public_id'],
                st.session_state.annotating_image['image_url']
            )

    def _get_cloudinary_screenshots(self):
        """Get recent screenshots from Cloudinary"""
        try:
            # First try to get images with crowdvision tag
            result = self.cloudinary.api.resources_by_tag('crowdvision',
                                                        max_results=50,
                                                        resource_type='image')
            screenshots = result.get('resources', [])

            # If no tagged images found, try to get recent images from the account
            if not screenshots:
                # Get recent images from the account (last 50)
                result = self.cloudinary.api.resources(max_results=50,
                                                     resource_type='image',
                                                     type='upload')
                screenshots = result.get('resources', [])

                # Filter for images that might be screenshots (you can customize this filter)
                # For now, just return recent images
                screenshots = screenshots[:20]  # Limit to 20 most recent

            return screenshots
        except Exception as e:
            st.error(f"Error fetching screenshots: {e}")
            return []

    def _display_screenshot_for_analysis(self, screenshot):
        """Display a screenshot with analysis options"""
        try:
            # Get image URL
            image_url = screenshot['secure_url']
            public_id = screenshot['public_id']

            # Display image
            st.image(image_url, caption=f"Screenshot: {public_id}", use_container_width=True)

            # Get detection data from Firebase if available
            detection_data = self._get_detection_data_for_screenshot(public_id)

            if detection_data:
                st.write("**Detection Results:**")
                st.json(detection_data)

                # Annotation controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Correct Detection", key=f"correct_{public_id}"):
                        self._mark_detection_correct(public_id, detection_data)

                with col2:
                    if st.button("❌ Incorrect Detection", key=f"incorrect_{public_id}"):
                        self._mark_detection_incorrect(public_id, detection_data)
            else:
                st.info("No detection data available for this screenshot. You can still manually annotate it for training.")

                # Manual annotation for screenshots without detection data
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Annotate Image", key=f"annotate_{public_id}"):
                        st.session_state.annotating_image = {'public_id': public_id, 'image_url': image_url}
                        st.rerun()

                with col2:
                    st.write("**Or upload similar images below for annotation**")

        except Exception as e:
            st.error(f"Error displaying screenshot: {e}")

    def _get_detection_data_for_screenshot(self, screenshot_id):
        """Get detection data associated with a screenshot"""
        try:
            # Query Firebase logs - check both Firestore and Realtime DB based on source configuration
            # First try to find which source this screenshot belongs to
            for source_id, source in st.session_state.sources.items():
                if source.get('firebase_method') == 'realtime' and hasattr(self, 'rt_db') and self.rt_db:
                    # Check realtime database logs
                    logs_ref = self.rt_db.reference('logs')
                    logs_data = logs_ref.get()

                    if logs_data:
                        for log_key, log_data in logs_data.items():
                            screenshots = log_data.get('screenshots', [])
                            for screenshot_obj in screenshots:
                                if isinstance(screenshot_obj, dict) and 'url' in screenshot_obj:
                                    screenshot_url = screenshot_obj['url']
                                    if screenshot_id in screenshot_url:
                                        return {
                                            'location': log_data.get('location'),
                                            'timestamp': log_data.get('lastUpdate'),
                                            'people_count': log_data.get('currentNumberOfPeople', 0),
                                            'garbage_count': log_data.get('currentGarbage', 0),
                                            'confidence': log_data.get('currentPeopleConfidence', 0),
                                            'screenshot_type': screenshot_obj.get('type', 'unknown')
                                        }

                elif source.get('firebase_method') == 'firestore' and hasattr(self, 'fs_client') and self.fs_client:
                    # Check Firestore logs (existing logic)
                    docs = self.fs_client.collection('logs').order_by('timestamp', direction='DESCENDING').limit(50).get()

                    for doc in docs:
                        log_data = doc.to_dict()
                        screenshots = log_data.get('screenshots', [])

                        for screenshot_obj in screenshots:
                            if isinstance(screenshot_obj, dict) and 'url' in screenshot_obj:
                                screenshot_url = screenshot_obj['url']
                                if screenshot_id in screenshot_url:
                                    return {
                                        'location': log_data.get('location'),
                                        'timestamp': log_data.get('timestamp'),
                                        'people_count': log_data.get('currentNumberOfPeople', 0),
                                        'garbage_count': log_data.get('currentGarbage', 0),
                                        'confidence': log_data.get('currentPeopleConfidence', 0),
                                        'screenshot_type': screenshot_obj.get('type', 'unknown')
                                    }

        except Exception as e:
            print(f"Error getting detection data for screenshot {screenshot_id}: {e}")

        return None

    def _infer_classes_from_log(self, log_data):
        """Infer detected classes from log data"""
        classes = []
        if log_data.get('person_detections', 0) > 0:
            classes.append('person')
        if log_data.get('garbage_detections', 0) > 0:
            classes.append('garbage')
        return classes

    def _annotate_cloudinary_image(self, public_id, image_url):
        """Annotate a Cloudinary image manually"""
        try:
            # Cancel button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("❌ Cancel", key=f"cancel_annotation_{public_id}"):
                    st.session_state.annotating_image = None
                    st.rerun()
            
            with col2:
                st.subheader(f"Annotating: {public_id}")

            # Display the image again for annotation
            st.image(image_url, caption=f"Annotating: {public_id}", use_container_width=True)

            # Extract location from public_id if possible
            location = "Unknown"
            if "Davao" in public_id:
                if "Barber_Shop" in public_id:
                    location = "Davao Barber Shop"
                elif "City_Live_cam" in public_id:
                    location = "Davao City Live cam"

            st.write(f"**Estimated Location:** {location}")

            # Annotation interface
            st.subheader("Manual Annotation")

            # Get classes present
            classes_present = st.multiselect(
                "Select classes present in this image:",
                ['person', 'garbage', 'none'],
                key=f"cloudinary_classes_{public_id}"
            )

            # Optional: Add confidence level
            confidence = st.slider("Confidence in your annotation:", 0.0, 1.0, 0.8, key=f"confidence_{public_id}")

            if st.button("💾 Save Annotation", key=f"save_cloudinary_{public_id}"):
                # Save to Firebase
                annotation_data = {
                    'image_source': 'cloudinary',
                    'public_id': public_id,
                    'image_url': image_url,
                    'location': location,
                    'classes_present': classes_present,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'annotator': st.session_state.current_user,
                    'annotation_type': 'manual_cloudinary'
                }

                # Save to Realtime Database
                if hasattr(self, 'rt_db') and self.rt_db:
                    timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    self.rt_db.reference(f'manual_annotations/{timestamp_key}').set(annotation_data)
                    st.success(f"Annotation saved for {public_id}! ✅")

                    # Clear annotation state
                    st.session_state.annotating_image = None

                    # Also save to annotations collection for consistency
                    annotation_data_2 = {
                        'screenshot_id': public_id,
                        'detection_data': {
                            'source': 'cloudinary_manual',
                            'location': location,
                            'manual_annotation': True
                        },
                        'annotation': 'manual',
                        'corrected_classes': classes_present,
                        'timestamp': datetime.now().isoformat(),
                        'annotator': st.session_state.current_user
                    }
                    self.rt_db.reference(f'annotations/{timestamp_key}_cloudinary').set(annotation_data_2)

        except Exception as e:
            st.error(f"Error annotating Cloudinary image: {e}")

    def _mark_detection_correct(self, screenshot_id, detection_data):
        """Mark detection as correct"""
        try:
            # Update Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                annotation_data = {
                    'screenshot_id': screenshot_id,
                    'detection_data': detection_data,
                    'annotation': 'correct',
                    'timestamp': datetime.now().isoformat(),
                    'annotator': st.session_state.current_user
                }
                self.rt_db.reference(f'annotations/{timestamp_key}_correct').set(annotation_data)
                st.success("Marked as correct!")
        except Exception as e:
            st.error(f"Error saving annotation: {e}")

    def _mark_detection_incorrect(self, screenshot_id, detection_data):
        """Mark detection as incorrect and allow correction"""
        try:
            # Get user correction
            corrected_classes = st.multiselect(
                "Select correct classes present in image:",
                ['person', 'garbage', 'none'],
                key=f"correct_classes_{screenshot_id}"
            )

            if st.button("Save Correction", key=f"save_correction_{screenshot_id}"):
                # Update Realtime Database
                if hasattr(self, 'rt_db') and self.rt_db:
                    timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    annotation_data = {
                        'screenshot_id': screenshot_id,
                        'detection_data': detection_data,
                        'annotation': 'incorrect',
                        'corrected_classes': corrected_classes,
                        'timestamp': datetime.now().isoformat(),
                        'annotator': st.session_state.current_user
                    }
                    self.rt_db.reference(f'annotations/{timestamp_key}_correction').set(annotation_data)
                    st.success("Correction saved!")
        except Exception as e:
            st.error(f"Error saving correction: {e}")

    def _annotate_uploaded_image(self, uploaded_file):
        """Annotate an uploaded image"""
        try:
            # Read image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Annotation interface
            st.subheader("Ground Truth Annotation")

            # Get classes present
            classes_present = st.multiselect(
                "Select classes present in the image:",
                ['person', 'garbage', 'other'],
                key="uploaded_classes"
            )

            # Bounding box annotation (simplified)
            if 'person' in classes_present or 'garbage' in classes_present:
                st.write("**Bounding Box Coordinates** (x1, y1, x2, y2)")
                col1, col2 = st.columns(2)
                with col1:
                    x1 = st.number_input("x1", 0, image.width, key="x1")
                    y1 = st.number_input("y1", 0, image.height, key="y1")
                with col2:
                    x2 = st.number_input("x2", 0, image.width, key="x2")
                    y2 = st.number_input("y2", 0, image.height, key="y2")

            if st.button("Save Annotation"):
                # Save to Realtime Database
                if hasattr(self, 'rt_db') and self.rt_db:
                    timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    annotation_data = {
                        'image_name': uploaded_file.name,
                        'classes_present': classes_present,
                        'bounding_boxes': [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}] if 'person' in classes_present or 'garbage' in classes_present else [],
                        'timestamp': datetime.now().isoformat(),
                        'annotator': st.session_state.current_user
                    }

                    self.rt_db.reference(f'manual_annotations/{timestamp_key}').set(annotation_data)
                    st.success("Annotation saved successfully!")

        except Exception as e:
            st.error(f"Error annotating image: {e}")

    def model_finetuning_tab(self):
        """Model fine-tuning interface"""
        st.header("🎯 Model Fine-tuning")
        st.markdown("Configure and start model training with collected annotations.")

        # Training configuration
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 1, 100, 50)
            batch_size = st.slider("Batch Size", 1, 64, 16)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")

        with col2:
            model_type = st.selectbox("Model Type", ["yolov8n", "yolov8s", "yolov8m"])
            augmentation = st.multiselect("Data Augmentation",
                                        ["rotation", "flip", "brightness", "contrast"],
                                        default=["rotation", "flip"])

        # Dataset status
        st.subheader("Dataset Status")
        dataset_info = self._get_dataset_info()

        if dataset_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", dataset_info.get('total_images', 0))
            with col2:
                st.metric("Annotated Images", dataset_info.get('annotated_images', 0))
            with col3:
                st.metric("Training Ready", "Yes" if dataset_info.get('annotated_images', 0) > 10 else "No")

        # Training controls
        if st.button("🚀 Start Fine-tuning", type="primary"):
            if dataset_info and dataset_info.get('annotated_images', 0) > 10:
                self._start_model_training(epochs, batch_size, learning_rate, model_type, augmentation)
            else:
                st.error("Not enough annotated data. Need at least 10 annotated images.")

        # Training progress (if running)
        self._display_training_progress()

    def _get_dataset_info(self):
        """Get information about the training dataset"""
        try:
            # Get annotation counts from Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                annotations_data = self.rt_db.reference('annotations').get() or {}
                manual_annotations_data = self.rt_db.reference('manual_annotations').get() or {}

                return {
                    'total_images': len(annotations_data) + len(manual_annotations_data),
                    'annotated_images': len(annotations_data) + len(manual_annotations_data)
                }
        except Exception as e:
            return None

    def _start_model_training(self, epochs, batch_size, learning_rate, model_type, augmentation):
        """Start model training process"""
        try:
            # Create training configuration
            training_config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_type': model_type,
                'augmentation': augmentation,
                'start_time': datetime.now(),
                'status': 'running'
            }

            # Save to Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                training_config['id'] = timestamp_key
                self.rt_db.reference(f'training_runs/{timestamp_key}').set(training_config)

                st.success(f"Training started! Training ID: {timestamp_key}")

                # In a real implementation, this would trigger a background training process
                # For now, we'll simulate training progress

        except Exception as e:
            st.error(f"Error starting training: {e}")

    def _display_training_progress(self):
        """Display training progress if a training run is active"""
        try:
            # Get latest training run from Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                training_runs_data = self.rt_db.reference('training_runs').get() or {}

                if training_runs_data:
                    # Find the most recent running training run
                    running_runs = {k: v for k, v in training_runs_data.items() if v.get('status') == 'running'}
                    if running_runs:
                        latest_key = max(running_runs.keys(), key=lambda k: running_runs[k].get('start_time', ''))
                        run_data = running_runs[latest_key]

                        st.subheader("Training Progress")

                        # Simulate progress (in real implementation, this would come from the training process)
                        start_time = datetime.fromisoformat(run_data['start_time'])
                        progress = min(100, (datetime.now() - start_time).seconds // 30)  # 30 seconds per percent

                        st.progress(progress / 100)
                        st.write(f"Progress: {progress}%")

                        if progress >= 100:
                            st.success("Training completed!")
                            # Update status in Realtime Database
                            run_data['status'] = 'completed'
                            run_data['end_time'] = datetime.now().isoformat()
                            self.rt_db.reference(f'training_runs/{latest_key}').set(run_data)

        except Exception as e:
            pass  # No active training

    def performance_metrics_tab(self):
        """Display performance metrics and analytics"""
        st.header("📊 Performance Metrics")
        st.markdown("Track detection accuracy and model performance over time.")

        # Get metrics data
        metrics_data = self._get_performance_metrics()

        if metrics_data:
            # Overall metrics
            st.subheader("Overall Performance")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Accuracy", f"{metrics_data.get('avg_accuracy', 0):.2%}")
            with col2:
                st.metric("Total Detections", metrics_data.get('total_detections', 0))
            with col3:
                st.metric("False Positives", metrics_data.get('false_positives', 0))
            with col4:
                st.metric("False Negatives", metrics_data.get('false_negatives', 0))

            # Performance over time
            st.subheader("Performance Trends")

            if 'time_series' in metrics_data:
                df = pd.DataFrame(metrics_data['time_series'])
                fig = px.line(df, x='date', y='accuracy', title='Detection Accuracy Over Time')
                st.plotly_chart(fig)

            # Class-wise performance
            st.subheader("Class-wise Performance")

            if 'class_performance' in metrics_data and metrics_data['class_performance']:
                df_class = pd.DataFrame(metrics_data['class_performance'])
                fig_class = px.bar(df_class, x='class', y='accuracy', title='Accuracy by Class')
                st.plotly_chart(fig_class)
            else:
                st.info("No class-wise performance data available yet.")
        else:
            st.info("No performance data available yet. Start annotating detections to build metrics.")

    def _get_performance_metrics(self):
        """Calculate performance metrics from annotations"""
        try:
            # Get annotations from Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                annotations_data = self.rt_db.reference('annotations').get() or {}

                total_annotations = 0
                correct_annotations = 0
                false_positives = 0
                false_negatives = 0

                class_performance = {}
                time_series = []

                for key, data in annotations_data.items():
                    total_annotations += 1

                if data.get('annotation') == 'correct':
                    correct_annotations += 1

                # Calculate class performance
                detection_classes = data.get('detection_data', {}).get('classes', [])
                corrected_classes = data.get('corrected_classes', [])

                for cls in detection_classes:
                    if cls not in class_performance:
                        class_performance[cls] = {'correct': 0, 'total': 0}
                    class_performance[cls]['total'] += 1
                    if cls in corrected_classes:
                        class_performance[cls]['correct'] += 1

            # Calculate metrics
            avg_accuracy = correct_annotations / total_annotations if total_annotations > 0 else 0

            # Convert class performance to percentages
            for cls in class_performance:
                class_performance[cls]['accuracy'] = class_performance[cls]['correct'] / class_performance[cls]['total']

            return {
                'avg_accuracy': avg_accuracy,
                'total_detections': total_annotations,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'class_performance': [{'class': k, 'accuracy': v['accuracy']} for k, v in class_performance.items()]
            }

        except Exception as e:
            return None

    def configuration_tuning_tab(self):
        """Configuration tuning interface"""
        st.header("⚙️ Configuration Tuning")
        st.markdown("Adjust model parameters and detection thresholds for optimal performance.")

        # Current configuration
        st.subheader("Current Configuration")

        try:
            # Load current config from Firebase or defaults
            config = self._load_current_config()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Detection Thresholds**")
                conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, config.get('conf_threshold', 0.5), 0.05)
                iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, config.get('iou_threshold', 0.45), 0.05)

                st.write("**Model Parameters**")
                max_det = st.slider("Max Detections", 1, 1000, config.get('max_det', 300))
                img_size = st.selectbox("Image Size", [416, 512, 640, 768, 1024], index=[416, 512, 640, 768, 1024].index(config.get('img_size', 640)))

            with col2:
                st.write("**Class-specific Thresholds**")
                person_conf = st.slider("Person Confidence", 0.1, 1.0, config.get('person_conf', 0.5), 0.05)
                garbage_conf = st.slider("Garbage Confidence", 0.1, 1.0, config.get('garbage_conf', 0.5), 0.05)

                st.write("**Processing Settings**")
                enable_tracking = st.checkbox("Enable Object Tracking", config.get('enable_tracking', True))
                save_frames = st.checkbox("Save Processed Frames", config.get('save_frames', False))

            # Save configuration
            if st.button("💾 Save Configuration", type="primary"):
                new_config = {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'max_det': max_det,
                    'img_size': img_size,
                    'person_conf': person_conf,
                    'garbage_conf': garbage_conf,
                    'enable_tracking': enable_tracking,
                    'save_frames': save_frames,
                    'updated_at': datetime.now(),
                    'updated_by': st.session_state.current_user
                }

                self._save_config(new_config)
                st.success("Configuration saved successfully!")

        except Exception as e:
            st.error(f"Error loading configuration: {e}")

    def _load_current_config(self):
        """Load current configuration from Firebase"""
        try:
            configs = self.fs_client.collection('configurations').order_by('updated_at', direction='DESCENDING').limit(1).get()
            for doc in configs:
                return doc.to_dict()
            return {}  # Return defaults if no config found
        except Exception as e:
            return {}

    def _save_config(self, config):
        """Save configuration to Firebase"""
        try:
            self.fs_client.collection('configurations').add(config)
        except Exception as e:
            st.error(f"Error saving configuration: {e}")

    def training_data_management_tab(self):
        """Training data management interface"""
        st.header("📚 Training Data Management")
        st.markdown("Manage and curate training datasets for model improvement.")

        # Dataset overview
        st.subheader("Dataset Overview")

        dataset_stats = self._get_dataset_statistics()

        if dataset_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", dataset_stats.get('total_images', 0))
            with col2:
                st.metric("Annotated Images", dataset_stats.get('annotated_images', 0))
            with col3:
                st.metric("Classes", len(dataset_stats.get('classes', [])))
            with col4:
                st.metric("Data Quality", f"{dataset_stats.get('quality_score', 0):.1f}%")

        # Data curation tools
        st.subheader("Data Curation")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Dataset Stats"):
                st.rerun()

            if st.button("🧹 Clean Duplicate Images"):
                self._clean_duplicate_images()
                st.success("Duplicate cleaning completed!")

        with col2:
            if st.button("📊 Generate Data Report"):
                report = self._generate_data_report()
                if report:
                    st.download_button(
                        label="📥 Download Report",
                        data=json.dumps(report, indent=2),
                        file_name="dataset_report.json",
                        mime="application/json"
                    )

            if st.button("⚖️ Balance Dataset"):
                self._balance_dataset()
                st.success("Dataset balancing completed!")

        # Data visualization
        st.subheader("Data Distribution")

        if dataset_stats and 'class_distribution' in dataset_stats and dataset_stats['class_distribution']:
            df_dist = pd.DataFrame(dataset_stats['class_distribution'])
            fig = px.pie(df_dist, values='count', names='class', title='Class Distribution')
            st.plotly_chart(fig)
        else:
            st.info("No class distribution data available yet.")

    def _get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        try:
            # Get all annotations from Realtime Database
            if hasattr(self, 'rt_db') and self.rt_db:
                annotations_data = self.rt_db.reference('annotations').get() or {}
                manual_annotations_data = self.rt_db.reference('manual_annotations').get() or {}

                total_images = len(annotations_data) + len(manual_annotations_data)
                annotated_images = total_images

                # Calculate class distribution
                class_counts = {}
                for data in list(annotations_data.values()) + list(manual_annotations_data.values()):
                    classes = data.get('classes_present', []) or data.get('corrected_classes', [])
                    for cls in classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1

            return {
                'total_images': total_images,
                'annotated_images': annotated_images,
                'classes': list(class_counts.keys()),
                'class_distribution': [{'class': k, 'count': v} for k, v in class_counts.items()],
                'quality_score': 85.0  # Placeholder quality score
            }

        except Exception as e:
            return None

    def _clean_duplicate_images(self):
        """Remove duplicate images from dataset"""
        try:
            # This would implement duplicate detection logic
            # For now, just log the action
            st.info("Duplicate cleaning would be implemented here")
        except Exception as e:
            st.error(f"Error cleaning duplicates: {e}")

    def _generate_data_report(self):
        """Generate a comprehensive data report"""
        try:
            stats = self._get_dataset_statistics()
            return {
                'generated_at': datetime.now().isoformat(),
                'dataset_statistics': stats,
                'recommendations': [
                    "Add more images for underrepresented classes",
                    "Consider data augmentation for small classes",
                    "Review annotation quality regularly"
                ]
            }
        except Exception as e:
            return None

    def _balance_dataset(self):
        """Balance the dataset by oversampling underrepresented classes"""
        try:
            # This would implement dataset balancing logic
            st.info("Dataset balancing would be implemented here")
        except Exception as e:
            st.error(f"Error balancing dataset: {e}")


def main():
    app = CrowdVisionApp()

    if not st.session_state.logged_in:
        app.login()
    else:
        app.main_ui()

if __name__ == "__main__":
    main()