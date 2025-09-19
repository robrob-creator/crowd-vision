from ultralytics import YOLO
import cv2
import argparse
import sys
import time
import os
import shutil
import subprocess
import numpy as np
from datetime import datetime, timezone

# Cloudinary optional imports will be loaded lazily when enabled
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
except ImportError:
    cloudinary = None

# Firebase optional imports will be loaded lazily when enabled
try:
    from firebase_utils import init_firebase
except ImportError:
    # Create a dummy function if firebase_utils is not available
    def init_firebase(*args, **kwargs):
        print("[WARN] Firebase utils not available")
        return None, None, None


def resolve_source(src: str | int) -> tuple[int | str, dict]:
    """Resolve the video source. Returns (source, open_params).
    - If src is a digit, use as webcam index.
    - If src looks like a YouTube URL and yt_dlp is available, resolve to direct stream URL.
    - For http/https/rtsp, prefer FFMPEG backend if available.
    """
    open_params = {}

    if isinstance(src, str) and src.isdigit():
        return int(src), open_params

    if isinstance(src, int):
        return src, open_params

    # Attempt to resolve ANY webpage URL to a direct stream using yt_dlp if installed
    if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
        # First try yt_dlp
        tried_resolvers = []
        try:
            from yt_dlp import YoutubeDL  # type: ignore
            tried_resolvers.append("yt-dlp")

            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                # Prefer formats commonly compatible with OpenCV+FFmpeg
                "format": "best[ext=mp4]/best/bestvideo+bestaudio/best",
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(src, download=False)
                if info.get("entries"):
                    info = info["entries"][0]
                stream_url = info.get("url")
                if stream_url:
                    print(f"[INFO] Resolved YouTube URL via yt-dlp: {src}")
                    src = stream_url
        except Exception as e:
            print(f"[WARN] yt_dlp failed to resolve URL: {e}")

        # Then try streamlink if installed and yt_dlp failed
        if (isinstance(src, str) and (src.startswith("http://") or src.startswith("https://"))
                and not src.startswith("rtmp://") and not src.startswith("rtsp://")):
            # If it still looks like a webpage (not .m3u8/.mp4) and streamlink is present, try it
            if not (src.endswith(".m3u8") or src.endswith(".mp4") or ".m3u8" in src or ".mp4" in src):
                if shutil.which("streamlink"):
                    tried_resolvers.append("streamlink")
                    try:
                        # Ask streamlink for the direct stream URL of the 'best' quality
                        out = subprocess.check_output([
                            "streamlink", "--stream-url", src, "best"
                        ], stderr=subprocess.STDOUT, text=True, timeout=15)
                        direct = out.strip()
                        if direct:
                            print(f"[INFO] Resolved URL via streamlink: {src}")
                            src = direct
                    except subprocess.CalledProcessError as e:
                        print(f"[WARN] streamlink failed: {e.output}")
                    except Exception as e:
                        print(f"[WARN] streamlink error: {e}")
                else:
                    print("[INFO] Tip: install streamlink for better live site support: brew install streamlink")

    # For URLs, hint to use FFMPEG backend
    if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://") or src.startswith("rtsp://")):
        open_params = {"apiPreference": cv2.CAP_FFMPEG}

    return src, open_params


def open_capture(src: str | int) -> cv2.VideoCapture:
    """Open video capture with proper URL resolution"""
    source, params = resolve_source(src)
    if params.get("apiPreference") is not None:
        cap = cv2.VideoCapture(source, params["apiPreference"])
    else:
        cap = cv2.VideoCapture(source)
    return cap


def main(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser(description="YOLO Live Stream Detection and Firebase Reporting")
        parser.add_argument('--source', type=str, required=True, help='Video source: webcam, file path, URL, or YouTube link')
        parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model to use for person detection')
        parser.add_argument('--garbage-model', type=str, default='models/garbage_detector.pt', help='Path to YOLO garbage detection weights')
        parser.add_argument('--location', type=str, required=True, help='Location of the camera/stream')
        parser.add_argument('--interval', type=int, default=120, help='Interval (seconds) to send metrics to Firebase')
        parser.add_argument('--garbage-conf', type=float, default=0.5, help='Confidence threshold for garbage detections')
        parser.add_argument('--person-conf', type=float, default=0.5, help='Confidence threshold for person detections')
        parser.add_argument('--firebase-method', type=str, choices=['firestore', 'realtime'], required=True, help='Firebase method: firestore or realtime')
        parser.add_argument('--firebase-credentials', type=str, required=True, help='Path to Firebase service account JSON')
        parser.add_argument('--firebase-path', type=str, required=True, help='Firebase DB path')
        parser.add_argument('--firebase-db-url', type=str, help='Firebase Realtime DB URL')
        parser.add_argument('--log-metrics', action='store_true', help='Print metrics to stdout')
        parser.add_argument('--cloudinary-cloud-name', type=str, help='Cloudinary cloud name for screenshot uploads')
        parser.add_argument('--cloudinary-api-key', type=str, help='Cloudinary API key for screenshot uploads')
        parser.add_argument('--cloudinary-api-secret', type=str, help='Cloudinary API secret for screenshot uploads')
        parser.add_argument('--process-every', type=int, default=1, help='Process every Nth frame (default: 1, process every frame)')
        parser.add_argument('--privacy', type=str, default='public', help='Privacy setting for the location (default: public)')
    else:
        from argparse import Namespace
        args = Namespace(**args_dict)

    print(f"[INFO] Starting detection for {args.location}")
    print(f"[INFO] Source: {args.source}")

    # Check for required dependencies
    try:
        import yt_dlp
        print("[INFO] yt-dlp available for YouTube/streaming support")
    except ImportError:
        print("[WARN] yt-dlp not installed. YouTube URLs may not work.")
        print("[HINT] Install with: pip install yt-dlp")

    # Check FFMPEG availability
    if shutil.which("ffmpeg"):
        print("[INFO] FFMPEG available for advanced video processing")
    else:
        print("[WARN] FFMPEG not found. Streaming may be limited.")
        print("[HINT] Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")

    # Load YOLO model for person detection
    try:
        person_model = YOLO(args.model)
        print("[INFO] Person detection model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load person model: {e}")
        return

    # Load YOLO model for garbage detection
    try:
        if args.garbage_model and os.path.exists(args.garbage_model):
            garbage_model = YOLO(args.garbage_model)
            print("[INFO] Garbage detection model loaded")
        else:
            garbage_model = YOLO('yolov8n.pt')  # fallback
            print("[WARN] Using default YOLO model for garbage detection")
    except Exception as e:
        print(f"[ERROR] Failed to load garbage model: {e}")
        return

    # Initialize Firebase if requested
    firebase_app, db_rtdb, firestore = None, None, None
    if args.firebase_credentials and os.path.exists(args.firebase_credentials):
        print(f"[INFO] Firebase credentials file exists: {args.firebase_credentials}")
        try:
            success, firebase_app, db_rtdb, firestore = init_firebase(args.firebase_method, args.firebase_credentials, args.firebase_db_url, app_name=f"{args.location}_app")
            if success:
                print(f"[INFO] Firebase initialized successfully - method: {args.firebase_method}, firestore: {firestore is not None}, db_rtdb: {db_rtdb is not None}")
            else:
                print("[WARN] Firebase initialization returned False")
        except Exception as e:
            print(f"[WARN] Firebase initialization failed: {e}")
    else:
        print(f"[WARN] Firebase credentials not found: {args.firebase_credentials}")

    # Initialize Cloudinary if credentials provided
    cloudinary_available = False
    if hasattr(args, 'cloudinary_cloud_name') and args.cloudinary_cloud_name:
        if cloudinary is not None:
            try:
                cloudinary.config(
                    cloud_name=args.cloudinary_cloud_name,
                    api_key=args.cloudinary_api_key,
                    api_secret=args.cloudinary_api_secret
                )
                cloudinary_available = True
                print("[INFO] Cloudinary initialized successfully")
            except Exception as e:
                print(f"[WARN] Cloudinary initialization failed: {e}")
        else:
            print("[WARN] Cloudinary library not available")
    else:
        print("[INFO] Cloudinary credentials not provided - screenshots disabled")

    # Open video source (skip if in test mode)
    cap = None
    if not args.test_mode:
        cap = open_capture(args.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {args.source}")

            # Provide specific guidance based on source type
            if isinstance(args.source, str) and ('youtube.com' in args.source or 'youtu.be' in args.source):
                print("[HINT] YouTube streams may not work reliably with OpenCV.")
                print("[HINT] Try using a direct RTMP/RTSP stream URL instead.")
                print("[HINT] Or use a local video file for testing.")
            elif isinstance(args.source, str) and args.source.startswith(('http://', 'https://')):
                print("[HINT] For web streams, ensure FFMPEG is available: brew install ffmpeg")
                print("[HINT] Try using streamlink: pip install streamlink")
            elif isinstance(args.source, int) or (isinstance(args.source, str) and args.source.isdigit()):
                print("[HINT] Webcam access failed. This is normal in headless/container environments.")
                print("[HINT] Try using a video file: python detectron_detector.py --source /path/to/video.mp4")
                print("[HINT] Or use --test-mode to test without video input.")
            else:
                print("[HINT] Check if the file exists and is a valid video format.")
                print("[HINT] Supported formats: MP4, AVI, MOV, etc.")

            return

        print("[INFO] Video source opened successfully")
    else:
        print("[INFO] Running in test mode (no video input required)")

    frame_count = 0
    person_count = 0
    garbage_count = 0
    min_people_count = 0
    max_people_count = 0
    min_garbage_count = 0
    max_garbage_count = 0
    max_people_in_frame = 0
    total_people_detected = 0
    processed_frames = 0
    last_metrics_time = time.time()
    
    # Variables for detection tracking
    prev_person_count = 0
    prev_garbage_count = 0
    latest_intruder_screenshot = None

    def upload_screenshot_to_cloudinary(frame, location, detection_type, count):
        """Upload a screenshot to Cloudinary when new detections occur"""
        if not cloudinary_available:
            return None
        
        try:
            # Create a temporary file for the screenshot using proper temp directory
            import tempfile
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            temp_filename = f"screenshot_{location.replace(' ', '_')}_{detection_type}_{count}_{timestamp}.jpg"
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save the frame as JPEG
            cv2.imwrite(temp_path, frame)
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                temp_path,
                folder=f"detections/{location.replace(' ', '_')}",
                public_id=f"{detection_type}_{count}_{timestamp}",
                resource_type="image"
            )
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            screenshot_url = upload_result.get('secure_url')
            print(f"[INFO] Screenshot uploaded to Cloudinary: {screenshot_url}")
            return screenshot_url
            
        except Exception as e:
            print(f"[ERROR] Failed to upload screenshot to Cloudinary: {e}")
            return None

    def check_data_watchers_mode():
        """Check if data watchers mode is enabled in Firebase Realtime Database"""
        if not db_rtdb or not firebase_app:
            return False

        try:
            # Check the data watchers mode setting in location-specific settings
            mode_ref = db_rtdb.reference(f'locations/{args.location}/settings/data_watchers_mode', app=firebase_app)
            mode_value = mode_ref.get()
            return mode_value is True
        except Exception as e:
            print(f"[WARN] Failed to check data watchers mode: {e}")
            return False

    try:
        while True:
            if args.test_mode:
                # Generate a test frame with some random content
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some test text
                cv2.putText(frame, f"Test Frame {frame_count}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Simulate some delay
                time.sleep(0.1)
            else:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(args.source, str) and args.source.startswith(('http://', 'https://')):
                        print("[WARN] Stream ended or connection lost, retrying...")
                        time.sleep(5)
                        cap = open_capture(args.source)
                        continue
                    else:
                        print("[WARN] Failed to read frame, retrying...")
                        time.sleep(1)
                        continue

            frame_count += 1

            # Process every Nth frame
            if frame_count % args.process_every != 0:
                continue

            # Detect persons with different confidence levels for min/max counts
            # High confidence (0.6) for minimum count - more conservative
            person_results_min = person_model(frame, conf=0.6, classes=[0])
            min_persons = 0
            min_person_confidences = []
            if person_results_min:
                for result in person_results_min:
                    boxes = result.boxes
                    min_persons += len(boxes)
                    for box in boxes:
                        min_person_confidences.append(float(box.conf[0]))

            # Low confidence (0.1) for maximum count - more inclusive
            person_results_max = person_model(frame, conf=0.3, classes=[0])
            max_persons = 0
            max_person_confidences = []
            if person_results_max:
                for result in person_results_max:
                    boxes = result.boxes
                    max_persons += len(boxes)
                    for box in boxes:
                        max_person_confidences.append(float(box.conf[0]))

            # Calculate average confidence levels
            avg_min_person_confidence = sum(min_person_confidences) / len(min_person_confidences) if min_person_confidences else 0.0
            avg_max_person_confidence = sum(max_person_confidences) / len(max_person_confidences) if max_person_confidences else 0.0

            # Use mid-range confidence for current count (existing logic)
            current_persons = min_persons  # Use conservative count as current
            current_person_confidence = avg_min_person_confidence  # Confidence of current count

            # Update max people tracking
            max_people_in_frame = max(max_people_in_frame, current_persons)
            total_people_detected += current_persons
            processed_frames += 1

            # Detect garbage with different confidence levels for min/max counts
            # High confidence (0.6) for minimum count - more conservative
            garbage_results_min = garbage_model(frame, conf=0.6)
            min_garbage = 0
            min_garbage_confidences = []
            if garbage_results_min:
                for result in garbage_results_min:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # Exclude person class
                            min_garbage += 1
                            min_garbage_confidences.append(float(box.conf[0]))

            # Low confidence (0.1) for maximum count - more inclusive
            garbage_results_max = garbage_model(frame, conf=0.3)
            max_garbage = 0
            max_garbage_confidences = []
            if garbage_results_max:
                for result in garbage_results_max:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # Exclude person class
                            max_garbage += 1
                            max_garbage_confidences.append(float(box.conf[0]))

            # Calculate average confidence levels for garbage
            avg_min_garbage_confidence = sum(min_garbage_confidences) / len(min_garbage_confidences) if min_garbage_confidences else 0.0
            avg_max_garbage_confidence = sum(max_garbage_confidences) / len(max_garbage_confidences) if max_garbage_confidences else 0.0

            # Use conservative count as current
            current_garbage = min_garbage
            current_garbage_confidence = avg_min_garbage_confidence  # Confidence of current count

            # Draw bounding boxes on frame for live feed
            display_frame = frame.copy()

            # Draw person detections (blue boxes) - use max confidence results to show all potential detections
            if person_results_max:
                for result in person_results_max:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Person {box.conf[0]:.2f}",
                                  (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw garbage detections (red boxes) - use max confidence results to show all potential detections
            if garbage_results_max:
                for result in garbage_results_max:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # Exclude person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_name = garbage_model.names[cls] if hasattr(garbage_model, 'names') else f"Class {cls}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"{class_name} {box.conf[0]:.2f}",
                                      (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Update counts
            person_count = current_persons
            garbage_count = current_garbage
            min_people_count = min_persons
            max_people_count = max_persons
            min_garbage_count = min_garbage
            max_garbage_count = max_garbage

            # Screenshot capture is now done only when Firebase metrics are sent (follows --interval)

            # Update previous counts for next iteration
            prev_person_count = person_count
            prev_garbage_count = garbage_count

            # Send metrics to Firebase periodically
            current_time = time.time()
            if current_time - last_metrics_time >= args.interval and firestore:
                print(f"[INFO] Interval reached ({args.interval}s), attempting Firebase save")
                try:
                    print("in saving  to firebase")
                    
                    # Capture screenshot only when sending metrics to Firebase (follows --interval)
                    data_watchers_enabled = check_data_watchers_mode() if db_rtdb else False
                    if cloudinary_available and db_rtdb and data_watchers_enabled and person_count > 0:
                        print(f"[INFO] Capturing intruder screenshot for Firebase metrics - {person_count} people")
                        screenshot_url = upload_screenshot_to_cloudinary(display_frame, args.location, 'intruder_screenshot', person_count)
                        if screenshot_url:
                            print(f"[INFO] Intruder detection screenshot: {screenshot_url}")
                            latest_intruder_screenshot = screenshot_url
                    
                    metrics = {
                        'currentNumberOfPeople': person_count,
                        'currentPeopleConfidence': round(current_person_confidence, 3),
                        'minPeopleCount': min_people_count,
                        'maxPeopleCount': max_people_count,
                        'minConfidenceLevel': 0.6,
                        'maxConfidenceLevel': 0.3,
                        'avgMinPeopleConfidence': round(avg_min_person_confidence, 3),
                        'avgMaxPeopleConfidence': round(avg_max_person_confidence, 3),
                        'previousNumberOfPeople': prev_person_count,
                        'countChanged': person_count != prev_person_count,
                        'intruder_screenshot': latest_intruder_screenshot,
                        'currentGarbage': garbage_count,
                        'currentGarbageConfidence': round(current_garbage_confidence, 3),
                        'minGarbageCount': min_garbage_count,
                        'maxGarbageCount': max_garbage_count,
                        'avgMinGarbageConfidence': round(avg_min_garbage_confidence, 3),
                        'avgMaxGarbageConfidence': round(avg_max_garbage_confidence, 3),
                        'lastUpdate': datetime.now(timezone.utc).isoformat(),
                        'location': args.location,
                        'privacy': args.privacy
                    }

                    print(f"[DEBUG] Attempting to send metrics to Firebase. Method: {args.firebase_method}, firestore: {firestore is not None}, db_rtdb: {db_rtdb is not None}")
                    print(f"[DEBUG] Current time: {current_time}, last_metrics_time: {last_metrics_time}, interval: {args.interval}")
                    print(f"[DEBUG] Time since last metrics: {current_time - last_metrics_time}")

                    if args.firebase_method == 'firestore' and firestore:
                        # Save current status to locations collection
                        print(f"[DEBUG] Using Firestore, saving current status to locations/{args.location.replace(' ', '_')}")
                        firestore.collection("locations").document(args.location.replace(' ', '_')).set(metrics)
                        
                        # Save log entry to logs collection
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                        doc_id = f"{args.location.replace(' ', '_')}_{timestamp}"
                        print(f"[DEBUG] Using Firestore, saving log to logs/{doc_id}")
                        firestore.collection("logs").document(doc_id).set(metrics)
                        print("[DEBUG] Firestore metrics and logs sent successfully")

                    elif args.firebase_method == 'realtime' and db_rtdb:
                        # Save current status to locations (merge with existing data to preserve settings)
                        print(f"[DEBUG] Using Realtime DB, updating current status to locations/{args.location.replace(' ', '_')}")
                        db_rtdb.reference(f"locations/{args.location.replace(' ', '_')}", app=firebase_app).update(metrics)
                        
                        # Save log entry to logs
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                        log_key = f"{args.location.replace(' ', '_')}_{timestamp}"
                        print(f"[DEBUG] Using Realtime DB, saving log to logs/{log_key}")
                        db_rtdb.reference(f"logs/{log_key}", app=firebase_app).set(metrics)
                        print("[DEBUG] Realtime DB metrics and logs sent successfully")
                    else:
                        print(f"[WARN] Firebase method {args.firebase_method} not supported or clients not available")

                    if args.log_metrics:
                        changed_status = "CHANGED" if person_count != prev_person_count else "UNCHANGED"
                        people_range = f"{min_people_count}-{max_people_count}" if min_people_count != max_people_count else str(min_people_count)
                        garbage_range = f"{min_garbage_count}-{max_garbage_count}" if min_garbage_count != max_garbage_count else str(min_garbage_count)
                        print(f"[METRICS] {args.location}: Current: {people_range} people (conf: {current_person_confidence:.3f}, min:0.6, max:0.3) (Prev: {prev_person_count}) [{changed_status}], {garbage_range} garbage items (conf: {current_garbage_confidence:.3f})")

                    last_metrics_time = current_time
                except Exception as e:
                    print(f"[ERROR] Failed to send metrics: {e}")
                    import traceback
                    traceback.print_exc()
            elif current_time - last_metrics_time >= args.interval:
                print(f"[DEBUG] Firebase not available - app: {firebase_app is not None}, firestore: {firestore is not None}, db_rtdb: {db_rtdb is not None}")
                print(f"[DEBUG] Firebase method: {args.firebase_method}, credentials: {args.firebase_credentials}")

            # Print status
            people_range = f"{min_people_count}-{max_people_count}" if min_people_count != max_people_count else str(min_people_count)
            garbage_range = f"{min_garbage_count}-{max_garbage_count}" if min_garbage_count != max_garbage_count else str(min_garbage_count)
            print(f"[STATUS] {args.location}: Frame {frame_count}, Current: {people_range} people (conf: {current_person_confidence:.3f}) (Prev: {prev_person_count}), {garbage_range} garbage items (conf: {current_garbage_confidence:.3f})")

            # Small delay to prevent overwhelming the system
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[INFO] Detection stopped by user")
    except Exception as e:
        print(f"[ERROR] Detection loop failed: {e}")
    finally:
        if cap is not None:
            cap.release()
        print(f"[INFO] Detection finished for {args.location}")


def run_detection(args_dict):
    """Entry point for the detection system"""
    main(args_dict)


if __name__ == "__main__":
    main()
