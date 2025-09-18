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
        parser.add_argument('--interval', type=int, default=15, help='Interval (seconds) to send metrics to Firebase')
        parser.add_argument('--garbage-conf', type=float, default=0.25, help='Confidence threshold for garbage detections')
        parser.add_argument('--person-conf', type=float, default=0.5, help='Confidence threshold for person detections')
        parser.add_argument('--firebase-method', type=str, choices=['firestore', 'realtime'], required=True, help='Firebase method: firestore or realtime')
        parser.add_argument('--firebase-credentials', type=str, required=True, help='Path to Firebase service account JSON')
        parser.add_argument('--firebase-path', type=str, required=True, help='Firebase DB path')
        parser.add_argument('--firebase-db-url', type=str, help='Firebase Realtime DB URL')
        parser.add_argument('--log-metrics', action='store_true', help='Print metrics to stdout')
        parser.add_argument('--privacy', action='store_true', help='Mark location as private')
        parser.add_argument('--process-every', type=int, default=8, help='Process every Nth frame')
        parser.add_argument('--test-mode', action='store_true', help='Run in test mode with generated frames (no video input needed)')
        args = parser.parse_args()
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
                print("[INFO] Firebase initialized successfully")
            else:
                print("[WARN] Firebase initialization returned False")
        except Exception as e:
            print(f"[WARN] Firebase initialization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] Firebase credentials file not found: {args.firebase_credentials}")

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
    last_metrics_time = time.time()

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

            # Detect persons (class 0 is person in COCO dataset)
            person_results = person_model(frame, conf=args.person_conf, classes=[0])
            current_persons = 0
            if person_results:
                for result in person_results:
                    current_persons += len(result.boxes)

            # Detect garbage (all classes except person)
            garbage_results = garbage_model(frame, conf=args.garbage_conf)
            current_garbage = 0
            if garbage_results:
                for result in garbage_results:
                    # Count all detections as garbage (excluding person class if it's the same model)
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # Exclude person class
                            current_garbage += 1

            # Draw bounding boxes on frame for live feed
            display_frame = frame.copy()

            # Draw person detections (blue boxes)
            if person_results:
                for result in person_results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Person {box.conf[0]:.2f}",
                                  (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw garbage detections (red boxes)
            if garbage_results:
                for result in garbage_results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # Exclude person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_name = garbage_model.names[cls] if hasattr(garbage_model, 'names') else f"Class {cls}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"{class_name} {box.conf[0]:.2f}",
                                      (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Save processed frame for live feed
            feed_filename = f"live_feed_{args.location.replace(' ', '_')}.jpg"
            try:
                cv2.imwrite(feed_filename, display_frame)
            except Exception as e:
                print(f"[WARN] Could not save live feed frame: {e}")

            # Update counts
            person_count = current_persons
            garbage_count = current_garbage

            # Send metrics to Firebase periodically
            current_time = time.time()
            if current_time - last_metrics_time >= args.interval:
                if firebase_app:
                    try:
                        metrics = {
                            'currentNumberOfPeople': person_count,
                            'currentGarbage': garbage_count,
                            'lastUpdate': datetime.now(timezone.utc).isoformat(),
                            'location': args.location,
                            'privacy': args.privacy
                        }

                        print(f"[DEBUG] Attempting to send metrics to Firebase. Method: {args.firebase_method}, firestore: {firestore is not None}, db_rtdb: {db_rtdb is not None}")

                        if args.firebase_method == 'firestore' and firestore:
                            print(f"[DEBUG] Using Firestore, path: {args.firebase_path.strip('/')}, doc: {args.location.replace(' ', '_')}")
                            firestore.collection(args.firebase_path.strip("/")).document(args.location.replace(' ', '_')).set(metrics)
                            print("[DEBUG] Firestore metrics sent successfully")
                        elif args.firebase_method == 'realtime' and db_rtdb:
                            print(f"[DEBUG] Using Realtime DB, path: {args.firebase_path.strip('/')}")
                            db_rtdb.reference(f"{args.firebase_path.strip('/')}/{args.location.replace(' ', '_')}", app=firebase_app).update(metrics)
                            print("[DEBUG] Realtime DB metrics sent successfully")
                        else:
                            print(f"[WARN] Firebase method {args.firebase_method} not supported or clients not available")

                        if args.log_metrics:
                            print(f"[METRICS] {args.location}: {person_count} people, {garbage_count} garbage items")

                        last_metrics_time = current_time
                    except Exception as e:
                        print(f"[ERROR] Failed to send metrics: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[DEBUG] Firebase not available - app: {firebase_app is not None}, firestore: {firestore is not None}, db_rtdb: {db_rtdb is not None}")

            # Print status
            print(f"[STATUS] {args.location}: Frame {frame_count}, {person_count} people, {garbage_count} garbage items")

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
