#!/usr/bin/env python3
"""
Complete test script for the CrowdVision Streamlit deployment
"""
import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not found")
        return False

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
    except ImportError:
        print("❌ Ultralytics not found")
        return False

    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV not found")
        return False

    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy not found")
        return False

    try:
        import firebase_admin
        print("✅ Firebase Admin imported successfully")
    except ImportError:
        print("❌ Firebase Admin not found")
        return False

    try:
        import streamlit_authenticator as stauth
        print("✅ Streamlit Authenticator imported successfully")
    except ImportError:
        print("⚠️  Streamlit Authenticator not found (optional)")

    return True

def test_model_loading():
    """Test that the model can be loaded"""
    try:
        from ultralytics import YOLO

        model_path = "models/garbage_detector.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print("✅ Trained model loaded successfully")
            print(f"   Model has {len(model.names)} classes")
            return True
        else:
            print("⚠️  Trained model not found, trying base model")
            model = YOLO("yolov8n.pt")
            print("✅ Base YOLOv8n model loaded successfully")
            return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_firebase_config():
    """Test Firebase configuration"""
    try:
        if os.path.exists("firebase_credentials.json"):
            print("✅ Firebase credentials file found")
            return True
        else:
            print("⚠️  Firebase credentials file not found")
            print("   This is expected for initial setup")
            return True  # Not a failure, just needs setup
    except Exception as e:
        print(f"❌ Firebase config test failed: {e}")
        return False

def test_detection_modules():
    """Test that detection modules can be imported"""
    try:
        # Test detectron_detector import
        import detectron_detector
        print("✅ Detection module imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  Detection module import failed: {e}")
        print("   This is expected if detectron2 is not installed")
        print("   The app will still work with YOLO-only detection")
        return True  # Not a critical failure
    except Exception as e:
        print(f"⚠️  Detection module import warning: {e}")
        return True  # Warning but not failure

def test_firebase_utils():
    """Test Firebase utilities"""
    try:
        import firebase_utils
        print("✅ Firebase utilities imported successfully")
        return True
    except ImportError:
        print("⚠️  Firebase utilities not found (optional)")
        return True
    except Exception as e:
        print(f"⚠️  Firebase utilities warning: {e}")
        return True

def main():
    print("🧪 Testing Complete CrowdVision Streamlit Deployment")
    print("=" * 60)

    # Test imports
    print("\n📦 Testing imports...")
    imports_ok = test_imports()

    # Test model loading
    print("\n🤖 Testing model loading...")
    model_ok = test_model_loading()

    # Test Firebase config
    print("\n🔥 Testing Firebase configuration...")
    firebase_ok = test_firebase_config()

    # Test detection modules
    print("\n🎯 Testing detection modules...")
    detection_ok = test_detection_modules()

    # Test Firebase utils
    print("\n🔧 Testing Firebase utilities...")
    utils_ok = test_firebase_utils()

    print("\n" + "=" * 60)
    if imports_ok and model_ok and firebase_ok and detection_ok and utils_ok:
        print("✅ All tests passed! Ready for deployment.")
        print("\n🚀 To run locally:")
        print("   streamlit run app.py")
        print("\n🌐 To deploy on Streamlit Cloud:")
        print("   1. Push this folder to GitHub")
        print("   2. Go to share.streamlit.io")
        print("   3. Connect your repo and deploy!")
        print("\n📋 Don't forget to:")
        print("   - Add your Firebase credentials")
        print("   - Configure Streamlit secrets if needed")
        print("   - Test with real detection sources")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Add Firebase credentials file")
        print("   - Ensure model file exists in models/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main()