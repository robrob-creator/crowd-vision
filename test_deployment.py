#!/usr/bin/env python3
"""
Simple test script for the Streamlit deployment
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

def main():
    print("🧪 Testing CrowdVision Streamlit Deployment")
    print("=" * 50)

    # Test imports
    print("\n📦 Testing imports...")
    imports_ok = test_imports()

    # Test model loading
    print("\n🤖 Testing model loading...")
    model_ok = test_model_loading()

    print("\n" + "=" * 50)
    if imports_ok and model_ok:
        print("✅ All tests passed! Ready for deployment.")
        print("\n🚀 To run locally:")
        print("   streamlit run app.py")
        print("\n🌐 To deploy on Streamlit Cloud:")
        print("   1. Push this folder to GitHub")
        print("   2. Go to share.streamlit.io")
        print("   3. Connect your repo and deploy!")
    else:
        print("❌ Some tests failed. Please check dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()