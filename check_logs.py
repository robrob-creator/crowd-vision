#!/usr/bin/env python3
"""
Quick script to check Firebase logs for screenshots
"""
import firebase_admin
from firebase_admin import credentials, firestore
import os

def check_firebase_logs():
    """Check Firebase logs for screenshot data"""
    try:
        # Initialize Firebase
        cred_path = "config/firebase_credentials.json"
        if not os.path.exists(cred_path):
            print("❌ Firebase credentials not found")
            return

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

        # Get Firestore client
        db = firestore.client()

        # Query logs collection
        logs_ref = db.collection('logs')
        docs = logs_ref.limit(10).get()

        print(f"Found {len(docs)} log documents")

        screenshot_count = 0
        for doc in docs:
            data = doc.to_dict()
            screenshots = data.get('screenshots', [])
            if screenshots:
                screenshot_count += 1
                print(f"✅ Log {doc.id} has {len(screenshots)} screenshots")
                for screenshot in screenshots:
                    print(f"   - {screenshot.get('type', 'unknown')}: {screenshot.get('url', 'no url')}")
            else:
                print(f"❌ Log {doc.id} has no screenshots")

        print(f"\nSummary: {screenshot_count}/{len(docs)} logs have screenshot data")

    except Exception as e:
        print(f"❌ Error checking Firebase: {e}")

if __name__ == "__main__":
    check_firebase_logs()