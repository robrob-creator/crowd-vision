#!/usr/bin/env python3
"""
Quick script to check Firebase Realtime Database data
"""
import firebase_admin
from firebase_admin import credentials, db
import os

def check_realtime_db():
    """Check Firebase Realtime Database data"""
    try:
        # Initialize Firebase
        cred_path = "config/firebase_credentials.json"
        if not os.path.exists(cred_path):
            cred_path = "firebase_credentials.json"

        if not os.path.exists(cred_path):
            print("❌ Firebase credentials not found")
            return

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://inventi-fc7cc-default-rtdb.firebaseio.com'
        })

        # Get Realtime Database reference
        ref = db.reference('locations')

        # Get all data under locations
        data = ref.get()
        if data:
            print(f"✅ Found data in Realtime Database:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No data found in Realtime Database under 'locations'")

        # Check specific location
        location_ref = db.reference('locations/Davao_Barber_Shop')
        location_data = location_ref.get()
        if location_data:
            print(f"✅ Found data for Davao_Barber_Shop:")
            print(f"  {location_data}")
        else:
            print("❌ No data found for Davao_Barber_Shop")

    except Exception as e:
        print(f"❌ Error checking Realtime Database: {e}")

if __name__ == "__main__":
    check_realtime_db()