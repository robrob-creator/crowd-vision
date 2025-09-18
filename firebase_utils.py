# Firebase utilities

firebase_admin = None
firestore = None
db_rtdb = None

def init_firebase(firebase_method, firebase_credentials, firebase_db_url=None, app_name='default'):
    global firebase_admin, firestore, db_rtdb
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import credentials
        if firebase_admin._apps and app_name in firebase_admin._apps:
            # already initialized
            app = firebase_admin.get_app(app_name)
        else:
            cred = None
            if firebase_credentials:
                cred = credentials.Certificate(firebase_credentials)
            # Initialize app
            if firebase_method == "realtime":
                # For Realtime DB, a databaseURL must be provided. Expect env FIREBASE_DB_URL.
                import os
                db_url = firebase_db_url or os.getenv("FIREBASE_DB_URL")
                if not db_url:
                    print("Error: FIREBASE_DB_URL env var is required for Realtime Database.")
                    return False, None, None, None
                firebase_admin.initialize_app(cred, {"databaseURL": db_url}, name=app_name)
                app = firebase_admin.get_app(app_name)
                from firebase_admin import db as rtdb  # type: ignore
                db_rtdb = rtdb
                # Also initialize Firestore for logs
                from firebase_admin import firestore as fs  # type: ignore
                firestore = fs
                return True, app, db_rtdb, firestore
            else:
                firebase_admin.initialize_app(cred, name=app_name)
                app = firebase_admin.get_app(app_name)
                from firebase_admin import firestore as fs  # type: ignore
                firestore = fs
                return True, app, None, firestore
        return True, app, db_rtdb, firestore
    except Exception as e:
        print("Firebase initialization failed:", e)
        print("Install SDK: pip install firebase-admin and provide credentials.")
        return False, None, None, None
