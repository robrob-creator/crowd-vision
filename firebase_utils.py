# Firebase utilities

def init_firebase(firebase_method, firebase_credentials, firebase_db_url=None, app_name='default'):
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import credentials

        # Check if app already exists
        if firebase_admin._apps and app_name in firebase_admin._apps:
            # already initialized
            app = firebase_admin.get_app(app_name)
            # Return appropriate clients based on method
            if firebase_method == "realtime":
                from firebase_admin import db as rtdb  # type: ignore
                from firebase_admin import firestore as fs  # type: ignore
                return True, app, rtdb, fs.client(app)
            else:
                from firebase_admin import firestore as fs  # type: ignore
                return True, app, None, fs.client(app)

        # Initialize new app
        cred = None
        if firebase_credentials:
            cred = credentials.Certificate(firebase_credentials)

        if firebase_method == "realtime":
            # For Realtime DB, a databaseURL must be provided
            import os
            db_url = firebase_db_url or os.getenv("FIREBASE_DB_URL")
            if not db_url:
                print("Error: FIREBASE_DB_URL env var is required for Realtime Database.")
                return False, None, None, None
            firebase_admin.initialize_app(cred, {"databaseURL": db_url}, name=app_name)
            app = firebase_admin.get_app(app_name)
            from firebase_admin import db as rtdb  # type: ignore
            from firebase_admin import firestore as fs  # type: ignore
            return True, app, rtdb, fs.client(app)
        else:
            firebase_admin.initialize_app(cred, name=app_name)
            app = firebase_admin.get_app(app_name)
            from firebase_admin import firestore as fs  # type: ignore
            return True, app, None, fs.client(app)

    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        print("Install SDK: pip install firebase-admin and provide credentials.")
        return False, None, None, None
