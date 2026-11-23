import os
import sys

# Ensure root directory is accessible for importing modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

# --- WSGI Middleware Fix for Vercel Routing Path mismatch ---
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = environ.get("HTTP_X_MATCHED_PATH", "")
        if matched_path:
            environ["PATH_INFO"] = matched_path.split("?")[0]
        else:
            path_info = environ.get("PATH_INFO", "")
            for prefix in ("/api/index.py", "/api/index"):
                if path_info.startswith(prefix):
                    path_info = path_info[len(prefix):] or "/"
                    environ["PATH_INFO"] = path_info
                    break

        if environ.get("PATH_INFO") in ("/api/index", "/api/index.py", ""):
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)
