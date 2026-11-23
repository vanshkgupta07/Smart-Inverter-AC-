import os
import sys
from urllib.parse import parse_qs

# Ensure root directory is accessible for importing modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

# --- WSGI Middleware for Vercel Serverless Routing ---
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query_string = environ.get("QUERY_STRING", "")
        if "_path=" in query_string:
            params = parse_qs(query_string)
            if "_path" in params and params["_path"]:
                target_path = params["_path"][0]
                environ["PATH_INFO"] = target_path if target_path.startswith("/") else "/" + target_path
        elif environ.get("HTTP_X_FORWARDED_URI"):
            environ["PATH_INFO"] = environ["HTTP_X_FORWARDED_URI"].split("?")[0]
        elif environ.get("PATH_INFO") in ("/api/index.py", "/api/index", "/api", ""):
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)
