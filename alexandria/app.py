import os
import json
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from chat_agent import get_agent_response, DEFAULT_WISE_PROMPT, DEFAULT_SCRIBE_PROMPT, update_agent
from file_processor import process_uploaded_file
from qdrant_client import QdrantClient
from google_auth_oauthlib.flow import Flow
from google_drive_ingestor import ingest_drive_files  # Modified to accept creds_data parameter
from dotenv import load_dotenv 

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")  # Needed for session management

load_dotenv()
pod_id = os.getenv("POD_ID")

# Qdrant configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"https://{pod_id}-11434.proxy.runpod.net/")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "google_drive_docs"

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    qdrant_client = None

# Define Google Drive OAuth scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def load_client_config():
    with open('credentials.json', 'r') as f:
        return json.load(f)

# ---------------------------
# Page Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html", 
                           default_wise_prompt=DEFAULT_WISE_PROMPT, 
                           default_scribe_prompt=DEFAULT_SCRIBE_PROMPT)

@app.route("/files")
def files_page():
    valid_creds = False
    user_email = None
    if "credentials" in session:
        creds_data = session.get("credentials")
        # Check that a refresh token exists and that token is present
        if creds_data.get("refresh_token") and creds_data.get("token"):
            valid_creds = True
            user_email = session.get("user_email", "Unknown")
    return render_template("files.html", valid_creds=valid_creds, user_email=user_email)

# ---------------------------
# OAuth Endpoints for Google Drive
# ---------------------------

@app.route("/start_auth")
def start_auth():
    # Clear any existing credentials so we always start fresh
    session.pop('credentials', None)
    flow = Flow.from_client_config(
        load_client_config(),
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'  # Force re-consent to ensure a refresh token is returned
    )
    session['state'] = state
    return redirect(authorization_url)

@app.route("/oauth2callback")
def oauth2callback():
    state = session.get('state')
    if not state:
        return "Missing state in session.", 400

    flow = Flow.from_client_config(
        load_client_config(),
        scopes=SCOPES,
        state=state,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials

    # Check if refresh token is present; if not, force reauthentication.
    if credentials.refresh_token is None:
        session.pop('credentials', None)
        return "Authentication failed: no refresh token obtained. Please try logging in again.", 400

    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    # Build the Drive service to test authentication and get user info.
    from googleapiclient.discovery import build
    drive_service = build('drive', 'v3', credentials=credentials)
    about_info = drive_service.about().get(fields="user(emailAddress)").execute()
    user_email = about_info.get("user", {}).get("emailAddress", "Unknown")
    session['user_email'] = user_email
    print(f"Authenticated as: {user_email}")
    return redirect(url_for("files_page"))

# ---------------------------
# API Endpoints
# ---------------------------

@app.route("/chat_api", methods=["POST"])
def chat_api():
    user_message = request.form.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = get_agent_response(user_message)
    except Exception as e:
        return jsonify({"error": f"Failed to get response: {e}"}), 500

    return jsonify({"response": response})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No file selected"}), 400
    for file in files:
        if file.filename.strip() == "":
            return jsonify({"error": "One of the selected files has no filename"}), 400
        try:
            process_uploaded_file(file)
        except Exception as e:
            return jsonify({"error": f"Failed to process file {file.filename}: {e}"}), 500
    return jsonify({"message": "Files processed and added to the knowledge base"})

# Google Drive ingestion endpoint
@app.route("/ingest_drive", methods=["POST"])
def ingest_drive():
    folder_id = request.form.get("folder_id")
    try:
        # Use credentials from session if available.
        creds_data = session.get("credentials")
        ingest_drive_files(folder_id=folder_id, creds_data=creds_data)
        return jsonify({"message": "Google Drive ingestion started."})
    except Exception as e:
        return jsonify({"error": str(e), "suggestion": "Ensure folder ID is correct and accessible"}), 500

@app.route("/update_agent", methods=["POST"])
def update_agent_route():
    agent_type = request.form.get("agent_type")
    prompt = request.form.get("prompt")
    try:
        update_agent(agent_type, prompt)
        return jsonify({"message": "Agent updated successfully."})
    except Exception as e:
        return jsonify({"message": f"Error updating agent: {str(e)}"}), 500

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)