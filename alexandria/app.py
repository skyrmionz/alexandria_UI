from flask import Flask, request, render_template, jsonify
from chat_agent import get_agent_response
from file_processor import process_uploaded_file
from qdrant_client import QdrantClient
import os
from google_drive_ingestor import ingest_drive_files

app = Flask(__name__)

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "google_drive_docs"

# ---------------------------
# Page Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/files")
def files_page():
    return render_template("files.html")

# ---------------------------
# API Endpoints
# ---------------------------

# Chat API endpoint
@app.route("/chat_api", methods=["POST"])
def chat_api():
    user_message = request.form.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response = get_agent_response(user_message)
    return jsonify({"response": response})

# File upload endpoint (supports multiple files)
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No file selected"}), 400
    for file in files:
        if file.filename == "":
            continue  # Skip empty file fields
        process_uploaded_file(file)
    return jsonify({"message": "Files processed and added to the knowledge base"})

# Google Drive ingestion endpoint.
# If a folder_id is provided, only that folder is processed; otherwise, the entire drive.
@app.route("/ingest_drive", methods=["POST"])
def ingest_drive():
    folder_id = request.form.get("folder_id")
    try:
        ingest_drive_files(folder_id)  # folder_id is optional
        return jsonify({"message": "Google Drive ingestion started."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to list uploaded files from Qdrant.
@app.route("/list_files", methods=["GET"])
def list_files():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        collections = client.get_collections().collections
    except Exception as e:
        return jsonify({"files": [], "message": f"Error accessing Qdrant: {e}"})
    
    if COLLECTION_NAME not in [col.name for col in collections]:
        return jsonify({
            "files": [],
            "message": "The collection 'google_drive_docs' does not exist yet. Please upload files first."
        })
    try:
        scroll_result = client.scroll(collection_name=COLLECTION_NAME, limit=100)
        file_names = [point.payload.get("source", "Unknown") for point in scroll_result.points]
        return jsonify({"files": file_names})
    except Exception as e:
        return jsonify({"files": [], "message": f"Error retrieving files: {e}"})

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
