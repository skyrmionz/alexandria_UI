import os
import json
import datetime
import traceback
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_cors import CORS
import time
import re
import hashlib
import humanize
from dotenv import load_dotenv
import uuid

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from chat_agent import get_agent_response, update_agent, DEFAULT_WISE_PROMPT, DEFAULT_SCRIBE_PROMPT, compiled_workflow, chat_with_agent, revise_document, simple_chat_with_agent
from file_processor import process_uploaded_file
from qdrant_client import QdrantClient
from google_auth_oauthlib.flow import Flow
from google_drive_ingestor import ingest_drive_files
from langchain.schema import HumanMessage, AIMessage
from uuid import uuid4

# For knowledge search
from tools.knowledge_base_tool import search_knowledge_base, search_documents

# For logging
import logging
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
# Configure session to be permanent and last for 30 days
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
CORS(app)

load_dotenv()

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "google_drive_docs"

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Running in limited mode without knowledge base functionality.")
    qdrant_client = None

# Define Google Drive OAuth scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def load_client_config():
    with open('credentials.json', 'r') as f:
        return json.load(f)

# Global conversation state variable for chat messages.
conversation_state = {"messages": []}

# Global artifacts storage (in-memory for now, could be moved to a database)
artifacts = {}
next_artifact_id = 1

# Load artifacts from JSON file if it exists
def load_artifacts_from_json():
    global artifacts, next_artifact_id
    try:
        artifacts_file = os.path.join(app.static_folder, 'artifacts', 'artifacts.json')
        if os.path.exists(artifacts_file):
            with open(artifacts_file, 'r') as f:
                artifact_list = json.load(f)
                for artifact in artifact_list:
                    artifact_id = artifact.get('id')
                    if artifact_id:
                        artifacts[artifact_id] = artifact
                        # Update next_artifact_id to be greater than any existing ID
                        try:
                            id_int = int(artifact_id)
                            next_artifact_id = max(next_artifact_id, id_int + 1)
                        except ValueError:
                            pass
            print(f"Loaded {len(artifacts)} artifacts from JSON file")
    except Exception as e:
        print(f"Error loading artifacts from JSON: {e}")

# Load artifacts when the application starts
load_artifacts_from_json()

# Configuration
API_MODELS = ["alexandria/research", "alexandria/scribe"]
FILES_DIR = "static/files"
KNOWLEDGE_DIR = "static/knowledge"
ARTIFACTS_DIR = "static/artifacts"

# Ensure directories exist
for directory in [FILES_DIR, KNOWLEDGE_DIR, ARTIFACTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------------------
# Page Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    # Clear the conversation state when starting a new chat
    session["conversation_state"] = {"messages": []}
    
    # Set default agent type if not already set
    if "agent_type" not in session:
        session["agent_type"] = "wise"
        print(f"CHAT PAGE: Initialized session agent_type to 'wise'")
    else:
        print(f"CHAT PAGE: Existing agent_type in session: {session['agent_type']}")
    
    return render_template("chat.html", 
                         default_wise_prompt=DEFAULT_WISE_PROMPT, 
                         default_scribe_prompt=DEFAULT_SCRIBE_PROMPT)

@app.route("/files")
def files_page():
    valid_creds = False
    user_email = None
    
    if "credentials" in session:
        creds_data = session.get("credentials")
        if creds_data and isinstance(creds_data, dict):
            if creds_data.get("refresh_token") and creds_data.get("token"):
                try:
                    # Additional validation could be done here if needed
                    valid_creds = True
                    user_email = session.get("user_email", "Unknown")
                except Exception as e:
                    print(f"Error validating credentials: {str(e)}")
                    # Clear invalid credentials
                    session.pop("credentials", None)
                    session.pop("user_email", None)
    
    return render_template("files.html", valid_creds=valid_creds, user_email=user_email)

@app.route("/artifacts")
def artifacts_page():
    return render_template("artifacts.html")

@app.route("/article")
def article():
    article_id = request.args.get("id")
    if article_id in artifacts:
        return render_template("article.html", article_data=artifacts[article_id])
    return render_template("article.html", article_data={
        "title": "Article Not Found",
        "author": "Unknown",
        "content": "<p>The requested article could not be found.</p>"
    })

# ---------------------------
# OAuth Endpoints for Google Drive
# ---------------------------

@app.route("/start_auth")
def start_auth():
    session.pop('credentials', None)
    flow = Flow.from_client_config(
        load_client_config(),
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
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
    """
    Process a chat message and return a response.
    
    JSON payload:
    {
        "message": "User's message",
        "reset": true/false (optional),
        "agent_type": "agent type" (optional)
    }
    """
    data = request.get_json()
    message = data.get("message", "").strip()
    reset = data.get("reset", False)
    
    # Get agent_type from request if provided, otherwise from session
    agent_type_from_request = data.get("agent_type")
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Initialize session state if it doesn't exist
    if "conversation_state" not in session:
        session["conversation_state"] = {
            "messages": []
        }
    
    # Reset conversation if requested
    if reset:
        session["conversation_state"] = {
            "messages": []
        }
    
    # Get the current conversation state
    conversation_state = session["conversation_state"]
    
    # Add the user's message to the conversation state
    conversation_state["messages"].append({
        "type": "human",
        "content": message
    })
    
    # Prepare chat history for the agent
    chat_history = ""
    for msg in conversation_state["messages"]:
        prefix = "Human: " if msg["type"] == "human" else "AI: "
        chat_history += prefix + msg["content"] + "\n"
    
    # Use agent_type from request if provided, otherwise from session
    if agent_type_from_request:
        agent_type = agent_type_from_request
        # Update session with the new agent_type
        session["agent_type"] = agent_type
        print(f"CHAT API: Using agent_type from request: {agent_type}")
    else:
        # Get the current agent type from the session
        # Default to "wise" if not found
        agent_type = session.get("agent_type", "wise")
        print(f"CHAT API: Using agent_type from session: {agent_type}")
    
    print(f"CHAT API: Final agent_type being used: {agent_type}")
    
    try:
        print(f"CHAT API: Using STANDARD agent: {agent_type}")
        # Use the regular chat agent
        response_data = chat_with_agent(message, chat_history, agent_type)
        print(f"CHAT API: Got standard response, length: {len(str(response_data))}")
        
        # Extract the response text from the dictionary
        response_text = response_data.get("response", "I'm sorry, I couldn't generate a response.")
        
        # Add the response to the conversation state
        conversation_state["messages"].append({
            "type": "ai",
            "content": response_text
        })
        
        # Save the updated conversation state
        session["conversation_state"] = conversation_state
        
        # Ensure the session is saved
        session.modified = True
        
        # Return the response
        return jsonify({
            "response": response_text
        })
    except Exception as e:
        print(f"Chat processing error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to process chat: {str(e)}"}), 500

@app.route("/update_agent", methods=["POST"])
def update_agent_route():
    data = request.get_json()
    agent_type = data.get("agent_type")
    prompt = data.get("prompt")
    
    print(f"UPDATING AGENT: Received request to change agent type to: {agent_type}")
    
    # Store the agent type in the session
    session["agent_type"] = agent_type
    print(f"UPDATING AGENT: Set session['agent_type'] to: {agent_type}")
    
    # Force-write the session to ensure it's saved
    session.modified = True
    
    try:
        print(f"UPDATING AGENT: Calling update_agent() for {agent_type}")
        update_agent(agent_type, prompt)
        
        # Clear the conversation state when changing agent type
        session["conversation_state"] = {"messages": []}
        return jsonify({"message": f"Agent updated successfully to {agent_type}."})
    except Exception as e:
        print(f"Error updating agent: {str(e)}")
        return jsonify({"message": f"Error updating agent: {str(e)}"}), 500

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

@app.route("/ingest_drive", methods=["POST"])
def ingest_drive():
    folder_id = request.form.get("folder_id")
    try:
        creds_data = session.get("credentials")
        ingest_drive_files(folder_id=folder_id, creds_data=creds_data)
        return jsonify({"message": "Google Drive ingestion completed successfully."})
    except Exception as e:
        return jsonify({"error": str(e), "suggestion": "Ensure folder ID is correct and accessible"}), 500

@app.route("/create_artifact", methods=["POST"])
def create_artifact():
    global next_artifact_id, artifacts
    try:
        print("Received create_artifact request")
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400
            
        print("Received artifact data:", data)
        title = data.get("title")
        author = data.get("author")
        content = data.get("content")
        
        if not all([title, author, content]):
            print("Missing required fields:", {
                "title": bool(title),
                "author": bool(author),
                "content": bool(content)
            })
            return jsonify({
                "error": "Missing required fields",
                "details": {
                    "title": bool(title),
                    "author": bool(author),
                    "content": bool(content)
                }
            }), 400
            
        # Clean up the content to ensure proper HTML formatting
        content = content.strip()
        if not content.startswith("<h1>"):
            content = f"<h1>{title}</h1>\n{content}"
            
        artifact_id = str(next_artifact_id)
        next_artifact_id += 1
        
        artifacts[artifact_id] = {
            "id": artifact_id,
            "title": title,
            "author": author,
            "content": content,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        print(f"Created new artifact: {title} (ID: {artifact_id})")
        print(f"Current artifacts in memory: {len(artifacts)}")
        
        return jsonify({
            "message": "Artifact created successfully",
            "artifact_id": artifact_id,
            "artifact": artifacts[artifact_id]
        })
        
    except Exception as e:
        print(f"Error creating artifact: {str(e)}")
        return jsonify({
            "error": "Failed to create artifact",
            "details": str(e)
        }), 500

@app.route("/get_artifacts")
def get_artifacts():
    print(f"Returning {len(artifacts)} artifacts")
    return jsonify({"artifacts": list(artifacts.values())})

@app.route("/get_current_agent")
def get_current_agent():
    """Return the current agent type from the session."""
    agent_type = session.get("agent_type", "wise")
    print(f"GET CURRENT AGENT: Returning {agent_type} from session")
    return jsonify({"agent_type": agent_type})

# Update the revise_document route to work with the new multi-agent architecture
@app.route('/api/revise-document', methods=['POST'])
def revise_document_route():
    """API endpoint to revise a document using the multi-agent architecture"""
    try:
        data = request.get_json()
        document_query = data.get('document_query', '')
        query = data.get('query', '')
        message = data.get('message', '')
        
        # Create a message with the delegation format
        delegated_message = f"""
I need to revise a document. Here are the details:

<delegate_document_revision>
document_query: "{document_query}"
revision_query: "{query}"
</delegate_document_revision>

{message}
        """
        
        # Use the chat_with_agent function with the enhanced message
        response = chat_with_agent(delegated_message, agent_type="wise")
        
        if response and isinstance(response, dict):
            return jsonify(response)
        else:
            return jsonify({"error": "Invalid response from agent"}), 500
    except Exception as e:
        logger.exception(f"Error in revise_document API: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run the App
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)