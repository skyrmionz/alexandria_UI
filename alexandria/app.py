import os
import json
import datetime
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from chat_agent import get_agent_response, update_agent, DEFAULT_WISE_PROMPT, DEFAULT_SCRIBE_PROMPT, compiled_workflow
from file_processor import process_uploaded_file
from qdrant_client import QdrantClient
from google_auth_oauthlib.flow import Flow
from google_drive_ingestor import ingest_drive_files
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

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

# Global conversation state variable for chat messages.
conversation_state = {"messages": []}

# Global artifacts storage (in-memory for now, could be moved to a database)
artifacts = {}
next_artifact_id = 1

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
    return render_template("chat.html", 
                         default_wise_prompt=DEFAULT_WISE_PROMPT, 
                         default_scribe_prompt=DEFAULT_SCRIBE_PROMPT)

@app.route("/files")
def files_page():
    valid_creds = False
    user_email = None
    if "credentials" in session:
        creds_data = session.get("credentials")
        if creds_data.get("refresh_token") and creds_data.get("token"):
            valid_creds = True
            user_email = session.get("user_email", "Unknown")
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
    if "conversation_state" not in session:
        session["conversation_state"] = {"messages": []}
    
    try:
        message = request.form.get("message", "")
        if not message.strip():
            return jsonify({"error": "Empty message"}), 400
            
        conversation_state = session["conversation_state"]
        
        if "messages" not in conversation_state:
            conversation_state["messages"] = []
        
        # Store messages in a serializable format
        conversation_state["messages"].append({
            "type": "human",
            "content": message
        })
        
        if qdrant_client is None:
            return jsonify({"error": "Knowledge base is not available"}), 503
            
        # Convert serialized messages back to LangChain format for the workflow
        langchain_messages = [
            HumanMessage(content=msg["content"]) if msg["type"] == "human"
            else AIMessage(content=msg["content"])
            for msg in conversation_state["messages"]
        ]
        
        result = compiled_workflow.invoke(
            {"messages": langchain_messages},
            config={
                "thread_id": str(uuid4()),
                "checkpoint_id": str(uuid4()),
                "checkpoint_ns": "alexandria_chat"
            }
        )
        
        if not result or "messages" not in result:
            return jsonify({"error": "Invalid response from language model"}), 500
            
        # Store the result back in serializable format
        conversation_state["messages"] = [
            {
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content
            }
            for msg in result["messages"]
        ]
        session["conversation_state"] = conversation_state
        
        # Get the last AI message for the response
        assistant_message = result["messages"][-1].content if result["messages"] else ""
        return jsonify({"response": assistant_message})
        
    except Exception as e:
        print(f"Chat processing error: {str(e)}")  # Log the error
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

@app.route("/update_agent", methods=["POST"])
def update_agent_route():
    agent_type = request.form.get("agent_type")
    prompt = request.form.get("prompt")
    try:
        update_agent(agent_type, prompt)
        # Clear the conversation state when changing agent type
        session["conversation_state"] = {"messages": []}
        return jsonify({"message": "Agent updated successfully."})
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

# ---------------------------
# Run the App
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)