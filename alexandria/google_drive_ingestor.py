# google_drive_ingestor.py

import os
import uuid
import pickle
import io

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------
# 1. Google Drive Authentication
# ------------------------

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_PICKLE = 'token.pickle'
CLIENT_SECRET_FILE = 'credentials.json'

def authenticate_drive():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            # Use fixed port 8080 to match your authorized redirect URI
            creds = flow.run_local_server(port=8080)
        with open(TOKEN_PICKLE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

# ------------------------
# 2. Google Drive File Listing and Downloading
# ------------------------

def list_files(service, query=""):
    try:
        results = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def download_file(service, file_id, mime_type, destination):
    try:
        if mime_type.startswith("application/vnd.google-apps"):
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        else:
            request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {destination}: {int(status.progress() * 100)}%")
        fh.close()
        return destination
    except HttpError as error:
        print(f"An error occurred while downloading file {file_id}: {error}")
        return None

# ------------------------
# 3. Ingest Files into Qdrant
# ------------------------

def ingest_drive_files(folder_id=None):
    # Authenticate and build the Drive service.
    creds = authenticate_drive()
    service = build('drive', 'v3', credentials=creds)
    
    # If folder_id is provided, limit to files in that folder.
    if folder_id:
        query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder'"
    else:
        query = "mimeType!='application/vnd.google-apps.folder'"
    
    files = list_files(service, query=query)
    
    qdrant_client = QdrantClient(host="localhost", port=6333)
    collection_name = "google_drive_docs"
    existing = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in existing]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']
        print(f"Processing file: {file_name} (MIME: {mime_type})")
        
        destination = f"/tmp/{file_id}_{file_name.replace(' ', '_')}.txt"
        downloaded = download_file(service, file_id, mime_type, destination)
        if not downloaded:
            continue
        
        try:
            with open(downloaded, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            os.remove(destination)
            continue
        
        if not content.strip():
            print(f"Empty content for file {file_name}")
            os.remove(destination)
            continue
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_id))
        vector = embeddings.embed_query(content)
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": vector,
                "payload": {"source": file_name, "text": content}
            }]
        )
        print(f"Ingested {file_name} into Qdrant.")
        os.remove(destination)

if __name__ == "__main__":
    # For testing, you can pass a folder ID here if desired.
    ingest_drive_files()
