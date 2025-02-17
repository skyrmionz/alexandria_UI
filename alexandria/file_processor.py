import os
import tempfile
import uuid  # Used to generate unique IDs for each chunk
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "knowledge_base"

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection():
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

create_collection()

# Instantiate the Hugging Face embedding model using the large variant
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Set up a text splitter to divide long text into chunks (e.g., 500 characters with 50 characters overlap)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def process_uploaded_file(file_storage):
    """
    Process an uploaded file by:
      1. Extracting text from supported file types (PDF, image, CSV, Excel)
      2. Splitting the text into smaller chunks
      3. Embedding each chunk using the HuggingFace model
      4. Upserting each chunk into Qdrant with a unique ID.
    """
    filename = file_storage.filename
    ext = filename.split('.')[-1].lower()
    text = ""
    
    # Save the uploaded file temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        file_storage.save(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        if ext == "pdf":
            # Extract text from a PDF file
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
        elif ext in ["png", "jpg", "jpeg", "bmp"]:
            # Extract text from an image using OCR
            image = Image.open(tmp_path)
            text = pytesseract.image_to_string(image)
        elif ext == "csv":
            try:
                df = pd.read_csv(tmp_path)
                text = df.to_string(index=False)
            except Exception as e:
                text = f"Error reading CSV: {str(e)}"
        elif ext in ["xls", "xlsx"]:
            try:
                df = pd.read_excel(tmp_path)
                text = df.to_string(index=False)
            except Exception as e:
                text = f"Error reading Excel file: {str(e)}"
        else:
            text = "Unsupported file type for text extraction."
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    
    # Proceed only if valid text was extracted
    if text.strip() and "Unsupported" not in text:
        # Split the extracted text into manageable chunks.
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            # Generate a unique point ID based on the filename and chunk index.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}-{i}"))
            vector = embeddings.embed_query(chunk)
            # Upsert this chunk as a separate point.
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {"source": filename, "text": chunk}
                }]
            )
    else:
        print("No valid text extracted from file:", filename)
    
    os.remove(tmp_path)