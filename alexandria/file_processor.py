# file_processor.py

import os
import tempfile
import uuid  # Import uuid to generate valid point IDs
from langchain_community.embeddings import HuggingFaceEmbeddings  # Using community package
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

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
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Updated dimension to 384
        )

create_collection()

# Instantiate the Hugging Face embedding model using the small variant
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def process_uploaded_file(file_storage):
    """
    Process an uploaded file: extract text (supports PDFs and images via OCR),
    vectorize the text using the Hugging Face model, and insert the data into Qdrant.
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
                    text += page_text + "\n"
        elif ext in ["png", "jpg", "jpeg", "bmp"]:
            # Extract text from an image using OCR
            image = Image.open(tmp_path)
            text = pytesseract.image_to_string(image)
        else:
            text = "Unsupported file type for text extraction."
    except Exception as e:
        text = f"Error extracting text: {str(e)}"

    # Proceed only if valid text was extracted
    if text.strip() and "Unsupported" not in text:
        # Generate a deterministic UUID based on the filename
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
        vector = embeddings.embed_query(text)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": point_id,
                "vector": vector,
                "payload": {"source": filename, "text": text}
            }]
        )
    else:
        print("No valid text extracted from file:", filename)
    
    os.remove(tmp_path)
