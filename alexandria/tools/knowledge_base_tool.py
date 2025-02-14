# tools/knowledge_base_tool.py

import os
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "google_drive_docs"

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

def create_collection():
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [col.name for col in collections]:
        from qdrant_client.http.models import Distance, VectorParams
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

create_collection()

# Instantiate the Hugging Face embedding model using the small variant
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def knowledge_base_search_tool(query: str) -> str:
    """
    Search the Qdrant vector store for relevant content.
    If the collection doesn't exist, return a message stating that files have not been uploaded yet.
    """
    # First, check if the collection exists
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [col.name for col in collections]:
        return "The knowledge base is empty. Please upload files to populate it."
    
    query_vector = embeddings.embed_query(query)
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3  # Return the top 3 matches
        )
    except Exception as e:
        return f"Error searching the knowledge base: {str(e)}"
    
    aggregated = ""
    for result in results:
        aggregated += result.payload.get("text", "") + "\n"
    return aggregated or "No relevant information found in the knowledge base."
