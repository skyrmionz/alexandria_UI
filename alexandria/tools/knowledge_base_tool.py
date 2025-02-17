import os
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Qdrant configuration
# (We won't rely on COLLECTION_NAME here; we'll search both)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTIONS_TO_SEARCH = ["google_drive_docs", "knowledge_base"]

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

def create_collection(collection_name):
    collections = client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        from qdrant_client.http.models import Distance, VectorParams
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

# Create collections as needed (example for one collection)
create_collection("google_drive_docs")
# create_collection("knowledge_base")  # Create if needed

# Instantiate the Hugging Face embedding model using the large variant
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

def knowledge_base_search_tool(query: str) -> str:
    """
    Search the Qdrant vector store for relevant content from two collections:
    "google_drive_docs" and "knowledge_base".
    If no collection exists or no results are found, return an appropriate message.
    """
    # Determine which collections exist
    available_collections = [col.name for col in client.get_collections().collections]
    collections_to_search = [name for name in COLLECTIONS_TO_SEARCH if name in available_collections]
    
    if not collections_to_search:
        return "The knowledge base is empty. Please upload files to populate it."
    
    query_vector = embeddings.embed_query(query)
    aggregated = ""
    
    # Search in each available collection
    for coll in collections_to_search:
        try:
            results = client.search(
                collection_name=coll,
                query_vector=query_vector,
                limit=10  # Return the top 10 matches per collection
            )
        except Exception as e:
            aggregated += f"Error searching collection {coll}: {str(e)}\n"
            continue
        
        # Process results if any
        for result in results:
            title = result.payload.get("source", "")
            content = result.payload.get("text", "")
            aggregated += f"[{coll}] {title}: {content}\n"
    
    return aggregated or "No relevant information found in the knowledge base."