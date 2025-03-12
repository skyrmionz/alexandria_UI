import os
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTIONS_TO_SEARCH = ["google_drive_docs", "knowledge_base", "artifacts"]

# Connect to Qdrant
try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    qdrant_available = True
except Exception as e:
    print(f"Error connecting to Qdrant in knowledge_base_tool.py: {e}")
    print("Knowledge base search will not be available.")
    client = None
    qdrant_available = False

def create_collection(collection_name):
    """Create a collection in Qdrant if it doesn't exist."""
    if not qdrant_available:
        print(f"Skipping collection creation for {collection_name} as Qdrant is disabled or unavailable")
        return
        
    try:
        collections = client.get_collections().collections
        if collection_name not in [col.name for col in collections]:
            from qdrant_client.http.models import Distance, VectorParams
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    except Exception as e:
        print(f"Error creating collection {collection_name}: {e}")

# Create collections as needed (example for one collection)
if qdrant_available:
    create_collection("google_drive_docs")
    # create_collection("knowledge_base")  # Create if needed
    create_collection("artifacts")  # Create artifacts collection

# Instantiate the Hugging Face embedding model using the large variant
try:
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
except Exception as e:
    print(f"Error loading embeddings model: {e}")
    embeddings = None

def knowledge_base_search_tool(query: str, search_type: str = "all") -> str:
    """
    Search the Qdrant vector store for relevant content.
    
    Args:
        query: The search query
        search_type: Type of search to perform - "all", "documents", or "artifacts"
        
    Returns:
        A string containing the search results
    """
    if not qdrant_available or not embeddings:
        return "Knowledge base search is not available. The system is running in limited mode."
        
    # Determine which collections exist
    try:
        available_collections = [col.name for col in client.get_collections().collections]
    except Exception as e:
        return f"Error accessing knowledge base: {str(e)}"
    
    # Filter collections based on search_type
    if search_type.lower() == "documents":
        collections_to_search = [name for name in ["google_drive_docs", "knowledge_base"] if name in available_collections]
    elif search_type.lower() == "artifacts":
        collections_to_search = [name for name in ["artifacts"] if name in available_collections]
    else:  # "all" or any other value
        collections_to_search = [name for name in COLLECTIONS_TO_SEARCH if name in available_collections]
    
    if not collections_to_search:
        if search_type.lower() == "artifacts":
            return "No artifacts found in the knowledge base. You can create new artifacts using the Scribe mode."
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
        if results:
            aggregated += f"\n--- Results from {coll} ---\n"
            for result in results:
                title = result.payload.get("source", "")
                content = result.payload.get("text", "")
                doc_type = result.payload.get("type", "document")
                
                # Format differently based on document type
                if doc_type == "artifact":
                    aggregated += f"[ARTIFACT] {title}\n{content[:300]}...\n\n"
                else:
                    aggregated += f"[DOCUMENT] {title}\n{content[:300]}...\n\n"
    
    return aggregated or f"No relevant information found for '{query}' in the knowledge base."

def search_artifacts(query: str) -> str:
    """
    Search specifically for artifacts in the knowledge base.
    
    Args:
        query: The search query
        
    Returns:
        A string containing the artifact search results
    """
    if not qdrant_available or not embeddings:
        return "Artifact search is not available. The system is running in limited mode."
    return knowledge_base_search_tool(query, search_type="artifacts")

def search_documents(query: str) -> str:
    """
    Search specifically for documents in the knowledge base.
    
    Args:
        query: The search query
        
    Returns:
        A string containing the document search results
    """
    if not qdrant_available or not embeddings:
        return "Document search is not available. The system is running in limited mode."
    return knowledge_base_search_tool(query, search_type="documents")