from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from backend.data_loader import load_all_chunks

# Embedding model — lightweight and fast
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "support_docs"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension

# Module-level singletons
_client = QdrantClient(":memory:")
_model = SentenceTransformer(EMBEDDING_MODEL)
_chunks = []


def initialize_collection():
    """Load data chunks, embed them, and upsert into Qdrant in-memory collection."""
    global _chunks

    _chunks = load_all_chunks()

    # Create collection (delete first if it already exists)
    if _client.collection_exists(COLLECTION_NAME):
        _client.delete_collection(COLLECTION_NAME)
    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Embed all chunks (extract text first)
    texts = [chunk["text"] for chunk in _chunks]
    embeddings = _model.encode(texts, show_progress_bar=True)

    # Build points and upsert
    points = [
        PointStruct(id=i, vector=embedding.tolist(), payload=chunk)
        for i, (embedding, chunk) in enumerate(zip(embeddings, _chunks))
    ]

    _client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"[Qdrant] Initialized collection with {len(points)} documents.")


def search(query: str, top_k: int = 3, user_id: str = None) -> list[str]:
    """Embed a query and return the top-k most similar text chunks."""
    query_embedding = _model.encode(query).tolist()
    
    query_filter = None
    if user_id:
        query_filter = Filter(
            should=[
                FieldCondition(key="type", match=MatchValue(value="policy")),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
        )

    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
    )

    return [hit.payload["text"] for hit in results.points]
