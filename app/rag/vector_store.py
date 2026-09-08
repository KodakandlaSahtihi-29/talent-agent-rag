"""In-Memory Vector Store powered by PyTorch Tensors.

Engineered by Sahithi Kodakandla.
Stores document chunks and their PyTorch embeddings, enabling
fast semantic top-K retrieval for the RAG pipeline.
"""
from typing import List, Dict, Any
from app.core.pytorch_engine import get_pytorch_engine

class PyTorchVectorStore:
    """Vector database storing text chunks and PyTorch tensor representations."""

    def __init__(self):
        self.engine = get_pytorch_engine()
        self.records: List[Dict[str, Any]] = []

    def clear(self):
        """Resets the vector index."""
        self.records.clear()

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Embeds text chunks into PyTorch tensors and adds them to the index."""
        for c in chunks:
            tensor = self.engine.text_to_tensor(c["text"])
            self.records.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "preview": c.get("preview", c["text"][:60]),
                "tensor": tensor
            })

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top-K most semantically relevant chunks using PyTorch similarity."""
        if not self.records:
            return []

        results = []
        for rec in self.records:
            sim = self.engine.compute_similarity(query, rec["text"])
            results.append({
                "chunk_id": rec["chunk_id"],
                "text": rec["text"],
                "preview": rec["preview"],
                "similarity": round(sim * 100, 1)
            })

        # Sort descending by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
