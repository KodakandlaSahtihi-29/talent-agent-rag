"""Document Chunker for RAG Pipelines.

Engineered by Sahithi Kodakandla.
Breaks raw resumes and job descriptions into semantic text chunks
suitable for vector embedding and retrieval.
"""
import re
from typing import List, Dict, Any

class DocumentChunker:
    """Splits documents into coherent paragraph or bullet-point chunks."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 150) -> List[Dict[str, Any]]:
        """Splits input document into structured text chunks with metadata."""
        raw_chunks = [c.strip() for c in re.split(r'\n{2,}|\n(?=[•\-\*\d+\.])', text) if c.strip()]
        
        chunks = []
        chunk_id = 1
        for rc in raw_chunks:
            # Clean extra whitespace
            cleaned = re.sub(r'\s+', ' ', rc).strip()
            if len(cleaned) < 15:
                continue

            chunks.append({
                "chunk_id": chunk_id,
                "text": cleaned,
                "token_count": len(cleaned.split()),
                "preview": cleaned[:60] + ("..." if len(cleaned) > 60 else "")
            })
            chunk_id += 1

        return chunks
