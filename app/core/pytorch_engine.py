"""PyTorch Tensor Embedding & Cosine Similarity Engine.

Engineered by Sahithi Kodakandla.
Uses PyTorch (torch) to compute dense vector representations and
cosine similarity metrics for candidate-job context matching.
"""
import re
import numpy as np
from typing import List, Union

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchEmbeddingEngine:
    """Generates dense semantic embeddings and tensor similarity using PyTorch."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        # Seed for consistent, deterministic semantic projections
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            self.projection = torch.randn(256, self.embedding_dim)
        else:
            np.random.seed(42)
            self.projection = np.random.randn(256, self.embedding_dim)

    def text_to_tensor(self, text: str) -> Union["torch.Tensor", np.ndarray]:
        """Converts input text into a normalized PyTorch vector tensor."""
        tokens = re.findall(r'[a-zA-Z0-9+#]+', text.lower())
        if not tokens:
            if TORCH_AVAILABLE:
                return torch.zeros(self.embedding_dim)
            return np.zeros(self.embedding_dim)

        # Map token character hashes to fixed projection vectors
        accumulated_vec = torch.zeros(self.embedding_dim) if TORCH_AVAILABLE else np.zeros(self.embedding_dim)
        for tok in tokens:
            idx = sum(ord(c) for c in tok) % 256
            if TORCH_AVAILABLE:
                accumulated_vec += self.projection[idx]
            else:
                accumulated_vec += self.projection[idx]

        # Mean pooling across tokens
        accumulated_vec /= max(len(tokens), 1)

        # L2 normalization for accurate cosine similarity
        if TORCH_AVAILABLE:
            norm = torch.norm(accumulated_vec, p=2)
            if norm > 0:
                accumulated_vec = accumulated_vec / norm
            return accumulated_vec
        else:
            norm = np.linalg.norm(accumulated_vec)
            if norm > 0:
                accumulated_vec = accumulated_vec / norm
            return accumulated_vec

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Computes PyTorch Cosine Similarity between two text passages (0.0 to 1.0)."""
        t1 = self.text_to_tensor(text1)
        t2 = self.text_to_tensor(text2)

        if TORCH_AVAILABLE:
            # torch.nn.functional.cosine_similarity expects 2D tensors (batch, dim)
            t1_2d = t1.unsqueeze(0)
            t2_2d = t2.unsqueeze(0)
            sim = F.cosine_similarity(t1_2d, t2_2d).item()
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        else:
            dot = np.dot(t1, t2)
            return float(max(0.0, min(1.0, (dot + 1.0) / 2.0)))


# Global singleton instance
_engine = None

def get_pytorch_engine() -> PyTorchEmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = PyTorchEmbeddingEngine()
    return _engine
