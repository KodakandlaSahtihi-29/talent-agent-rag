"""Agent 1: The Matcher Agent (RAG + PyTorch).

Engineered by Sahithi Kodakandla.
Chunks the Job Description, embeds it into PyTorch tensors,
retrieves requirements matching candidate skills, and flags skill gaps.
"""
import re
from typing import Dict, Any, List
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import PyTorchVectorStore
from app.core.pytorch_engine import get_pytorch_engine

# Common technical competencies catalog
KNOWN_TECH_SKILLS = [
    "python", "pytorch", "tensorflow", "fastapi", "flask", "django",
    "docker", "kubernetes", "aws", "gcp", "azure", "sql", "postgresql",
    "mysql", "mongodb", "redis", "git", "linux", "rest api", "graphql",
    "machine learning", "deep learning", "nlp", "rag", "pandas", "numpy",
    "scikit-learn", "ci/cd", "microservices", "spark", "javascript", "react"
]

class MatcherAgent:
    """Autonomous agent performing RAG indexing and candidate-job compatibility analysis."""

    def __init__(self):
        self.vector_store = PyTorchVectorStore()
        self.pytorch_engine = get_pytorch_engine()

    def _extract_skills(self, text: str) -> List[str]:
        """Extracts recognized technical skills from text using token matching."""
        text_lower = text.lower()
        found = []
        for skill in KNOWN_TECH_SKILLS:
            # Word boundary matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill.title())
        return sorted(list(set(found)))

    def analyze(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Executes RAG ingestion, PyTorch embedding, and skill-gap matching."""
        # 1. RAG Chunker: Split JD into chunks
        jd_chunks = DocumentChunker.chunk_text(jd_text)
        
        # 2. RAG Indexing: Embed JD chunks into PyTorch Vector Store
        self.vector_store.clear()
        self.vector_store.add_documents(jd_chunks)

        # 3. Extract skills
        resume_skills = self._extract_skills(resume_text)
        jd_skills = self._extract_skills(jd_text)

        matched_skills = [s for s in jd_skills if s in resume_skills]
        missing_skills = [s for s in jd_skills if s not in resume_skills]

        # 4. RAG Retrieval: For missing skills, retrieve the exact JD context
        retrieved_gaps = []
        for missing in missing_skills[:4]:
            top_chunks = self.vector_store.search(missing, top_k=1)
            context = top_chunks[0]["text"] if top_chunks else "Required technical qualification in JD."
            retrieved_gaps.append({
                "skill": missing,
                "retrieved_jd_context": context
            })

        # 5. Compute PyTorch global similarity score
        semantic_sim = self.pytorch_engine.compute_similarity(resume_text, jd_text)
        
        skill_ratio = len(matched_skills) / max(len(jd_skills), 1)
        overall_score = round((skill_ratio * 0.60 + semantic_sim * 0.40) * 100, 1)

        return {
            "agent_name": "Matcher Agent (RAG + PyTorch)",
            "overall_score": overall_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "retrieved_gaps": retrieved_gaps,
            "indexed_chunks_count": len(jd_chunks)
        }
