"""Unit Tests for TalentRAG Multi-Agent RAG Platform.

Engineered by Sahithi Kodakandla.
Validates PyTorch tensor mathematics, document chunking,
vector store retrieval, and all 3 autonomous agents.
"""
import pytest
from app.core.pytorch_engine import get_pytorch_engine
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import PyTorchVectorStore
from app.agents.matcher_agent import MatcherAgent
from app.agents.interviewer_agent import InterviewerAgent
from app.agents.evaluator_agent import EvaluatorAgent

def test_pytorch_tensor_similarity():
    engine = get_pytorch_engine()
    # Identical texts must have similarity near 1.0
    sim_identical = engine.compute_similarity("FastAPI Python REST API", "FastAPI Python REST API")
    assert sim_identical > 0.95
    
    # Completely different texts
    sim_diff = engine.compute_similarity("Deep learning neural networks", "French culinary recipes")
    assert 0.0 <= sim_diff <= 1.0

def test_document_chunker():
    text = """• Requirement 1: Build FastAPI backend services.
• Requirement 2: Deploy Docker containers to Kubernetes.
• Requirement 3: Train neural networks using PyTorch."""
    chunks = DocumentChunker.chunk_text(text)
    assert len(chunks) == 3
    assert "chunk_id" in chunks[0]
    assert "FastAPI" in chunks[0]["text"]

def test_pytorch_vector_store():
    store = PyTorchVectorStore()
    chunks = [
        {"chunk_id": 1, "text": "Python and FastAPI microservices development with Docker."},
        {"chunk_id": 2, "text": "Kubernetes cluster management and Helm chart deployments."},
        {"chunk_id": 3, "text": "PyTorch deep learning model training and inference optimization."}
    ]
    store.add_documents(chunks)
    
    results = store.search("FastAPI REST APIs", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == 1
    assert results[0]["similarity"] > 0.0

def test_matcher_agent():
    matcher = MatcherAgent()
    resume = "Proficient in Python, FastAPI, Docker, SQL, and Git."
    jd = "Looking for an engineer with Python, FastAPI, Docker, and Kubernetes experience."
    
    analysis = matcher.analyze(resume, jd)
    assert analysis["overall_score"] > 50.0
    assert "Python" in analysis["matched_skills"]
    assert "Kubernetes" in analysis["missing_skills"]
    assert len(analysis["retrieved_gaps"]) > 0

def test_interviewer_agent():
    interviewer = InterviewerAgent()
    missing = ["Kubernetes", "Pytorch"]
    gaps = [{"skill": "Kubernetes", "retrieved_jd_context": "Manage Kubernetes clusters."}]
    
    questions = interviewer.generate_questions(missing, gaps)
    assert len(questions) >= 2
    assert "Kubernetes" in questions[0]["skill_targeted"]
    assert "evaluation_rubric" in questions[0]

def test_evaluator_agent():
    evaluator = EvaluatorAgent()
    rubric = "Containers share host kernel, cgroups, namespaces vs hypervisor in VMs."
    answer = "A container shares the host operating system kernel and uses namespaces and cgroups for isolation instead of a full hypervisor."
    
    result = evaluator.evaluate_answer("Explain Docker vs VM", answer, rubric)
    assert result["score"] > 60.0
    assert result["rating"] in ["Strong Technical Depth", "Acceptable Foundation"]
    assert len(result["covered_key_points"]) > 0
