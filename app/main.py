"""FastAPI Application Entry Point for TalentRAG.

Engineered by Sahithi Kodakandla.
Exposes REST endpoints for the 3-agent RAG workflow.
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agents.matcher_agent import MatcherAgent
from app.agents.interviewer_agent import InterviewerAgent
from app.agents.evaluator_agent import EvaluatorAgent

app = FastAPI(
    title="TalentRAG — Autonomous Multi-Agent RAG Hiring Platform",
    description="FastAPI backend coordinating Matcher, Interviewer, and Evaluator agents using PyTorch.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = MatcherAgent()
interviewer = InterviewerAgent()
evaluator = EvaluatorAgent()

class MatchRequest(BaseModel):
    resume_text: str
    jd_text: str

class GenerateQuestionsRequest(BaseModel):
    missing_skills: List[str]
    retrieved_gaps: Optional[List[dict]] = []

class EvaluateAnswerRequest(BaseModel):
    question: str
    candidate_answer: str
    rubric: str

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "project": "TalentRAG",
        "author": "Sahithi Kodakandla",
        "agents": ["MatcherAgent", "InterviewerAgent", "EvaluatorAgent"],
        "engine": "PyTorch + RAG"
    }

@app.post("/api/match", tags=["Agent 1: Matcher"])
def match_candidate(req: MatchRequest):
    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="Resume and Job Description text are required.")
    return matcher.analyze(req.resume_text, req.jd_text)

@app.post("/api/generate-questions", tags=["Agent 2: Interviewer"])
def generate_questions(req: GenerateQuestionsRequest):
    return {
        "agent_name": "Interviewer Agent",
        "questions": interviewer.generate_questions(req.missing_skills, req.retrieved_gaps)
    }

@app.post("/api/evaluate-answer", tags=["Agent 3: Evaluator"])
def evaluate_answer(req: EvaluateAnswerRequest):
    return evaluator.evaluate_answer(req.question, req.candidate_answer, req.rubric)
