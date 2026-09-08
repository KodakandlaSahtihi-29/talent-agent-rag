"""Agent 3: The Evaluator Agent.

Engineered by Sahithi Kodakandla.
Evaluates candidate interview answers against technical rubrics
using PyTorch Cosine Similarity and semantic keyword verification.
"""
from typing import Dict, Any
from app.core.pytorch_engine import get_pytorch_engine

class EvaluatorAgent:
    """Autonomous agent that objectively scores candidate technical responses."""

    def __init__(self):
        self.pytorch_engine = get_pytorch_engine()

    def evaluate_answer(self, question: str, candidate_answer: str, rubric: str) -> Dict[str, Any]:
        """Scores candidate response against the reference technical rubric."""
        if not candidate_answer or len(candidate_answer.strip()) < 10:
            return {
                "score": 0.0,
                "rating": "Insufficient Answer",
                "feedback": "Answer is too brief. Please provide a detailed technical explanation.",
                "strengths": [],
                "areas_to_improve": ["Elaborate on the technical concepts", "Reference core architectural components"]
            }

        # 1. Compute PyTorch tensor cosine similarity between answer and rubric
        semantic_sim = self.pytorch_engine.compute_similarity(candidate_answer, rubric)
        score = round(semantic_sim * 100, 1)

        # 2. Determine rating band
        if score >= 75:
            rating = "Strong Technical Depth"
            feedback = "Excellent response! You clearly articulated the core concepts and design trade-offs."
        elif score >= 50:
            rating = "Acceptable Foundation"
            feedback = "Good foundation. You mentioned relevant principles, but could expand more on production edge cases."
        else:
            rating = "Developing / Needs Review"
            feedback = "Your answer touches on the topic, but misses key architectural and implementation specifics mentioned in the rubric."

        # 3. Identify keywords from rubric present in candidate answer
        rubric_words = [w.lower() for w in rubric.replace(',', '').replace(';', '').split() if len(w) > 4]
        ans_lower = candidate_answer.lower()
        strengths = [w.title() for w in rubric_words if w in ans_lower][:3]
        areas_to_improve = [w.title() for w in rubric_words if w not in ans_lower][:3]

        return {
            "agent_name": "Evaluator Agent (PyTorch Scoring)",
            "score": score,
            "rating": rating,
            "feedback": feedback,
            "covered_key_points": strengths if strengths else ["General awareness of the concept"],
            "missed_key_points": areas_to_improve if areas_to_improve else ["None - thorough coverage"]
        }
