"""Agent 2: The Technical Interviewer Agent.

Engineered by Sahithi Kodakandla.
Generates tailored technical interview questions based on the candidate's
skill gaps and the retrieved JD context.
"""
from typing import Dict, Any, List

# Template question bank based on common technical skills
QUESTION_TEMPLATES = {
    "Docker": {
        "conceptual": "Explain the architectural difference between a Docker container and a Virtual Machine (VM).",
        "practical": "How would you write a multi-stage Dockerfile to minimize image size for a production Python API?",
        "rubric": "Container shares host kernel, cgroups, namespaces vs hypervisor. Multi-stage build separates build dependencies from runtime image using slim base."
    },
    "Kubernetes": {
        "conceptual": "What is the role of a Kubernetes Pod versus a Deployment, and how does a Service enable networking?",
        "practical": "Describe how you would configure a rolling update with liveness and readiness probes to ensure zero-downtime deployments.",
        "rubric": "Pods are smallest deployable units; Deployments manage replica sets and updates; Services provide stable IP/DNS load-balancing. Probes prevent traffic routing to unready pods."
    },
    "Pytorch": {
        "conceptual": "How does PyTorch's dynamic computational graph (Autograd) differ from static graph execution?",
        "practical": "Walk through the typical PyTorch training loop: forward pass, loss calculation, backward pass, and optimizer step.",
        "rubric": "Autograd computes gradients dynamically on-the-fly. Loop calls model(inputs), criterion(outputs, labels), optimizer.zero_grad(), loss.backward(), optimizer.step()."
    },
    "Fastapi": {
        "conceptual": "How does FastAPI leverage Python type hints, Pydantic, and Starlette for asynchronous performance?",
        "practical": "How do you implement dependency injection in FastAPI to manage database sessions across API endpoints?",
        "rubric": "Uses async/await event loops with Starlette, automatic schema validation via Pydantic. Depends(get_db) yields session and safely handles cleanup."
    },
    "Aws": {
        "conceptual": "Explain the difference between compute services like AWS Lambda (serverless) vs EC2 (IaaS).",
        "practical": "How would you design a secure, highly available backend architecture on AWS using VPC, ALB, and RDS?",
        "rubric": "Lambda is event-driven ephemeral execution; EC2 is persistent virtual server. Architecture uses public subnet for ALB, private subnets for EC2/ECS and RDS across multiple AZs."
    },
    "Default": {
        "conceptual": "What are the core design principles and trade-offs when implementing this technology in production?",
        "practical": "Describe a challenging bug or performance bottleneck you encountered with this skill and how you resolved it.",
        "rubric": "Discussion of scalability, latency, memory overhead, testing, and production monitoring."
    }
}


class InterviewerAgent:
    """Autonomous agent that generates targeted technical interview questions."""

    def generate_questions(self, missing_skills: List[str], retrieved_gaps: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generates 3 realistic technical questions focusing on candidate gap areas."""
        questions = []
        
        # Target up to 3 gaps or fallback
        target_skills = missing_skills[:3] if missing_skills else ["Fastapi", "Docker", "Pytorch"]

        q_id = 1
        for skill in target_skills:
            tpl = QUESTION_TEMPLATES.get(skill, QUESTION_TEMPLATES["Default"])
            
            # Select question type
            if q_id == 1:
                q_text = tpl["conceptual"]
                q_type = "Conceptual Understanding"
            elif q_id == 2:
                q_text = tpl["practical"]
                q_type = "System & Implementation"
            else:
                q_text = tpl.get("practical", tpl["conceptual"])
                q_type = "Production Scenario"

            questions.append({
                "question_id": q_id,
                "skill_targeted": skill,
                "question_type": q_type,
                "question_text": f"[{skill}] {q_text}",
                "evaluation_rubric": tpl["rubric"]
            })
            q_id += 1

        return questions
