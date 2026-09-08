# TalentRAG — Autonomous Multi-Agent RAG Platform for Technical Hiring

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io)
[![Multi-Agent RAG](https://img.shields.io/badge/Architecture-Multi--Agent%20%7C%20RAG-brightgreen.svg)]()

> **Designed & Engineered by [Sahithi Kodakandla](https://github.com/KodakandlaSahtihi-29)**  
> *B.Tech in Computer Science and Engineering — GITAM University, Visakhapatnam*  
> [GitHub](https://github.com/KodakandlaSahtihi-29) • [LinkedIn](https://www.linkedin.com/in/sahithi-kodakandla-7ba166293/) • [Email](mailto:sahithikodakandla594@gmail.com)

**TalentRAG** is an autonomous multi-agent hiring intelligence platform that combines **Retrieval-Augmented Generation (RAG)** with **PyTorch deep learning embeddings** to screen candidate qualifications, generate skill-gap interview questions, and evaluate technical answers against reference rubrics.

---

## 1. Multi-Agent Architecture

`mermaid
flowchart TD
    subgraph Inputs["1. Input Documents"]
        R[Candidate Resume Text/PDF]
        J[Job Description Text/PDF]
    end

    subgraph RAGCore["2. RAG & PyTorch Embedding Engine"]
        J --> CH[Document Chunker<br/>Paragraph & Bullet-point Segmentation]
        CH --> PT[PyTorch Neural Embeddings<br/>torch.Tensor 64D Projections]
        PT --> VDB[(In-Memory Vector Store)]
    end

    subgraph Agents["3. Autonomous 3-Agent Workflow"]
        R & VDB --> A1["🤖 Agent 1: The Matcher Agent (RAG)<br/>• Computes PyTorch similarity<br/>• Retrieves exact JD requirement context<br/>• Identifies missing skill gaps"]
        A1 -->|Missing Gaps & Retrieved Context| A2["🤖 Agent 2: Technical Interviewer Agent<br/>• Generates 3 targeted questions<br/>• Formulates evaluation rubrics"]
        A2 -->|Targeted Technical Question| C[Candidate Response]
        C & A2 --> A3["🤖 Agent 3: Evaluator Agent<br/>• PyTorch Cosine Similarity Scoring<br/>• Rubric keyword verification<br/>• Delivers constructive feedback"]
    end

    subgraph UI["4. User Experience"]
        A1 & A2 & A3 --> ST[Streamlit Light-Theme Dashboard]
        A1 & A2 & A3 --> API[FastAPI REST Endpoints]
    end
`

---

## 2. The 3 Autonomous Agents

| Agent | Responsibility | Underlying Technology |
| :--- | :--- | :--- |
| **🤖 Agent 1: The Matcher Agent** | Chunks the Job Description, embeds it into PyTorch vectors, and retrieves requirement paragraphs matching the candidate's skills. Flags missing gaps. | Document Chunker, PyTorch Tensors, In-Memory Vector Store |
| **🤖 Agent 2: The Interviewer Agent** | Takes the missing competencies identified by Agent 1 and dynamically drafts 3 code-level, conceptual, and production scenario questions. | Targeted Question Generator, Dynamic Rubric Formulator |
| **🤖 Agent 3: The Evaluator Agent** | Evaluates the candidate's typed answer against the technical rubric using PyTorch Cosine Similarity, assigning an objective accuracy score (0–100%) and feedback. | PyTorch 	orch.nn.functional.cosine_similarity, Rubric Verification |

---

## 3. Technology Stack

* **Language**: Python 3.11+
* **Deep Learning Framework**: **PyTorch (	orch)** — dense vector embeddings and tensor cosine similarity
* **Retrieval-Augmented Generation (RAG)**: Semantic document chunker and vector retrieval
* **Web Frontend**: **Streamlit** (clean, light-theme interface with hidden developer toolbars)
* **REST API**: **FastAPI** & **Uvicorn**
* **Testing**: **PyTest** (6 automated test cases covering PyTorch math, vector retrieval, and agents)

---

## 4. Quick Start Guide

### 1. Clone the repository
`powershell
git clone https://github.com/KodakandlaSahtihi-29/talent-agent-rag.git
cd talent-agent-rag
`

### 2. Install dependencies
`powershell
pip install -r requirements.txt
`

### 3. Run the Streamlit Dashboard
`powershell
streamlit run frontend/streamlit_app.py --server.port 8502
`
Open your browser at: **http://localhost:8502**.

### 4. Run the FastAPI REST Backend (Optional)
`powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001
`
View interactive Swagger API docs at: **http://localhost:8001/docs**.

### 5. Run Automated Tests
`powershell
pytest tests/ -v
`

---

## 5. How to Explain This Project in an Interview

### 30-Second Elevator Pitch:
> *\"In **TalentRAG**, I built an autonomous multi-agent platform that tackles technical recruitment. Instead of treating candidate screening as a black-box, I separated the system into **3 focused agents**: a **Matcher Agent** that uses **RAG** and **PyTorch** vector embeddings to index job descriptions and retrieve context; an **Interviewer Agent** that generates code-level questions targeting the candidate's exact gaps; and an **Evaluator Agent** that objectively grades candidate responses using PyTorch tensor similarity against a technical rubric.\"*

### Top 3 Interview Questions & Answers:

1. **Why did you use RAG instead of feeding the whole job description to an AI?**  
   *Answer*: Standard job descriptions contain boilerplate company overviews, benefits, and administrative details. RAG chunks the document and retrieves only the top-k technical qualification paragraphs relevant to the candidate, drastically reducing token noise and grounding the evaluation in verified requirements.

2. **Where and why did you use PyTorch?**  
   *Answer*: I used **PyTorch (	orch)** to construct the dense embedding tensors and compute the vector similarity using 	orch.nn.functional.cosine_similarity. This provides high-performance, deterministic tensor computations without relying on external proprietary embedding APIs.

3. **How do the agents communicate?**  
   *Answer*: Agent 1 (Matcher) produces structured output containing the detected skill gaps and retrieved JD contexts. Agent 2 (Interviewer) consumes this payload to formulate targeted questions and rubrics. Finally, Agent 3 (Evaluator) consumes the question rubric along with the candidate's answer to produce an accuracy score and actionable feedback.

---

## 6. Author & Contact

**Kodakandla Sahithi**  
*B.Tech in Computer Science and Engineering*  
GITAM University, Visakhapatnam  

* 🌐 **GitHub**: [@KodakandlaSahtihi-29](https://github.com/KodakandlaSahtihi-29)
* 💼 **LinkedIn**: [sahithi-kodakandla](https://www.linkedin.com/in/sahithi-kodakandla-7ba166293/)
* 📧 **Email**: [sahithikodakandla594@gmail.com](mailto:sahithikodakandla594@gmail.com)
