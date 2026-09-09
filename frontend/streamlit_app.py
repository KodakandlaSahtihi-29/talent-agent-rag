"""TalentRAG — Autonomous Multi-Agent RAG Platform.

Engineered by Sahithi Kodakandla.
Coordinates 3 autonomous agents (Matcher, Interviewer, Evaluator)
using PyTorch neural tensor embeddings and Retrieval-Augmented Generation (RAG).
"""
import sys
from pathlib import Path
import streamlit as st

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.agents.matcher_agent import MatcherAgent
from app.agents.interviewer_agent import InterviewerAgent
from app.agents.evaluator_agent import EvaluatorAgent

st.set_page_config(
    page_title="TalentRAG — Multi-Agent AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Light Theme CSS
css_file = ROOT_DIR / "frontend" / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load sample data
SAMPLE_RESUME_PATH = ROOT_DIR / "data" / "sample_resume.txt"
SAMPLE_JD_PATH = ROOT_DIR / "data" / "sample_jd.txt"

sample_resume_content = SAMPLE_RESUME_PATH.read_text(encoding="utf-8") if SAMPLE_RESUME_PATH.exists() else ""
sample_jd_content = SAMPLE_JD_PATH.read_text(encoding="utf-8") if SAMPLE_JD_PATH.exists() else ""

# Initialize Agents
@st.cache_resource
def get_agents():
    return MatcherAgent(), InterviewerAgent(), EvaluatorAgent()

matcher_agent, interviewer_agent, evaluator_agent = get_agents()

# Sidebar: Project Architecture & Interview Quick-Story
with st.sidebar:
    st.markdown("### 🤖 TalentRAG Architecture")
    st.markdown("**Autonomous Multi-Agent RAG Platform** designed to automate technical screening and conduct skill-targeted interviews.")
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developer Profile")
    st.markdown("""
    **Sahithi Kodakandla**  
    *B.Tech in Computer Science & Engineering*  
    GITAM University, Visakhapatnam  
    
    [![GitHub](https://img.shields.io/badge/GitHub-KodakandlaSahtihi--29-181717?logo=github)](https://github.com/KodakandlaSahtihi-29/talent-agent-rag)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahithi--Kodakandla-0A66C2?logo=linkedin)](https://www.linkedin.com/in/sahithi-kodakandla-7ba166293/)
    """)
    
    st.markdown("---")
    st.markdown("### 💡 3-Agent Separation of Concerns")
    st.markdown("""
    1. **Agent 1 (Matcher)**: Chunks JD and computes **PyTorch** vector embeddings to find matching skills and context.
    2. **Agent 2 (Interviewer)**: Uses retrieved gaps to generate code-level interview questions.
    3. **Agent 3 (Evaluator)**: Scores candidate responses against the technical rubric using **PyTorch Cosine Similarity**.
    """)

# Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
    <div>
        <h1 style="margin: 0; font-size: 1.95rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 10px;">
            <span>🤖 TalentRAG</span>
            <span style="font-size: 0.75rem; background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 3px 10px; border-radius: 20px; font-weight: 600;">MULTI-AGENT & RAG</span>
        </h1>
        <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.95rem;">
            Autonomous Multi-Agent Hiring Assistant powered by PyTorch Vector Embeddings & RAG
        </p>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
            Crafted by <a href="https://github.com/KodakandlaSahtihi-29" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">Sahithi Kodakandla</a> • GITAM University
        </div>
    </div>
    <div style="text-align: right;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-right: 6px;"></span>
        <span style="color: #334155; font-size: 0.85rem; font-weight: 600;">PyTorch & Agents Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3-Step Navigation Tabs
step1, step2, step3 = st.tabs([
    "1️⃣ Step 1: Ingest & Match (Agent 1: RAG Matcher)",
    "2️⃣ Step 2: Technical Interview (Agent 2: Interviewer Agent)",
    "3️⃣ Step 3: Evaluate & Score (Agent 3: Evaluator Agent)"
])

# -----------------------------------------------------------------------------
# STEP 1: MATCHER AGENT (RAG + PYTORCH)
# -----------------------------------------------------------------------------
with step1:
    st.markdown("### 📄 Candidate Resume & Target Job Description")
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("⚡ Load Realistic Sample Scenario", help="Loads Candidate Profile vs Senior AI & Backend Job Description"):
            st.session_state["resume_input"] = sample_resume_content
            st.session_state["jd_input"] = sample_jd_content
            st.session_state["match_results"] = None
            st.session_state["generated_questions"] = None
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 👤 Candidate Resume")
        uploaded_resume = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="resume_uploader")
        if uploaded_resume is not None:
            try:
                if uploaded_resume.name.lower().endswith(".pdf"):
                    import fitz
                    pdf_bytes = uploaded_resume.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    extracted = "\n".join([page.get_text() for page in doc]).strip()
                else:
                    extracted = uploaded_resume.read().decode("utf-8", errors="ignore").strip()
                if extracted:
                    st.session_state["resume_input"] = extracted
                    st.toast(f"✅ Loaded resume from {uploaded_resume.name}!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        res_default = st.session_state.get("resume_input", sample_resume_content)
        resume_text = st.text_area("Resume Text / Preview:", value=res_default, height=200, key="resume_textarea")
        
    with c2:
        st.markdown("#### 💼 Job Description (JD)")
        uploaded_jd = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"], key="jd_uploader")
        if uploaded_jd is not None:
            try:
                if uploaded_jd.name.lower().endswith(".pdf"):
                    import fitz
                    pdf_bytes = uploaded_jd.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    extracted = "\n".join([page.get_text() for page in doc]).strip()
                else:
                    extracted = uploaded_jd.read().decode("utf-8", errors="ignore").strip()
                if extracted:
                    st.session_state["jd_input"] = extracted
                    st.toast(f"✅ Loaded JD from {uploaded_jd.name}!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        jd_default = st.session_state.get("jd_input", sample_jd_content)
        jd_text = st.text_area("Job Description Text / Preview:", value=jd_default, height=200, key="jd_textarea")

    if st.button("🚀 Run Agent 1: RAG Ingestion & Match", type="primary", use_container_width=True):
        with st.spinner("Agent 1 is chunking JD, computing PyTorch tensor embeddings, and retrieving skill alignment..."):
            match_res = matcher_agent.analyze(resume_text, jd_text)
            st.session_state["match_results"] = match_res
            st.session_state["generated_questions"] = None

    if "match_results" in st.session_state and st.session_state["match_results"]:
        res = st.session_state["match_results"]
        st.markdown("---")
        st.markdown("### 📊 Agent 1 Analysis & RAG Findings")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Overall Match Score", f"{res['overall_score']}%")
        with m2:
            st.metric("Matched Skills Count", f"{len(res['matched_skills'])} competencies")
        with m3:
            st.metric("Missing Gaps to Test", f"{len(res['missing_skills'])} areas")

        st.markdown("#### ✅ Matched Competencies")
        if res["matched_skills"]:
            badges = "".join([f'<span class="skill-badge-matched">✓ {s}</span>' for s in res["matched_skills"]])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No overlapping technical competencies detected.")

        st.markdown("#### ⚠️ Missing Gaps (Retrieved via RAG from JD)")
        if res["retrieved_gaps"]:
            for gap in res["retrieved_gaps"]:
                st.markdown(f"""
                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;">
                    <b style="color: #be123c;">Missing: {gap['skill']}</b>
                    <div style="color: #475569; font-size: 0.85rem; margin-top: 3px;"><b>Retrieved JD Context:</b> "{gap['retrieved_jd_context']}"</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Candidate covers all core requirements specified in the JD!")

# -----------------------------------------------------------------------------
# STEP 2: INTERVIEWER AGENT
# -----------------------------------------------------------------------------
with step2:
    st.markdown("### 🎙️ Agent 2: Technical Interview Question Generator")
    st.caption("Takes the missing skill gaps identified by Agent 1 and autonomously formulates targeted, code-level interview questions.")

    if "match_results" in st.session_state and st.session_state["match_results"]:
        m_data = st.session_state["match_results"]
        missing = m_data["missing_skills"]
        gaps = m_data["retrieved_gaps"]

        if st.button("⚡ Ask Agent 2 to Generate 3 Targeted Questions", type="primary"):
            with st.spinner("Agent 2 is formulating questions based on retrieved gaps..."):
                q_list = interviewer_agent.generate_questions(missing, gaps)
                st.session_state["generated_questions"] = q_list

        if "generated_questions" in st.session_state and st.session_state["generated_questions"]:
            st.markdown("---")
            st.markdown("#### 📋 Generated Technical Questions:")
            for q in st.session_state["generated_questions"]:
                st.markdown(f"""
                <div class="question-box">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span class="agent-pill">Question #{q['question_id']} • {q['question_type']}</span>
                        <span style="font-weight: 700; color: #2563eb;">Target: {q['skill_targeted']}</span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #0f172a;">{q['question_text']}</div>
                    <div style="color: #64748b; font-size: 0.825rem; margin-top: 6px;"><b>Evaluation Criteria (Rubric):</b> {q['evaluation_rubric']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.info("👉 Proceed to **Step 3** to submit candidate answers and receive PyTorch-based grading!")
    else:
        st.warning("Please run **Step 1: Ingest & Match** first so Agent 1 can identify skill gaps for Agent 2 to target.")

# -----------------------------------------------------------------------------
# STEP 3: EVALUATOR AGENT (PYTORCH SCORING)
# -----------------------------------------------------------------------------
with step3:
    st.markdown("### ✍️ Agent 3: Candidate Answer Evaluator")
    st.caption("Compares the candidate's typed technical answer against reference rubrics using PyTorch Cosine Similarity.")

    if "generated_questions" in st.session_state and st.session_state["generated_questions"]:
        q_options = {f"Q{q['question_id']}: {q['question_text']}": q for q in st.session_state["generated_questions"]}
        selected_q_label = st.selectbox("Select Question to Answer:", list(q_options.keys()))
        selected_q = q_options[selected_q_label]

        st.markdown(f"**Target Skill**: {selected_q['skill_targeted']} | **Expected Focus**: *{selected_q['evaluation_rubric']}*")

        # Pre-fill a realistic sample answer button
        if st.button("Fill Realistic Good Answer", help="Inserts a solid technical answer demonstrating the concept"):
            if selected_q['skill_targeted'] == "Docker":
                st.session_state["user_ans"] = "A Docker container shares the host OS kernel and uses Linux namespaces and cgroups for isolation, whereas a VM runs a complete guest OS on top of a hypervisor. For multi-stage builds, I create a builder stage to compile wheels and a slim runtime stage to copy only production artifacts, keeping the final image tiny."
            elif selected_q['skill_targeted'] == "Kubernetes":
                st.session_state["user_ans"] = "A Pod is the smallest deployable compute unit in Kubernetes containing one or more containers sharing network namespaces. Deployments manage pod replicas and rolling updates. Services provide stable DNS and internal load balancing so traffic routes seamlessly across healthy pods with readiness probes."
            else:
                st.session_state["user_ans"] = "Autograd in PyTorch calculates gradients dynamically during the forward pass. In the training loop, we perform model forward pass, calculate loss with criterion, zero the gradients with optimizer.zero_grad(), backpropagate using loss.backward(), and update weights with optimizer.step()."

        ans_val = st.session_state.get("user_ans", "")
        candidate_response = st.text_area("Your Technical Answer:", value=ans_val, height=130, placeholder="Type your technical explanation here...")

        if st.button("🧪 Evaluate Answer with PyTorch Agent", type="primary"):
            with st.spinner("Agent 3 is computing PyTorch tensor cosine similarity against the rubric..."):
                eval_result = evaluator_agent.evaluate_answer(
                    selected_q["question_text"],
                    candidate_response,
                    selected_q["evaluation_rubric"]
                )
                st.session_state["evaluation_output"] = eval_result

        if "evaluation_output" in st.session_state and st.session_state["evaluation_output"]:
            ev = st.session_state["evaluation_output"]
            st.markdown("---")
            st.markdown("#### 🎯 Agent 3 Evaluation Result:")
            
            s_col, r_col = st.columns(2)
            with s_col:
                st.metric("Technical Accuracy Score", f"{ev['score']}%")
            with r_col:
                st.metric("Assessment Rating", ev["rating"])

            st.success(f"💬 **Feedback**: {ev['feedback']}")

            f1, f2 = st.columns(2)
            with f1:
                st.markdown("**Key Concepts Covered:**")
                for k in ev["covered_key_points"]:
                    st.markdown(f"• :green[{k}]")
            with f2:
                st.markdown("**Concepts to Elaborate On:**")
                for m in ev["missed_key_points"]:
                    st.markdown(f"• :red[{m}]")
    else:
        st.warning("Please generate interview questions in **Step 2** first before evaluating candidate answers.")

# Developer Footer
st.markdown("""
<div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 0.85rem;">
    <p style="margin-bottom: 0.35rem; color: #334155;">
        <strong>TalentRAG</strong> — Autonomous Multi-Agent RAG Platform Designed & Engineered by 
        <a href="https://github.com/KodakandlaSahtihi-29" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">Sahithi Kodakandla</a>
    </p>
    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">
        B.Tech Computer Science & Engineering • GITAM University | Powered by PyTorch (torch), RAG & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
