import os
import tempfile

import streamlit as st

from resume_utils import extract_text, analyze_resume

from rag.loader import load_pdf
from rag.chunker import chunk_documents
from rag.vectorstore import add_documents_to_vectorstore
from rag.retriever import retrieve
from rag.generator import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(91, 82, 255, 0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(0, 180, 255, 0.10),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #070914 0%,
            #0a0d1a 48%,
            #080b15 100%
        );

    color: #f4f6ff;
}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.block-container {
    max-width: 1180px;
    padding-top: 42px;
    padding-bottom: 80px;
}


/* ==========================================================
   HEADER
   ========================================================== */

.hero-kicker {
    color: #9da5ff;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 43px;
    font-weight: 800;
    letter-spacing: -1.4px;
    line-height: 1.08;
    margin-bottom: 12px;
}

.hero-title span {
    background: linear-gradient(
        90deg,
        #a8adff,
        #7e8cff,
        #70c8ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #969db3;
    font-size: 15px;
    line-height: 1.7;
    max-width: 700px;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-caption {
    color: #80889f;
    font-size: 13px;
    margin-bottom: 18px;
}


/* ==========================================================
   CARDS
   ========================================================== */

.glass-card {
    background:
        linear-gradient(
            145deg,
            rgba(24, 28, 48, 0.88),
            rgba(13, 16, 29, 0.92)
        );

    border: 1px solid rgba(135, 145, 255, 0.13);

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.025);
}


/* ==========================================================
   INPUTS
   ========================================================== */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(13, 16, 29, 0.9) !important;

    border: 1px solid rgba(142, 151, 255, 0.16) !important;

    border-radius: 11px !important;

    color: #f5f6ff !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: rgba(126, 140, 255, 0.65) !important;

    box-shadow:
        0 0 0 1px rgba(126, 140, 255, 0.18),
        0 0 20px rgba(100, 110, 255, 0.08) !important;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            145deg,
            rgba(20, 24, 42, 0.92),
            rgba(11, 14, 26, 0.96)
        );

    border: 1px dashed rgba(130, 143, 255, 0.32);

    border-radius: 16px;

    padding: 10px;
}


/* ==========================================================
   PRIMARY BUTTON
   ========================================================== */

.stButton > button {
    border-radius: 11px;

    border: 1px solid rgba(134, 145, 255, 0.25);

    background:
        linear-gradient(
            135deg,
            #5965e8,
            #6f65e9
        );

    color: white;

    font-weight: 700;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        filter 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);

    filter: brightness(1.08);

    box-shadow:
        0 0 24px rgba(99, 110, 255, 0.28);
}


/* ==========================================================
   CANDIDATE CARD
   ========================================================== */

.candidate-card {
    background:
        linear-gradient(
            145deg,
            rgba(22, 27, 47, 0.96),
            rgba(12, 15, 27, 0.98)
        );

    border: 1px solid rgba(130, 142, 255, 0.13);

    border-radius: 18px;

    padding: 22px;

    margin-bottom: 15px;

    box-shadow:
        0 16px 45px rgba(0, 0, 0, 0.20);
}

.candidate-name {
    font-size: 18px;
    font-weight: 750;
    color: #f5f6ff;
}

.candidate-meta {
    color: #777f97;
    font-size: 12px;
    margin-top: 4px;
}

.status-selected {
    display: inline-block;

    color: #8cf0bf;

    background: rgba(54, 211, 153, 0.08);

    border: 1px solid rgba(54, 211, 153, 0.18);

    border-radius: 999px;

    padding: 6px 11px;

    font-size: 12px;

    font-weight: 700;

    margin-top: 15px;
}

.status-rejected {
    display: inline-block;

    color: #ff8f9b;

    background: rgba(255, 82, 102, 0.07);

    border: 1px solid rgba(255, 82, 102, 0.15);

    border-radius: 999px;

    padding: 6px 11px;

    font-size: 12px;

    font-weight: 700;

    margin-top: 15px;
}


/* ==========================================================
   TOP CANDIDATE
   ========================================================== */

.top-card {
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(100, 110, 255, 0.18),
            transparent 35%
        ),
        linear-gradient(
            145deg,
            rgba(31, 35, 66, 0.96),
            rgba(14, 17, 32, 0.98)
        );

    border: 1px solid rgba(128, 140, 255, 0.28);

    border-radius: 18px;

    padding: 22px;

    margin: 18px 0;

    box-shadow:
        0 0 35px rgba(91, 102, 255, 0.10);
}

.top-label {
    color: #9da5ff;

    font-size: 12px;

    font-weight: 750;

    letter-spacing: 0.7px;

    text-transform: uppercase;

    margin-bottom: 7px;
}


/* ==========================================================
   FLOATING AI BUTTON
   ========================================================== */

div[data-testid="stVerticalBlock"] .floating-ai {
    position: fixed;

    right: 28px;

    top: 52%;

    transform: translateY(-50%);

    width: 48px;

    height: 48px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        radial-gradient(
            circle,
            rgba(120, 132, 255, 0.38),
            rgba(71, 81, 190, 0.16)
        );

    border: 1px solid rgba(150, 160, 255, 0.42);

    color: #e8eaff;

    font-size: 21px;

    box-shadow:
        0 0 10px rgba(110, 120, 255, 0.30),
        0 0 28px rgba(91, 102, 255, 0.18);

    z-index: 9999;

    animation: aiGlow 2.2s infinite ease-in-out;
}

@keyframes aiGlow {

    0%, 100% {
        box-shadow:
            0 0 10px rgba(110, 120, 255, 0.28),
            0 0 25px rgba(91, 102, 255, 0.12);
    }

    50% {
        box-shadow:
            0 0 16px rgba(110, 120, 255, 0.48),
            0 0 36px rgba(91, 102, 255, 0.25);
    }
}


/* ==========================================================
   AI PANEL
   ========================================================== */

.ai-panel {
    background:
        linear-gradient(
            145deg,
            rgba(24, 28, 50, 0.98),
            rgba(11, 14, 27, 0.98)
        );

    border: 1px solid rgba(129, 141, 255, 0.20);

    border-radius: 18px;

    padding: 22px;

    margin-top: 20px;

    box-shadow:
        0 25px 80px rgba(0, 0, 0, 0.34);
}


/* ==========================================================
   METRIC
   ========================================================== */

.metric-label {
    color: #777f98;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.7px;
}

.metric-value {
    color: #edf0ff;

    font-size: 23px;

    font-weight: 750;

    margin-top: 3px;
}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {
    border-color: rgba(135, 145, 255, 0.08) !important;
}


/* ==========================================================
   HIDE STREAMLIT BRANDING
   ========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero-kicker">
    ✦ AI-POWERED RECRUITMENT
</div>

<div class="hero-title">
    AI Resume <span>Intelligence</span>
</div>

<div class="hero-subtitle">
    Screen candidates, rank talent and ask
    evidence-grounded questions about resumes.
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "candidate_results" not in st.session_state:
    st.session_state.candidate_results = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "ai_open" not in st.session_state:
    st.session_state.ai_open = False

if "rag_question" not in st.session_state:
    st.session_state.rag_question = ""

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ============================================================
# HR REQUIREMENTS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Configure HR Requirements</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-caption">'
    'Define the minimum criteria used for candidate screening.'
    '</div>',
    unsafe_allow_html=True,
)


requirement_card = st.container()

with requirement_card:

    col1, col2 = st.columns(
        [1.4, 0.8],
        gap="large",
    )

    with col1:

        skills = st.text_input(
            "Required Skills",
            value="Python, SQL, Machine Learning",
            placeholder="e.g. Python, SQL, React",
        )

    with col2:

        min_exp = st.number_input(
            "Minimum Experience (years)",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
        )

    col3, col4 = st.columns(
        [0.8, 1.4],
        gap="large",
    )

    with col3:

        min_projects = st.number_input(
            "Minimum Projects",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
        )

    with col4:

        education = st.selectbox(
            "Required Education",
            [
                "Any Degree",
                "B.E",
                "B.Tech",
                "M.E",
                "M.Tech",
                "B.Sc",
                "M.Sc",
                "BCA",
                "MCA",
                "B.Com",
                "M.Com",
                "BBA",
                "MBA",
                "BA",
                "MA",
                "PhD",
            ],
        )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📄 Resume Screening</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-caption">'
    'Upload candidate resumes for AI-powered screening.'
    '</div>',
    unsafe_allow_html=True,
)


uploaded_files = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="visible",
)


if uploaded_files:

    st.caption(
        f"📁 {len(uploaded_files)} resume(s) ready for screening"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_files:

    if st.button(
        "✦ Analyze Resumes",
        use_container_width=True,
    ):

        candidate_results = []
        rag_documents = []

        progress = st.progress(0)

        for index, file in enumerate(uploaded_files):

            pdf_path = None

            try:

                # ------------------------------------------------
                # TEMP PDF
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        file.getbuffer()
                    )

                    pdf_path = temp_file.name


                # ------------------------------------------------
                # RESUME SCREENING
                # ------------------------------------------------

                text = extract_text(
                    pdf_path
                )

                required_skills = [
                    skill.strip()
                    for skill in skills.split(",")
                    if skill.strip()
                ]

                result = analyze_resume(
                    text,
                    required_skills,
                    min_exp,
                    min_projects,
                    education,
                )


                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                candidate_results.append(
                    {
                        "name": file.name,
                        "score": result.get(
                            "score",
                            0,
                        ),
                        "result": result.get(
                            "result",
                            "Rejected",
                        ),
                    }
                )


                # ------------------------------------------------
                # RAG DOCUMENTS
                # ------------------------------------------------

                documents = load_pdf(
                    pdf_path
                )

                for document in documents:

                    if hasattr(
                        document,
                        "metadata",
                    ):

                        document.metadata[
                            "candidate"
                        ] = file.name

                        document.metadata[
                            "source"
                        ] = file.name

                rag_documents.extend(
                    documents
                )

            except Exception as error:

                st.error(
                    f"Failed to process {file.name}: {error}"
                )

            finally:

                if (
                    pdf_path
                    and os.path.exists(pdf_path)
                ):

                    os.remove(
                        pdf_path
                    )

            progress.progress(
                (index + 1)
                / len(uploaded_files)
            )


        # --------------------------------------------------------
        # CREATE RAG KNOWLEDGE BASE
        # --------------------------------------------------------

        if rag_documents:

            try:

                chunks = chunk_documents(
                    rag_documents
                )

                add_documents_to_vectorstore(
                    chunks
                )

                st.session_state.rag_ready = True

                st.session_state.processed_files = [
                    file.name
                    for file in uploaded_files
                ]

            except Exception as error:

                st.session_state.rag_ready = False

                st.error(
                    f"RAG indexing failed: {error}"
                )


        # --------------------------------------------------------
        # SAVE RESULTS
        # --------------------------------------------------------

        st.session_state.candidate_results = (
            candidate_results
        )

        st.rerun()


# ============================================================
# RESULTS
# ============================================================

candidate_results = (
    st.session_state.candidate_results
)


if candidate_results:

    st.markdown(
        '<div class="section-title">📊 Screening Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Candidates are evaluated against the HR requirements.'
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # SELECTED CANDIDATES
    # ========================================================

    selected_candidates = [
        candidate
        for candidate in candidate_results
        if str(
            candidate.get("result", "")
        ).lower()
        == "selected"
    ]


    # ========================================================
    # TOP CANDIDATE
    # ONLY IF AT LEAST ONE SELECTED
    # ========================================================

    if selected_candidates:

        top_candidate = max(
            selected_candidates,
            key=lambda candidate: float(
                candidate.get(
                    "score",
                    0,
                )
            ),
        )

        st.markdown(
            f"""
<div class="top-card">

<div class="top-label">
    🏆 TOP CANDIDATE
</div>

<div class="candidate-name">
    {top_candidate["name"]}
</div>

<div class="candidate-meta">
    Highest eligibility score among shortlisted candidates
</div>

<div style="
    margin-top:14px;
    color:#aeb5ff;
    font-size:24px;
    font-weight:800;
">
    {top_candidate["score"]}%
</div>

<div class="status-selected">
    ✓ Shortlisted
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # CANDIDATE CARDS
    # ========================================================

    for candidate in candidate_results:

        name = candidate.get(
            "name",
            "Unknown",
        )

        score = candidate.get(
            "score",
            0,
        )

        status = candidate.get(
            "result",
            "Rejected",
        )


        if str(status).lower() == "selected":

            status_html = """
<div class="status-selected">
    ✓ Shortlisted
</div>
"""

        else:

            status_html = """
<div class="status-rejected">
    ❌ Rejected — Does not meet the configured requirements
</div>
"""


        st.markdown(
            f"""
<div class="candidate-card">

<div class="candidate-name">
    {name}
</div>

<div class="candidate-meta">
    {name}
</div>

<div style="
    display:flex;
    justify-content:space-between;
    align-items:end;
    margin-top:18px;
">

<div>

<div class="metric-label">
    Eligibility
</div>

<div class="metric-value">
    {score}%
</div>

</div>

<div>
    {status_html}
</div>

</div>

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# FLOATING AI BUTTON
# ============================================================

st.markdown(
    """
<div class="floating-ai">
    ✦
</div>
""",
    unsafe_allow_html=True,
)


# Use a small button visually positioned near the
# floating icon to make it interactive.

# ============================================================
# AI BUTTON
# ============================================================

ai_col1, ai_col2 = st.columns([0.94, 0.06])

with ai_col2:

    if st.button(
        "✦",
        key="floating_ai_button",
        help="Open AI Resume Intelligence",
    ):

        st.session_state.ai_open = (
            not st.session_state.ai_open
        )

        st.rerun()


# ============================================================
# AI PANEL
# ============================================================

if st.session_state.ai_open:

    st.markdown(
        """
<div class="ai-panel">

<div style="
    color:#9da5ff;
    font-size:12px;
    font-weight:750;
    letter-spacing:.6px;
    text-transform:uppercase;
">
    ✦ AI Resume Intelligence
</div>

<div style="
    font-size:23px;
    font-weight:800;
    margin-top:5px;
">
    Recruiter AI
</div>

<div style="
    color:#8189a1;
    font-size:13px;
    margin-top:6px;
">
    Ask recruiter-style questions and receive
    answers grounded in resume evidence.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    if not st.session_state.rag_ready:

        st.info(
            "Upload and analyze at least one resume "
            "to activate Recruiter AI."
        )

    else:

        rag_question = st.text_area(
            "Recruiter question",
            value=st.session_state.rag_question,
            placeholder=(
                "Example: Can we shortlist this candidate "
                "for the role?"
            ),
            height=90,
            key="ai_question_input",
        )


        if st.button(
            "Ask",
            key="ask_ai_button",
            use_container_width=True,
        ):

            if not rag_question.strip():

                st.warning(
                    "Please enter a recruiter question."
                )

            else:

                st.session_state.rag_question = (
                    rag_question
                )

                try:

                    # =========================================
                    # RETRIEVE
                    # =========================================

                    with st.spinner(
                        "Searching resume evidence..."
                    ):

                        retrieved_chunks = retrieve(
                            query=rag_question,
                            top_k=8,
                        )


                    if not retrieved_chunks:

                        st.warning(
                            "The resume does not provide "
                            "enough relevant information."
                        )

                    else:

                        # =====================================
                        # HR REQUIREMENTS FOR AI
                        # =====================================

                        required_skills = [
                            skill.strip()
                            for skill in skills.split(",")
                            if skill.strip()
                        ]

                        hr_requirements = {
                            "skills": required_skills,
                            "min_experience": min_exp,
                            "min_projects": min_projects,
                            "education": education,
                        }


                        # =====================================
                        # GENERATE
                        # =====================================

                        with st.spinner(
                            "Generating recruiter assessment..."
                        ):

                            answer = generate_answer(
                                question=rag_question,
                                retrieved_chunks=(
                                    retrieved_chunks
                                ),
                                hr_requirements=(
                                    hr_requirements
                                ),
                            )


                        # =====================================
                        # ANSWER
                        # =====================================

                        st.markdown(
                            """
<div class="ai-panel">
""",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "### ✦ AI Assessment"
                        )

                        st.markdown(
                            answer
                        )

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )


                        # =====================================
                        # SOURCES
                        # =====================================

                        seen_sources = set()

                        source_items = []

                        for chunk in retrieved_chunks:

                            metadata = chunk.get(
                                "metadata",
                                {},
                            )

                            source = metadata.get(
                                "source",
                                "Unknown",
                            )

                            page = metadata.get(
                                "page",
                                "Unknown",
                            )

                            source_key = (
                                source,
                                page,
                            )

                            if (
                                source_key
                                in seen_sources
                            ):
                                continue

                            seen_sources.add(
                                source_key
                            )

                            source_items.append(
                                f"{source} — Page {page}"
                            )


                        if source_items:

                            st.caption(
                                "Evidence: "
                                + " • ".join(
                                    source_items
                                )
                            )


                except Exception as error:

                    st.error(
                        f"AI error: {error}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div style="
    text-align:center;
    color:#545c72;
    font-size:11px;
    margin-top:55px;
">
    AI Resume Intelligence · RAG · ChromaDB · Qwen
</div>
""",
    unsafe_allow_html=True,
)