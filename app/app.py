import sys
import streamlit as st
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================
API_URL = "http://127.0.0.1:8000/ask"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================
import requests

from rag.retrieval_pipeline import MedicalRetrievalPipeline
from rag.generator import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Knowledge AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ============================================================
# THEME
# ============================================================

if st.session_state.dark_mode:

    background = "#08101f"
    card = "#101a2d"
    card_hover = "#14233d"
    text = "#f8fafc"
    secondary = "#94a3b8"
    border = "#263550"
    input_bg = "#111827"

else:

    background = "#f8fafc"
    card = "#ffffff"
    card_hover = "#f1f5f9"
    text = "#0f172a"
    secondary = "#64748b"
    border = "#e2e8f0"
    input_bg = "#f1f5f9"


# ============================================================
# THEME
# ============================================================

if st.session_state.dark_mode:

    background = "#08101f"
    card = "#101a2d"
    card_hover = "#14233d"

    text = "#f8fafc"
    secondary = "#94a3b8"

    border = "#263550"
    input_bg = "#111827"

    # Dark mode text colors
    label_color = "#94a3b8"
    input_text = "#f8fafc"
    button_text = "#ffffff"

else:

    background = "#f8fafc"
    card = "#ffffff"
    card_hover = "#f1f5f9"

    text = "#0f172a"
    secondary = "#64748b"

    border = "#e2e8f0"
    input_bg = "#ffffff"

    # Light mode text colors
    label_color = "#475569"
    input_text = "#0f172a"
    button_text = "#ffffff"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GLOBAL APP
    ======================================================== */

    .stApp {{
        background-color: {background};
        color: {text};
    }}


    [data-testid="stHeader"] {{
        background-color: transparent;
    }}


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {{
        background-color: {background};
        border-right: 1px solid {border};
    }}


    .sidebar-title {{
        color: {text};
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 35px;
    }}


    .sidebar-section {{
        color: {secondary};
        font-size: 13px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}


    /* ========================================================
       PIPELINE
    ======================================================== */

    .pipeline-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 18px;
        margin-top: 25px;
    }}


    .pipeline-item {{
        background-color: {input_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        color: {text};
        font-size: 14px;
    }}


    /* ========================================================
       DISCLAIMER
    ======================================================== */

    .disclaimer {{
        background-color: #fff8dc;
        color: #92400e;
        border-left: 4px solid #facc15;
        padding: 14px;
        border-radius: 10px;
        margin-top: 25px;
        font-size: 13px;
    }}


    /* ========================================================
       HERO
    ======================================================== */

    .hero {{
        background: linear-gradient(
            135deg,
            #12304a,
            #202a55
        );

        border: 1px solid #29466b;
        border-radius: 26px;
        padding: 35px;
        margin-bottom: 25px;
    }}


    .hero h1 {{
        color: #ffffff;
        font-size: 38px;
        margin-bottom: 12px;
    }}


    .hero p {{
        color: #cbd5e1;
        font-size: 17px;
        line-height: 1.7;
    }}


    /* ========================================================
       STAT CARDS
    ======================================================== */

    .stat-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 22px;
        height: 150px;
        transition: 0.2s ease;
    }}


    .stat-card:hover {{
        background-color: {card_hover};
        transform: translateY(-3px);
    }}


    .stat-icon {{
        font-size: 28px;
        margin-bottom: 10px;
    }}


    .stat-title {{
        color: {secondary};
        font-size: 13px;
        margin-bottom: 8px;
    }}


    .stat-value {{
        color: {text};
        font-size: 20px;
        font-weight: 700;
    }}


    /* ========================================================
       QUESTION TITLE
    ======================================================== */

    .question-title {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 25px;
        margin-top: 25px;
        margin-bottom: 15px;
    }}


    .question-title h2 {{
        color: {text};
        margin: 0;
        font-size: 28px;
    }}


    /* ========================================================
       TEXTAREA
    ======================================================== */

    textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: 1px solid {border} !important;
    }}


    textarea::placeholder {{
        color: {secondary} !important;
    }}


    /* ========================================================
       INPUT LABELS
    ======================================================== */

    label {{
        color: {label_color} !important;
    }}


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {{
        background-color: {card};
        color: {text};
        border: 1px solid {border};
        border-radius: 10px;
        font-weight: 500;
    }}


    .stButton > button:hover {{
        background-color: {card_hover};
        color: {text};
        border-color: #38bdf8;
    }}


    /* PRIMARY ASK BUTTON */

    .stButton > button[kind="primary"] {{
        background: linear-gradient(
            135deg,
            #38bdf8,
            #818cf8
        );

        color: #ffffff !important;
        border: none;
        font-weight: 700;
    }}


    .stButton > button[kind="primary"]:hover {{
        opacity: 0.9;
    }}


    /* ========================================================
       ANSWER
    ======================================================== */

    .answer-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 25px;
        margin-top: 25px;
    }}


    .answer-title {{
        color: {text};
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 20px;
    }}


    /* ========================================================
       SOURCES
    ======================================================== */

    .source-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 15px;
        padding: 18px;
        margin: 10px 0;
    }}


    .source-title {{
        color: {text};
        font-weight: 700;
        font-size: 17px;
    }}


    .source-category {{
        color: {secondary};
        font-size: 14px;
        margin-top: 5px;
    }}


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {{
        text-align: center;
        color: {secondary};
        border-top: 1px solid {border};
        margin-top: 45px;
        padding: 25px;
        font-size: 13px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Explanation Level</div>',
        unsafe_allow_html=True
    )

    explanation_level = st.selectbox(
        "Explanation Level",
        ["beginner", "student", "advanced"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section">🎨 Appearance</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "☀️ Light Mode"
        if st.session_state.dark_mode
        else "🌙 Dark Mode",
        use_container_width=True
    ):

        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # PIPELINE CARD
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="pipeline-card">

            <div style="
                color: {text};
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 15px;
            ">
                🧠 AI Pipeline
            </div>

            <div class="pipeline-item">
                🔄 Query Rewriting
            </div>

            <div class="pipeline-item">
                🔎 Vector Search
            </div>

            <div class="pipeline-item">
                📊 Document Reranking
            </div>

            <div class="pipeline-item">
                🧩 Context Building
            </div>

            <div class="pipeline-item">
                🤖 LLM Generation
            </div>

            <div class="pipeline-item">
                📈 Request Monitoring
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="disclaimer">
            ⚠️ For educational purposes only.
        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <h1>🩺 Medical Knowledge AI</h1>

        <p>
        An AI-powered medical knowledge assistant using
        <strong>Retrieval-Augmented Generation</strong>,
        semantic search,
        document reranking,
        and LLM generation.
        </p>

    </div>
    """
)


# ============================================================
# STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html(
        """
        <div class="stat-card">

            <div class="stat-icon">📚</div>

            <div class="stat-title">
                Medical Knowledge Base
            </div>

            <div class="stat-value">
                2,706 Chunks
            </div>

        </div>
        """
    )


with col2:

    st.html(
        """
        <div class="stat-card">

            <div class="stat-icon">🔎</div>

            <div class="stat-title">
                Search Engine
            </div>

            <div class="stat-value">
                Semantic Search
            </div>

        </div>
        """
    )


with col3:

    st.html(
        """
        <div class="stat-card">

            <div class="stat-icon">📊</div>

            <div class="stat-title">
                Reranking
            </div>

            <div class="stat-value">
                Cross Encoder
            </div>

        </div>
        """
    )


with col4:

    st.html(
        """
        <div class="stat-card">

            <div class="stat-icon">🤖</div>

            <div class="stat-title">
                LLM
            </div>

            <div class="stat-value">
                Groq
            </div>

        </div>
        """
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.html(
    """
    <div class="question-title">

        <h2>💬 Ask a Medical Question</h2>

    </div>
    """
)


query = st.text_area(
    "Enter your question:",
    value="What is an A1C test?",
    height=100,
    placeholder="Example: What is an A1C test?"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("### 💡 Example Questions")


example_col1, example_col2, example_col3 = st.columns(3)


with example_col1:

    if st.button(
        "What is asthma?",
        use_container_width=True
    ):

        query = "What is asthma?"


with example_col2:

    if st.button(
        "What causes high blood pressure?",
        use_container_width=True
    ):

        query = "What causes high blood pressure?"


with example_col3:

    if st.button(
        "What are diabetes symptoms?",
        use_container_width=True
    ):

        query = "What are the symptoms of diabetes?"


# ============================================================
# ASK BUTTON
# ============================================================

st.markdown("")


ask_button = st.button(
    "🚀 Ask Medical Knowledge AI",
    use_container_width=True,
    type="primary"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "Please enter a medical question."
        )

    else:

     try:

            # ====================================================
            # CALL FASTAPI BACKEND
            # ====================================================

            with st.spinner(
                "Generating educational answer..."
            ):

                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "explanation_level": explanation_level
                    },
                    timeout=180
                )

            # ====================================================
            # HANDLE API RESPONSE
            # ====================================================

            if response.status_code != 200:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.json(
                    response.json()
                )

            else:

                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

                documents = result.get(
                    "documents",
                    []
                )

                monitoring = result.get(
                    "monitoring",
                    {}
                )

                # ====================================================
                # ANSWER HEADER
                # ====================================================

                st.html(
                    """
                    <div class="answer-card">

                        <div class="answer-title">
                            🤖 AI Answer
                        </div>

                    </div>
                    """
                )

                st.markdown(answer)

                # ====================================================
                # RETRIEVAL DETAILS
                # ====================================================

                with st.expander(
                    "🔍 View Retrieval Details"
                ):

                    st.write(
                        "Original Query:",
                        result.get(
                            "query"
                        )
                    )

                    st.write(
                        "Retrieved Documents:",
                        len(documents)
                    )

                    st.write(
                        "Latency:",
                        monitoring.get(
                            "latency_seconds"
                        )
                    )

                    st.write(
                        "Request ID:",
                        monitoring.get(
                            "request_id"
                        )
                    )

                # ====================================================
                # SOURCES
                # ====================================================

                st.markdown(
                    "### 📚 Medical Sources"
                )

                for index, document in enumerate(
                    documents,
                    start=1
                ):

                    title = document.get(
                        "title",
                        "Unknown"
                    )

                    category = document.get(
                        "category",
                        "Unknown"
                    )

                    score = document.get(
                        "rerank_score",
                        0
                    )

                    source_url = document.get(
                        "source_url",
                        ""
                    )

                    st.html(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                Source {index}: {title}
                            </div>

                            <div class="source-category">
                                Category: {category}
                            </div>

                            <div class="source-category">
                                Rerank Score: {score:.4f}
                            </div>

                        </div>
                        """
                    )

                    if source_url:

                        st.link_button(
                            "🔗 Open Medical Source",
                            source_url
                        )

     except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI backend."
            )

            st.info(
                "Make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            )

     except requests.exceptions.Timeout:

            st.error(
                "The request took too long."
            )

     except Exception as error:

            st.error(
                "An error occurred while processing "
                "your question."
            )

            st.exception(error)
# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        🩺 <strong>Medical Knowledge AI</strong>

        <br><br>

        Educational medical information assistant

        <br>

        Powered by RAG • Semantic Search • Reranking • Groq LLM

    </div>
    """
)