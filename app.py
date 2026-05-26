import os
import streamlit as st
from dotenv import load_dotenv

from ingestion import ingest_pdf
from agent import run_agent
from neo4j_utils import get_schema, test_connection

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_DB   = os.getenv("NEO4J_DATABASE")
HF_TOKEN   = os.getenv("HF_TOKEN")

missing = [k for k, v in {
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USER,
    "NEO4J_PASSWORD": NEO4J_PASS,
    "NEO4J_DATABASE": NEO4J_DB,
    "HF_TOKEN": HF_TOKEN,
}.items() if not v]

if missing:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    st.stop()


st.set_page_config(
    page_title="PDF → Neo4j Agent",
    page_icon="🧠",
    layout="wide"
)

# SIDEBAR

with st.sidebar:

    st.title(" PDF → Neo4j Agent")

    st.markdown("---")

    st.subheader("Neo4j Connection")

    st.write(f"**URI:** `{NEO4J_URI}`")
    st.write(f"**Database:** `{NEO4J_DB}`")

    ok, msg = test_connection(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASS
    )

    if ok:
        st.success(msg)
    else:
        st.error(msg)

    st.markdown("---")

    st.subheader("Environment")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✅ GROQ_API_KEY Loaded")
    else:
        st.error("❌ GROQ_API_KEY Missing")

# HEADER
st.title("PDF → Neo4j Graph RAG Agent")

st.caption(
    "Upload PDF → Store in Neo4j → Ask Questions in Natural Language"
)

# TABS

tab1, tab2, tab3 = st.tabs([
    "Ingest PDF",
    "Query Agent",
    "Database Information"
])

# TAB 1 — INGEST PDF

with tab1:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if st.button("Ingest PDF"):

        if uploaded_file is None:
            st.warning("Please upload a PDF.")
        else:

            pdf_bytes = uploaded_file.read()

            with st.spinner("Ingesting PDF into Neo4j..."):

                result = ingest_pdf(
                    pdf_bytes=pdf_bytes,
                    pdf_name=uploaded_file.name,
                    neo4j_uri=NEO4J_URI,
                    neo4j_user=NEO4J_USER,
                    neo4j_password=NEO4J_PASS,
                    database=NEO4J_DB
                )

            if result["success"]:

                st.success(
                    f"Successfully ingested "
                    f"{result['chunks']} chunks"
                )

                st.json(result["stats"])

            else:

                st.error(
                    f"Ingestion failed: {result['error']}"
                )

# TAB 2 — QUERY AGENT

with tab2:

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # ── Fixed top section ─────────────────────────────────────
    col_head, col_clear = st.columns([8, 1])
    with col_head:
        st.header("Ask Questions")
    with col_clear:
        st.write("")
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    col_input, col_btn = st.columns([8, 1])
    with col_input:
        user_question = st.text_input(
            label="question",
            label_visibility="collapsed",
            placeholder="Ask anything about the PDF...",
            key=f"q_{st.session_state.input_key}"
        )
    with col_btn:
        send = st.button("Send", use_container_width=True, type="primary")

    st.divider()

    # ── Scrollable message area ───────────────────────────────
    chat_area = st.container(height=450, border=False)
    with chat_area:
        if not st.session_state.messages:
            st.info("Ask a question above to get started.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("cypher"):
                    with st.expander("Generated Cypher"):
                        st.code(message["cypher"], language="cypher")
                if message.get("raw_result"):
                    with st.expander("Raw Result"):
                        st.json(message["raw_result"])

    # ── Handle submission ─────────────────────────────────────
    if send and user_question.strip() and not st.session_state.processing:

        q = user_question.strip()
        st.session_state.processing = True
        st.session_state.input_key += 1  # clear input immediately

        st.session_state.messages.append({
            "role": "user",
            "content": q
        })

        with st.spinner("Running Agent..."):
            result = run_agent(
                question=q,
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASS,
                database=NEO4J_DB
            )

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    result["answer"],
            "cypher":     result.get("cypher"),
            "raw_result": result.get("raw_result")
        })

        st.session_state.processing = False
        st.rerun()


# ============================================================
# TAB 3 — SCHEMA
# ============================================================

with tab3:

    st.header("Neo4j Database information")

    if st.button("Refresh Database information"):

        try:

            schema = get_schema(
                NEO4J_URI,
                NEO4J_USER,
                NEO4J_PASS,
                NEO4J_DB
            )

            st.session_state["schema"] = schema

        except Exception as e:

            st.error(str(e))

    if "schema" in st.session_state:

        schema = st.session_state["schema"]

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Labels")

            for label in schema["labels"]:
                st.badge(label)

        with col2:

            st.subheader("Relationships")

            for rel in schema["relationships"]:
                st.badge(rel)

        st.subheader("Property keys")

        st.write(schema["properties"])