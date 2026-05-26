#  PDF → Neo4j Graph RAG Agent

An intelligent Retrieval-Augmented Generation (RAG) system that transforms static PDF documents into a searchable knowledge graph using Neo4j and Large Language Models (LLMs).

## 🚀 Overview

This project allows users to upload PDF files, which are then parsed, chunked, and stored as a graph in Neo4j. A LangGraph-powered agent translates natural language questions into optimized Cypher queries, retrieves the most relevant document chunks, and synthesizes a comprehensive answer.

### Key Workflow:
1. **Ingestion**: `PDF` $\rightarrow$ `Text Extraction` $\rightarrow$ `Chunking` $\rightarrow$ `Neo4j Graph Storage`.
2. **Retrieval**: `User Question` $\rightarrow$ `LLM (Cypher Generation)` $\rightarrow$ `Neo4j Query` $\rightarrow$ `Relevant Chunks`.
3. **Generation**: `Relevant Chunks` $\rightarrow$ `LLM (Answer Synthesis)` $\rightarrow$ `Final Answer`.

## ✨ Features

- **PDF to Graph Pipeline**: Automated extraction and structured storage of PDF content in Neo4j.
- **Natural Language Interface**: Ask questions about your documents without writing Cypher.
- **Graph-Based Retrieval**: Uses graph relationships to maintain document structure and context.
- **Transparent Reasoning**: The UI displays the generated Cypher query and the raw database results for debugging and verification.
- **Interactive Dashboard**: Built with Streamlit for easy document management and querying.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Orchestration**: [LangGraph](https://www.langchain.com/langgraph), [LangChain](https://www.langchain.com/)
- **LLM Provider**: [Groq](https://groq.com/) (Meta-Llama models)
- **Database**: [Neo4j](https://neo4j.com/)
- **PDF Processing**: `pdfplumber`, `sentence-transformers`

## 📋 Prerequisites

- Python 3.10+
- A running Neo4j instance (Local, AuraDB, or Docker)
- A Groq API Key

## ⚙️ Setup & Installation

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/neo4j-cypher-assistant.git
cd neo4j-cypher-assistant
\`\`\`

### 2. Create a virtual environment
\`\`\`bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
\`\`\`

### 3. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Environment Configuration
Create a `.env` file in the root directory and add your credentials:

\`\`\`env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
\`\`\`

## 🚀 Usage

### Running the App
\`\`\`bash
streamlit run app.py
\`\`\`

### How to use:
1. **Ingest PDF**: Go to the "Ingest PDF" tab, upload your document, and click "Ingest PDF". This will create `Document` and `Chunk` nodes in your Neo4j database.
2. **Query Agent**: Switch to the "Query Agent" tab and ask any question regarding the uploaded PDF.
3. **Inspect Schema**: Use the "Database Information" tab to see the current labels and properties stored in your graph.

## 📐 Graph Schema
- **Nodes**: 
    - `Document`: Represents the uploaded file (stores name and total chunks).
    - `Chunk`: Represents a specific segment of text from the PDF.
- **Relationships**:
    - `(:Document)-[:HAS_CHUNK]->(:Chunk)`
    - `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)` (Preserves sequential order of text)
