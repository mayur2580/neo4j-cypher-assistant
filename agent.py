import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict
from IPython.display import Image, display

# Guarantee sibling modules are importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from groq import Groq
from langgraph.graph import StateGraph, END
import neo4j_utils  

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

def _get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _chat(messages: list, temperature: float = 0) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


class AgentState(TypedDict):
    question:   str
    neo4j_uri:  str
    neo4j_user: str
    neo4j_pass: str
    neo4j_db:   str
    schema:     dict
    cypher:     str
    raw_result: Any
    answer:     str
    error:      str


def fetch_schema(state: AgentState) -> AgentState:
    schema = neo4j_utils.get_schema(state["neo4j_uri"], state["neo4j_user"], state["neo4j_pass"], state["neo4j_db"])
    return {**state, "schema": schema}


def generate_cypher(state: AgentState) -> AgentState:
    schema_str = json.dumps(state["schema"], indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert Neo4j Cypher query generator.\n\n"
                f"Graph schema:\n{schema_str}\n\n"
                "Nodes: Document(name,total_chunks), Chunk(chunk_id,text,index)\n"
                "Relationships: (Document)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:NEXT_CHUNK]->(Chunk)\n\n"
                "IMPORTANT: All document content lives inside Chunk.text. "
                "To answer ANY question about the document content, ALWAYS search Chunk.text using CONTAINS.\n\n"
                "Rules:\n"
                "1. Return ONLY the Cypher query — no markdown, no backticks, no explanation.\n"
                "2. ALWAYS use: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower('<keyword>') RETURN c.text \n"
                "3. Pick the relevant keyword(s) from the question to search for.\n"
                "4. For questions with multiple keywords, use multiple CONTAINS with AND or OR.\n"
                "5. NEVER return 'Cannot answer this' — always attempt a chunk text search.\n\n"
                "Examples:\n"
                "Q: What is the CAGR of the AI market?\n"
                "A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'cagr' RETURN c.text \n\n"
                "Q: What companies are mentioned?\n"
                "A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'compan' RETURN c.text \n\n"
                "Q: What does the report say about job displacement?\n"
                "A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'job' AND toLower(c.text) CONTAINS 'displac' RETURN c.text "
            ),
        },
        {"role": "user", "content": f"Question: {state['question']}"},
    ]
    cypher = _chat(messages)
    if cypher.startswith("```"):
        cypher = "\n".join(l for l in cypher.splitlines() if not l.strip().startswith("```")).strip()
    return {**state, "cypher": cypher}


def execute_cypher(state: AgentState) -> AgentState:
    try:
        result = neo4j_utils.run_cypher(
            state["neo4j_uri"], state["neo4j_user"], state["neo4j_pass"], state["neo4j_db"], state["cypher"]
        )
        return {**state, "raw_result": result, "error": ""}
    except Exception as exc:
        return {**state, "raw_result": None, "error": str(exc)}


def generate_answer(state: AgentState) -> AgentState:
    if state.get("error"):
        return {**state, "answer": f"⚠️ Cypher error: {state['error']}"}
    messages = [
        {
            "role": "system",
            "content": #(
            #     "You are a helpful assistant that answers questions based on retrieved document chunks. "
            #     "Given the raw text chunks from the document, provide a concise, thorough answer. "
            #     "Include all relevant facts, figures, statistics, and context found in the chunks. "
            #     "Use bullet points, tables (if needed/possible) or sections where appropriate to make the answer easy to read. "
            #     "Do not truncate or summarize away important details — be comprehensive."
            #     "If the retrieved chunks do not contain information relevant to the question, say 'The retrieved document chunks do not contain information relevant to the question.'"
            # )
            ("""You are an expert AI assistant specialized in answering questions from retrieved document chunks.

                Your task is to generate accurate, detailed, and context-aware answers strictly based on the provided document chunks.

                Instructions:
                1. Use ONLY the information available in the retrieved chunks.
                2. Provide clear, structured, and comprehensive answers.
                3. Include all relevant:
                - facts
                - figures
                - statistics
                - dates
                - names
                - technical details
                - examples
                - relationships between concepts
                4. Organize the response using:
                - headings 
                - bullet points
                - numbered lists
                - tables - if it helps clarity and readability
                5. Do NOT omit important information for the sake of brevity.
                6. If multiple chunks contain overlapping information, merge them into a coherent response without repetition.
                7. If the answer is partially available, clearly mention what is available and what is missing.
                8. If the retrieved chunks do not contain enough information to answer the question, respond exactly with:
                "The retrieved document chunks do not contain information relevant to the question."
                9. Do NOT make up facts, assumptions, or external knowledge beyond the retrieved content.
                10. Keep the answer readable, professional, and logically structured.

                Response Style:
                - Concise but complete
                - Easy to scan
                - High information density
                - Professionally formatted

                Goal:
                Produce the best possible answer from the retrieved context while maintaining factual accuracy and completeness.
                """),
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\n"
                f"Retrieved document chunks:\n{json.dumps(state.get('raw_result', []), indent=2, default=str)}\n\n"
                "Provide a detailed, well-structured answer based on the above chunks. "
                "Include all relevant numbers, names, and facts mentioned."
            ),
        },
    ]
    return {**state, "answer": _chat(messages)}


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_schema",    fetch_schema)
    g.add_node("generate_cypher", generate_cypher)
    g.add_node("execute_cypher",  execute_cypher)
    g.add_node("generate_answer", generate_answer)
    g.set_entry_point("fetch_schema")
    g.add_edge("fetch_schema",    "generate_cypher")
    g.add_edge("generate_cypher", "execute_cypher")
    g.add_edge("execute_cypher",  "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile()


_graph = None


def run_agent(question: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "neo4j") -> dict:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    final = _graph.invoke({
        "question":   question,
        "neo4j_uri":  neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_pass": neo4j_password,
        "neo4j_db":   database,
        "schema":     {},
        "cypher":     "",
        "raw_result": None,
        "answer":     "",
        "error":      "",
    })
    return {
        "answer":     final.get("answer", "No answer generated."),
        "cypher":     final.get("cypher", ""),
        "raw_result": final.get("raw_result"),
        "error":      final.get("error", ""),
    }

