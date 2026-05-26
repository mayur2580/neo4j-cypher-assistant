import io
import os
import re
import json
import pdfplumber
from groq import Groq

from neo4j_utils import get_driver
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────
# GROQ ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────

def _groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def extract_entities(chunk_text: str) -> dict:
    """
    Use Groq to extract structured entities from a chunk.
    Returns dict with keys: people, organizations, statistics, concepts, topics
    """
    client = _groq_client()

    prompt = f"""Extract named entities from the following text.
Return ONLY a valid JSON object with these keys:
- "people": list of person names mentioned
- "organizations": list of company/institution names
- "statistics": list of numeric facts or percentages (e.g. "$142.3 billion", "37.3%")
- "concepts": list of technical or domain concepts (e.g. "Machine Learning", "GDPR")
- "topics": list of high-level topics or themes covered

Keep each list concise (max 5 items each). If none found, use empty list [].
Return ONLY the JSON, no explanation, no markdown.

Text:
{chunk_text}"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

    except Exception:
        return {
            "people": [],
            "organizations": [],
            "statistics": [],
            "concepts": [],
            "topics": []
        }


# ─────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes):

    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        for idx, page in enumerate(pdf.pages):

            text = page.extract_text() or ""

            pages.append({
                "page": idx + 1,
                "text": text
            })

    return pages


def detect_section(text: str) -> str | None:
    """Detect if a chunk starts with a section heading."""
    match = re.match(r'^\s*(\d+[\.\)]?\s+[A-Z][^\n]{3,60})', text.strip())
    return match.group(1).strip() if match else None


# ─────────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_bytes: bytes,
    pdf_name: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j"
):

    try:

        pages = extract_text_from_pdf(pdf_bytes)
        full_text = "\n\n".join(p["text"] for p in pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(full_text)

        driver = get_driver(neo4j_uri, neo4j_user, neo4j_password)

        stats = {
            "document":         pdf_name,
            "total_chunks":     len(chunks),
            "total_pages":      len(pages),
            "next_chunk_links": len(chunks) - 1,
            "people":           0,
            "organizations":    0,
            "statistics":       0,
            "concepts":         0,
            "topics":           0,
            "sections":         0,
            "co_occurrences":   0,
        }

        with driver.session(database=database) as session:

            # ── Constraints ───────────────────────────────────────────
            for constraint in [
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT stat_value IF NOT EXISTS FOR (s:Statistic) REQUIRE s.value IS UNIQUE",
                "CREATE CONSTRAINT section_title IF NOT EXISTS FOR (s:Section) REQUIRE s.title IS UNIQUE",
                "CREATE CONSTRAINT page_number IF NOT EXISTS FOR (p:Page) REQUIRE p.number IS UNIQUE",
            ]:
                session.run(constraint)

            # ── Document node ─────────────────────────────────────────
            session.run("""
                MERGE (d:Document {name: $name})
                SET d.total_chunks = $total_chunks,
                    d.total_pages  = $total_pages,
                    d.char_count   = $char_count
            """,
            name=pdf_name,
            total_chunks=len(chunks),
            total_pages=len(pages),
            char_count=len(full_text))

            # ── Page nodes + HAS_PAGE ─────────────────────────────────
            for page in pages:
                session.run("""
                    MATCH (d:Document {name: $doc})
                    MERGE (pg:Page {number: $number})
                    SET pg.char_count = $char_count
                    MERGE (d)-[:HAS_PAGE]->(pg)
                """,
                doc=pdf_name,
                number=page["page"],
                char_count=len(page["text"]))

            # ── Chunk nodes + HAS_CHUNK + ON_PAGE ────────────────────
            for idx, chunk_text in enumerate(chunks):

                chunk_id = f"{pdf_name}_{idx}"

                # Estimate which page this chunk belongs to
                page_num = 1
                for p in pages:
                    if chunk_text[:80] in p["text"]:
                        page_num = p["page"]
                        break

                session.run("""
                    MATCH (d:Document {name: $doc})
                    MATCH (pg:Page {number: $page_num})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text     = $text,
                        c.index    = $idx,
                        c.page_num = $page_num
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    MERGE (pg)-[:HAS_CHUNK]->(c)
                    MERGE (c)-[:ON_PAGE]->(pg)
                """,
                doc=pdf_name,
                chunk_id=chunk_id,
                text=chunk_text,
                idx=idx,
                page_num=page_num)

            # ── NEXT_CHUNK relationships ──────────────────────────────
            for idx in range(len(chunks) - 1):
                session.run("""
                    MATCH (c1:Chunk {chunk_id: $current_id})
                    MATCH (c2:Chunk {chunk_id: $next_id})
                    MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """,
                current_id=f"{pdf_name}_{idx}",
                next_id=f"{pdf_name}_{idx + 1}")

            # ── Section detection + BELONGS_TO_SECTION ───────────────
            current_section = None
            for idx, chunk_text in enumerate(chunks):

                chunk_id = f"{pdf_name}_{idx}"
                heading  = detect_section(chunk_text)

                if heading:
                    current_section = heading
                    stats["sections"] += 1

                if current_section:
                    session.run("""
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (s:Section {title: $title})
                        MERGE (c)-[:BELONGS_TO_SECTION]->(s)
                    """,
                    chunk_id=chunk_id,
                    title=current_section)

            # ── Groq entity extraction + rich relationships ───────────
            for idx, chunk_text in enumerate(chunks):

                chunk_id = f"{pdf_name}_{idx}"
                entities = extract_entities(chunk_text)

                # People → MENTIONS_PERSON
                for name in entities.get("people", []):
                    if name.strip():
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MERGE (p:Person {name: $name})
                            MERGE (c)-[:MENTIONS_PERSON]->(p)
                        """, chunk_id=chunk_id, name=name.strip())
                        stats["people"] += 1

                # Organizations → MENTIONS_ORG
                for org in entities.get("organizations", []):
                    if org.strip():
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MERGE (o:Organization {name: $name})
                            MERGE (c)-[:MENTIONS_ORG]->(o)
                        """, chunk_id=chunk_id, name=org.strip())
                        stats["organizations"] += 1

                # Statistics → HAS_STATISTIC
                for stat in entities.get("statistics", []):
                    if stat.strip():
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MERGE (s:Statistic {value: $value})
                            MERGE (c)-[:HAS_STATISTIC]->(s)
                        """, chunk_id=chunk_id, value=stat.strip())
                        stats["statistics"] += 1

                # Concepts → MENTIONS_CONCEPT
                for concept in entities.get("concepts", []):
                    if concept.strip():
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MERGE (co:Concept {name: $name})
                            MERGE (c)-[:MENTIONS_CONCEPT]->(co)
                        """, chunk_id=chunk_id, name=concept.strip())
                        stats["concepts"] += 1

                # Topics → ABOUT_TOPIC
                for topic in entities.get("topics", []):
                    if topic.strip():
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MERGE (t:Topic {name: $name})
                            MERGE (c)-[:ABOUT_TOPIC]->(t)
                        """, chunk_id=chunk_id, name=topic.strip())
                        stats["topics"] += 1

                # CO_OCCURS_WITH between concepts in the same chunk
                concepts = [c.strip() for c in entities.get("concepts", []) if c.strip()]
                for i in range(len(concepts)):
                    for j in range(i + 1, len(concepts)):
                        session.run("""
                            MERGE (a:Concept {name: $a})
                            MERGE (b:Concept {name: $b})
                            MERGE (a)-[:CO_OCCURS_WITH]->(b)
                        """, a=concepts[i], b=concepts[j])
                        stats["co_occurrences"] += 1

        driver.close()

        return {
            "success": True,
            "chunks":  len(chunks),
            "stats":   stats
        }

    except Exception as e:

        return {
            "success": False,
            "error":   str(e),
            "chunks":  0,
            "stats":   {}
        }