import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

st.title("📚 AI Document Assistant")
st.caption(
    "Upload PDF, DOCX, TXT, or MD files — or load supported files from a public Google Drive link."
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}  # source_id -> document records

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_source_ids" not in st.session_state:
    st.session_state.processed_source_ids = set()


# -----------------------------
# Models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Small, easy-to-run Sentence Transformer.
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# Each function returns records with:
# filename, page, text
# -----------------------------
def extract_pdf(file_bytes, filename):
    records = []
    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            records.append(
                {"filename": filename, "page": page_number, "text": text}
            )

    return records


def extract_docx(file_bytes, filename):
    doc = Document(io.BytesIO(file_bytes))

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    # DOCX has no reliable PDF-like page metadata in normal document parsing.
    return [
        {"filename": filename, "page": None, "text": text}
    ] if text else []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [{"filename": filename, "page": None, "text": text}] if text else []


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [{"filename": filename, "page": None, "text": text}] if text else []


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a natural break when possible.
        if end < len(text):
            break_point = text.rfind(" ", start, end)
            if break_point > start + chunk_size // 2:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(records):
    chunks = []

    for record in records:
        for chunk_number, chunk in enumerate(chunk_text(record["text"]), start=1):
            chunks.append(
                {
                    "filename": record["filename"],
                    "page": record["page"],
                    "chunk_id": chunk_number,
                    "text": chunk,
                }
            )

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_store(chunks):
    if not chunks:
        return None, None

    model = load_embedding_model()
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


def rebuild_search_index():
    all_chunks = []

    for source_records in st.session_state.documents.values():
        all_chunks.extend(create_chunks(source_records))

    st.session_state.chunks = all_chunks

    if all_chunks:
        embeddings, index = build_vector_store(all_chunks)
        st.session_state.embeddings = embeddings
        st.session_state.faiss_index = index
    else:
        st.session_state.embeddings = None
        st.session_state.faiss_index = None


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "why", "how",
    "when", "where", "who", "which", "of", "to", "in", "on", "for", "and",
    "or", "with", "from", "about", "does", "do", "can", "could", "would",
    "should", "this", "that", "these", "those", "it", "its", "be", "as"
}


def important_words(text):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


def keyword_scores(question, chunks):
    query_words = important_words(question)
    scores = []

    for chunk in chunks:
        chunk_words = important_words(chunk["text"])
        if not query_words:
            score = 0.0
        else:
            score = len(query_words.intersection(chunk_words)) / len(query_words)

        scores.append(score)

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=TOP_K):
    if not st.session_state.chunks or st.session_state.faiss_index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Get more candidates from semantic search, then combine with keywords.
    candidate_k = min(max(top_k * 3, 10), len(st.session_state.chunks))
    semantic_scores, semantic_ids = st.session_state.faiss_index.search(
        query_embedding, candidate_k
    )

    semantic_map = {
        int(idx): float(score)
        for idx, score in zip(semantic_ids[0], semantic_scores[0])
        if idx >= 0
    }

    key_scores = keyword_scores(question, st.session_state.chunks)

    candidate_ids = set(semantic_map.keys())

    # Add keyword candidates too.
    keyword_ids = np.argsort(key_scores)[::-1][:candidate_k]
    candidate_ids.update(int(i) for i in keyword_ids)

    results = []

    for idx in candidate_ids:
        semantic = semantic_map.get(idx, 0.0)
        keyword = float(key_scores[idx])

        # Both values are approximately 0..1.
        hybrid = 0.75 * semantic + 0.25 * keyword

        result = dict(st.session_state.chunks[idx])
        result["semantic_score"] = semantic
        result["keyword_score"] = keyword
        result["hybrid_score"] = hybrid
        results.append(result)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]


# -----------------------------
# Groq answer
# -----------------------------
def answer_question(question, retrieved_chunks):
    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. Add it to Streamlit secrets as "
            "`GROQ_API_KEY` before asking questions."
        )

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = f", page {chunk['page']}" if chunk["page"] else ""
        context_parts.append(
            f"[Source {i}: {chunk['filename']}{page}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question ONLY using the provided document context.
Do not use outside knowledge.
If the answer is not contained in the provided context, say:
"I couldn't find that information in the provided documents."

Be concise but useful. When appropriate, explain the answer using the
available context. Do not invent citations, facts, or page numbers.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
def download_drive_url(url):
    """
    Supports public Google Drive file/folder links.

    gdown handles common Drive file URLs and public folder URLs.
    A public folder may contain supported PDF/DOCX/TXT/MD files.
    """
    url = url.strip()

    if "drive.google.com" not in url:
        raise ValueError("Please provide a Google Drive file or folder URL.")

    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    # For a folder, gdown downloads supported files into the output folder.
    if "/folders/" in url:
        output_dir = Path(temp_dir)
        gdown.download_folder(
            url,
            output=str(output_dir),
            quiet=True,
            use_cookies=False,
        )
        paths = [
            p for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return [(p.name, p.read_bytes()) for p in paths]

    # For a single public file.
    output_file = Path(temp_dir) / "drive_file"
    downloaded = gdown.download(
        url,
        output=str(output_file),
        quiet=True,
        fuzzy=True,
        use_cookies=False,
    )

    if not downloaded:
        raise ValueError(
            "Google Drive download failed. Make sure the file is public."
        )

    # Try to preserve the original extension from the URL.
    match = re.search(r"[?&]id=([^&]+)", url)
    file_id = match.group(1) if match else "drive_file"

    # Google Drive download does not always expose the filename.
    # Read the response headers as a fallback.
    response = requests.get(url, stream=True, timeout=20)
    content_type = response.headers.get("content-type", "").lower()

    extension = None
    if "pdf" in content_type:
        extension = ".pdf"
    elif "word" in content_type:
        extension = ".docx"
    elif "text" in content_type:
        extension = ".txt"

    if extension is None:
        # User can still use a normal Drive file URL when its extension
        # is visible in the URL; otherwise ask them to use a file link
        # whose downloaded filename is recognizable.
        url_match = re.search(
            r"/([^/?#]+\.(?:pdf|docx|txt|md))(?:[?#]|$)",
            url,
            re.IGNORECASE,
        )
        if url_match:
            extension = Path(url_match.group(1)).suffix.lower()

    if extension is None:
        raise ValueError(
            "Could not determine the Drive file type. "
            "Use a public PDF, DOCX, TXT, or MD file link, or a public folder link."
        )

    final_path = output_file.with_suffix(extension)
    output_file.replace(final_path)

    return [(final_path.name, final_path.read_bytes())]


def process_source(filename, file_bytes):
    source_id = hashlib.sha256(
        filename.encode("utf-8") + b"\0" + file_bytes
    ).hexdigest()

    if source_id in st.session_state.processed_source_ids:
        return False, source_id

    records = extract_document(file_bytes, filename)

    if not records:
        raise ValueError(f"No text could be extracted from {filename}.")

    st.session_state.documents[source_id] = records
    st.session_state.processed_source_ids.add(source_id)

    return True, source_id


# -----------------------------
# Sidebar: document sources
# -----------------------------
with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, MD",
    )

    st.divider()

    st.subheader("Google Drive")
    drive_url = st.text_input(
        "Public Drive file/folder link",
        placeholder="https://drive.google.com/...",
    )

    if st.button("Load from Google Drive", use_container_width=True):
        if not drive_url:
            st.warning("Paste a Google Drive link first.")
        else:
            try:
                with st.spinner("Loading Google Drive files..."):
                    drive_files = download_drive_url(drive_url)

                    added = 0
                    for filename, file_bytes in drive_files:
                        try:
                            was_added, _ = process_source(filename, file_bytes)
                            added += int(was_added)
                        except Exception as exc:
                            st.warning(f"{filename}: {exc}")

                    if added:
                        rebuild_search_index()
                        st.success(f"Loaded {added} new document(s).")
                    else:
                        st.info("No new supported documents were added.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    if st.button("Clear all documents", use_container_width=True):
        st.session_state.documents = {}
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.processed_source_ids = set()
        st.rerun()


# -----------------------------
# Process local uploads
# -----------------------------
if uploaded_files:
    new_files = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        try:
            was_added, _ = process_source(uploaded_file.name, file_bytes)
            new_files += int(was_added)
        except Exception as exc:
            st.error(f"{uploaded_file.name}: {exc}")

    if new_files:
        with st.spinner("Creating chunks and embeddings..."):
            rebuild_search_index()
        st.success(f"Processed {new_files} new document(s).")


# -----------------------------
# Document information
# -----------------------------
st.header("Document Information")

if st.session_state.documents:
    total_records = sum(len(records) for records in st.session_state.documents.values())

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(st.session_state.documents))
    col2.metric("Extracted sections/pages", total_records)
    col3.metric("Created chunks", len(st.session_state.chunks))

    for records in st.session_state.documents.values():
        if not records:
            continue

        first = records[0]
        filename = first["filename"]
        full_text_length = sum(len(r["text"]) for r in records)

        with st.expander(f"📄 {filename}"):
            st.write(f"**Filename:** {filename}")
            st.write(f"**Extracted text characters:** {full_text_length:,}")

            pages = [r["page"] for r in records if r["page"] is not None]
            if pages:
                st.write(f"**Pages with text:** {len(pages)}")
                st.write(f"**Page range:** {min(pages)}–{max(pages)}")
            else:
                st.write("**Page number:** Not available for this file type.")

            preview = records[0]["text"][:1200]
            st.text_area(
                "Text preview",
                preview,
                height=180,
                key=f"preview_{hashlib.md5(filename.encode()).hexdigest()}",
            )
else:
    st.info("Upload a document or load a public Google Drive file/folder to begin.")


# -----------------------------
# Chunk information
# -----------------------------
if st.session_state.chunks:
    st.header("Chunking & Search Index")

    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(st.session_state.chunks))
    c2.metric("Embedding size", st.session_state.embeddings.shape[1])
    c3.metric("FAISS vectors", st.session_state.faiss_index.ntotal)

    with st.expander("View sample chunks"):
        for i, chunk in enumerate(st.session_state.chunks[:5], start=1):
            page = f" | Page {chunk['page']}" if chunk["page"] else ""
            st.markdown(f"**Chunk {i}: {chunk['filename']}{page}**")
            st.write(chunk["text"])
            st.divider()


# -----------------------------
# Q&A
# -----------------------------
st.header("Ask Questions")

question = st.text_input(
    "Ask something about your documents",
    placeholder="e.g. What are the main conclusions?",
)

if st.button("Ask", type="primary", disabled=not bool(question.strip())):
    if not st.session_state.chunks:
        st.warning("Please add a document first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            retrieved = hybrid_search(question, TOP_K)
            answer = answer_question(question, retrieved)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        if retrieved:
            for i, source in enumerate(retrieved, start=1):
                page = f"Page {source['page']}" if source["page"] else "Page not available"

                with st.expander(
                    f"{i}. {source['filename']} — {page} "
                    f"(hybrid score: {source['hybrid_score']:.3f})"
                ):
                    st.write(
                        f"**Semantic score:** {source['semantic_score']:.3f}  \n"
                        f"**Keyword score:** {source['keyword_score']:.3f}"
                    )
                    st.write(source["text"])
        else:
            st.info("No relevant chunks were found.")
