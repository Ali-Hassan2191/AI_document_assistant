# 📚 AI Document Assistant

A simple Streamlit RAG-style document assistant that supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public Google Drive files and folders

The application uses:

1. Document extraction
2. Overlapping text chunking
3. Sentence Transformers embeddings
4. FAISS vector search
5. Simple keyword search
6. Hybrid semantic + keyword ranking
7. Groq `openai/gpt-oss-120b` for answers
8. Retrieved-source display after every answer

## Project files

```text
document_assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

Create a virtual environment if desired, then:

```bash
pip install -r requirements.txt
```

## 2. Add your Groq API key

The app does **not** hardcode the API key.

### Local Streamlit

Create:

```text
.streamlit/secrets.toml
```

Put:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

You can also set the `GROQ_API_KEY` environment variable.

### Streamlit Community Cloud

Open your app's settings:

**App → Settings → Secrets**

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never commit `secrets.toml` to GitHub.

## 3. Run

```bash
streamlit run app.py
```

## 4. How the pipeline works

```text
PDF / DOCX / TXT / MD
        ↓
Text Extraction
        ↓
Overlapping Chunks
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Index
        ↓
        ┌─────────────────────┐
Question → Semantic Search    │
        └─────────────────────┘
                 +
        ┌─────────────────────┐
        │ Keyword Search      │
        └─────────────────────┘
                 ↓
          Hybrid Ranking
                 ↓
       Top Relevant Chunks
                 ↓
              Groq
                 ↓
             Answer
                 ↓
        Retrieved Sources
```

## 5. Embedding model

The app uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is loaded with `st.cache_resource`, so it is not loaded from scratch on every Streamlit rerun.

Documents are processed once per session. A SHA-256 hash of the filename and file bytes is used to avoid processing the same document again.

The extracted documents, chunks, embeddings, and FAISS index are kept in `st.session_state`.

## 6. Chunking

Default values:

```text
Chunk size: 900 characters
Overlap:     150 characters
```

Every chunk keeps:

- filename
- page number when available
- chunk number
- chunk text

For PDF files, page numbers come from the PDF pages.

For DOCX, TXT, and MD files, page numbers are normally unavailable, so they are shown as `Page not available`.

## 7. Hybrid search

The question is embedded and searched against FAISS.

A simple keyword score is also calculated by comparing important words from the question with words in each chunk.

The final score is:

```text
Hybrid score = 0.75 × semantic score
             + 0.25 × keyword score
```

The top chunks are sent to Groq as context.

## 8. Groq model

The application uses:

```text
openai/gpt-oss-120b
```

This is the model identifier documented by Groq for GPT-OSS 120B. The model is called through the Groq Python client.

The prompt instructs the model to answer only from retrieved document context. If the answer is not present, it should say:

```text
I couldn't find that information in the provided documents.
```

## 9. Google Drive

Paste a **public** Google Drive file or folder link into the sidebar.

Supported files:

- PDF
- DOCX
- TXT
- MD

Public folders are downloaded with `gdown`. The downloaded documents then use the same extraction, chunking, embedding, FAISS, keyword-search, and Groq pipeline as local uploads.

For private Drive files, OAuth/authenticated Drive API access would be needed; this simple version intentionally keeps Drive access public and easy to explain.

## 10. Important note about persistence

This version avoids recreating embeddings when the same documents are reused during a Streamlit session.

The data is stored in:

```python
st.session_state
```

This is session-level storage. If the Streamlit server restarts or the session is cleared, documents need to be processed again.

For a production application with a large document collection, use a persistent vector database/index and a document cache on disk or object storage.

## 11. Security

Do not put your Groq API key directly in `app.py`.

Use Streamlit secrets:

```toml
GROQ_API_KEY = "..."
```

Also avoid committing `.streamlit/secrets.toml` to Git.

## 12. Simple explanation for a presentation

You can explain the application in six steps:

**Step 1 — Upload**

The user uploads a document or provides a public Google Drive link.

**Step 2 — Extract**

The app extracts text and keeps the filename and PDF page number when available.

**Step 3 — Chunk**

Long text is divided into smaller overlapping pieces so retrieval can find specific information.

**Step 4 — Embed and index**

Each chunk is converted into a vector using Sentence Transformers and stored in FAISS.

**Step 5 — Search**

When the user asks a question, the app performs semantic vector search and keyword search, then combines their scores.

**Step 6 — Generate**

The best chunks are sent to Groq's `openai/gpt-oss-120b` model. The model is instructed to answer only from those chunks, and the retrieved chunks are displayed as sources.
