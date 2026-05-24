import os
import json
import uuid
import threading
import time
import sqlite3

from pathlib import Path
from typing import Optional

import httpx
import uvicorn
import numpy as np
import faiss
import pytesseract

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    HTTPException
)

from fastapi.responses import (
    HTMLResponse,
    FileResponse
)

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import BaseModel

# ─────────────────────────────────────────────
# WINDOWS TESSERACT PATH
# ─────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────

load_dotenv()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3"
)

PORT = int(
    os.getenv("PORT", 8001)
)

NGROK_AUTHTOKEN = os.getenv(
    "NGROK_AUTHTOKEN",
    ""
)

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# SQLITE DATABASE
# ─────────────────────────────────────────────

conn = sqlite3.connect(
    "novarag.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS documents (

    id TEXT PRIMARY KEY,

    name TEXT,

    size INTEGER,

    text TEXT,

    chunks INTEGER
)

""")

conn.commit()

# ─────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="NovaRAG Space Station"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────

public_dir = Path("public")

public_dir.mkdir(exist_ok=True)

index_file = public_dir / "index.html"

# DOES NOT CHANGE YOUR GRAPHICS/UI

if not index_file.exists():

    index_file.write_text("""

<!DOCTYPE html>

<html>

<head>

<title>NovaRAG</title>

<style>

body{

background:#0f172a;

color:white;

font-family:Arial;

display:flex;

justify-content:center;

align-items:center;

height:100vh;

margin:0;

}

h1{

font-size:40px;

}

</style>

</head>

<body>

<h1>🚀 NovaRAG Running Successfully</h1>

</body>

</html>

""", encoding="utf-8")

app.mount(
    "/static",
    StaticFiles(directory="public"),
    name="static"
)

# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────

knowledge_base = {}

chunk_store = []

# ─────────────────────────────────────────────
# VECTOR DATABASE
# ─────────────────────────────────────────────

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

dimension = 384

faiss_index = faiss.IndexFlatL2(
    dimension
)

# ─────────────────────────────────────────────
# REBUILD VECTOR INDEX
# ─────────────────────────────────────────────

def rebuild_faiss_index():

    global faiss_index

    faiss_index = faiss.IndexFlatL2(
        dimension
    )

    all_chunks = [

        c["text"]

        for c in chunk_store
    ]

    if all_chunks:

        embeddings = embed_model.encode(
            all_chunks
        )

        faiss_index.add(
            np.array(
                embeddings
            ).astype("float32")
        )

    print(
        f"✅ Rebuilt FAISS index "
        f"with {len(all_chunks)} chunks"
    )

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(
    text,
    chunk_size=800
):

    chunks = []

    if not text:

        return []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(chunk)

    return chunks

# ─────────────────────────────────────────────
# RESTORE DATABASE
# ─────────────────────────────────────────────

cursor.execute(
    "SELECT * FROM documents"
)

saved_docs = cursor.fetchall()

for doc in saved_docs:

    file_id = doc[0]

    text = doc[3]

    chunks = chunk_text(text)

    if chunks:

        embeddings = embed_model.encode(
            chunks
        )

        faiss_index.add(
            np.array(
                embeddings
            ).astype("float32")
        )

        for chunk in chunks:

            chunk_store.append({

                "doc_id": file_id,

                "text": chunk
            })

    knowledge_base[file_id] = {

        "id": doc[0],

        "name": doc[1],

        "size": doc[2],

        "text": doc[3],

        "chunks": doc[4]
    }

print(
    f"✅ Restored {len(saved_docs)} documents"
)

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    query,
    top_k=10
):

    if not chunk_store:

        return ""

    try:

        query_embedding = (
            embed_model.encode(
                [query]
            )
        )

        distances, indices = (
            faiss_index.search(
                np.array(
                    query_embedding
                ).astype("float32"),
                top_k
            )
        )

        results = []

        seen = set()

        for idx in indices[0]:

            if idx < len(chunk_store):

                text = chunk_store[idx]["text"]

                if text not in seen:

                    seen.add(text)

                    results.append(text)

        context = "\n\n".join(results)

        print(
            "\n===== RETRIEVED CONTEXT =====\n"
        )

        print(context[:4000])

        print(
            "\n=============================\n"
        )

        return context

    except Exception as e:

        print(
            "RETRIEVAL ERROR:",
            e
        )

        return ""

# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────

async def ollama_list_models():

    try:

        async with httpx.AsyncClient(
            timeout=10
        ) as c:

            r = await c.get(
                f"{OLLAMA_HOST}/api/tags"
            )

            if r.status_code != 200:

                print(r.text)

                return []

            data = r.json()

            return [

                m.get("name", "")

                for m in data.get(
                    "models",
                    []
                )
            ]

    except Exception as e:

        print(
            "OLLAMA ERROR:",
            e
        )

        return []

async def ollama_generate(
    prompt,
    system,
    model
):

    payload = {

        "model": model,

        "prompt": prompt,

        "system": system,

        "stream": False,

        "options": {

            "temperature": 0.3,

            "num_predict": 700
        }
    }

    try:

        async with httpx.AsyncClient(
            timeout=180
        ) as c:

            r = await c.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload
            )

            if r.status_code != 200:

                return (
                    f"Model failed: {r.text}"
                )

            data = r.json()

            response = data.get(
                "response",
                ""
            )

            if not response.strip():

                return (
                    "No response generated."
                )

            return response

    except Exception as e:

        print(
            "OLLAMA GENERATE ERROR:",
            e
        )

        return (
            f"Generation failed: {e}"
        )

async def ollama_stream(
    prompt,
    system,
    model
):

    payload = {

        "model": model,

        "prompt": prompt,

        "system": system,

        "stream": True,

        "options": {

            "temperature": 0.3,

            "num_predict": 700
        }
    }

    try:

        async with httpx.AsyncClient(
            timeout=180
        ) as c:

            async with c.stream(
                "POST",
                f"{OLLAMA_HOST}/api/generate",
                json=payload
            ) as r:

                async for line in r.aiter_lines():

                    if not line.strip():

                        continue

                    try:

                        chunk = json.loads(line)

                        if chunk.get("response"):

                            yield chunk["response"]

                        if chunk.get("done"):

                            break

                    except Exception:

                        continue

    except Exception as e:

        print(
            "STREAM ERROR:",
            e
        )

        yield "Generation failed."

# ─────────────────────────────────────────────
# OCR + TEXT EXTRACTION
# ─────────────────────────────────────────────

async def extract_text(
    path: Path,
    filename: str
):

    ext = Path(
        filename
    ).suffix.lower()

    try:

        if ext == ".pdf":

            import fitz

            from PIL import Image

            text = ""

            doc = fitz.open(
                str(path)
            )

            for page in doc:

                page_text = page.get_text(
                    "text"
                )

                if page_text.strip():

                    text += (
                        page_text + "\n"
                    )

                else:

                    pix = page.get_pixmap()

                    img = Image.frombytes(

                        "RGB",

                        [pix.width, pix.height],

                        pix.samples
                    )

                    ocr_text = pytesseract.image_to_string(
                        img
                    )

                    text += (
                        ocr_text + "\n"
                    )

            doc.close()

            print(
                "\n===== FINAL TEXT =====\n"
            )

            print(text[:5000])

            print(
                "\n======================\n"
            )

            return text[:30000]

        elif ext == ".docx":

            from docx import Document

            doc = Document(
                str(path)
            )

            text = "\n".join(

                p.text

                for p in doc.paragraphs
            )

            return text[:30000]

        else:

            return path.read_text(

                encoding="utf-8",

                errors="ignore"

            )[:30000]

    except Exception as e:

        print(
            "TEXT EXTRACTION ERROR:",
            e
        )

        return ""

# ─────────────────────────────────────────────
# HYBRID SYSTEM PROMPT
# ─────────────────────────────────────────────

def build_system(context):

    system = f"""

You are NovaRAG AI.

You are both:

1. A document-based RAG assistant
2. A normal AI chatbot

RULES:

- If uploaded documents contain relevant information,
  answer using those documents.

- If uploaded documents do NOT contain the answer,
  answer normally using your general knowledge.

- Prefer document information whenever available.

- Do NOT hallucinate fake document content.

DOCUMENT CONTEXT:

{context}

"""

    return system

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────

@app.get(
    "/",
    response_class=HTMLResponse
)
async def root():

    return FileResponse(
        "public/index.html"
    )

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health():

    models = await ollama_list_models()

    return {

        "ollama":
            True if models else False,

        "ollama_host":
            OLLAMA_HOST,

        "current_model":
            OLLAMA_MODEL,

        "models_available":
            models,

        "docs_loaded":
            len(knowledge_base),

        "total_chunks":
            len(chunk_store),

        "total_vectors":
            faiss_index.ntotal
    }

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

@app.get("/api/models")
async def get_models():

    models = await ollama_list_models()

    return {

        "success": True,

        "models": [

            {
                "name": m
            }

            for m in models
        ],

        "current": OLLAMA_MODEL
    }

# ─────────────────────────────────────────────
# DOCS
# ─────────────────────────────────────────────

@app.get("/api/docs")
async def list_docs():

    return {

        "docs": list(
            knowledge_base.values()
        )
    }

# ─────────────────────────────────────────────
# DELETE DOCUMENT
# ─────────────────────────────────────────────

@app.delete("/api/docs/{doc_id}")
async def delete_document(
    doc_id: str
):

    global chunk_store

    if doc_id not in knowledge_base:

        raise HTTPException(

            status_code=404,

            detail="Document not found"
        )

    del knowledge_base[doc_id]

    chunk_store = [

        c for c in chunk_store

        if c["doc_id"] != doc_id
    ]

    cursor.execute(

        "DELETE FROM documents WHERE id=?",

        (doc_id,)
    )

    conn.commit()

    rebuild_faiss_index()

    print(
        f"🗑 Deleted document: {doc_id}"
    )

    return {

        "success": True,

        "deleted": doc_id
    }

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...)
):

    results = []

    for f in files:

        content = await f.read()

        file_id = str(uuid.uuid4())

        dest = (
            UPLOAD_DIR /
            f"{file_id}_{f.filename}"
        )

        dest.write_bytes(content)

        text = await extract_text(
            dest,
            f.filename
        )

        chunks = chunk_text(text)

        if chunks:

            embeddings = (
                embed_model.encode(chunks)
            )

            faiss_index.add(
                np.array(
                    embeddings
                ).astype("float32")
            )

            for chunk in chunks:

                chunk_store.append({

                    "doc_id": file_id,

                    "text": chunk
                })

        knowledge_base[file_id] = {

            "id": file_id,

            "name": f.filename,

            "size": len(content),

            "text": text,

            "chunks": len(chunks)
        }

        cursor.execute("""

        INSERT INTO documents (

            id,
            name,
            size,
            text,
            chunks

        )

        VALUES (?, ?, ?, ?, ?)

        """, (

            file_id,

            f.filename,

            len(content),

            text,

            len(chunks)
        ))

        conn.commit()

        results.append({

            "id": file_id,

            "name": f.filename,

            "size": len(content),

            "chunks": len(chunks)
        })

    return {

        "success": True,

        "docs": results
    }

# ─────────────────────────────────────────────
# QUERY MODEL
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):

    query: str

    model: Optional[str] = None

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

@app.post("/api/query")
async def query_endpoint(
    req: QueryRequest
):

    model = (
        req.model
        or OLLAMA_MODEL
    )

    local_context = (
        retrieve_relevant_chunks(
            req.query
        )
    )

    system = build_system(
        local_context
    )

    text = await ollama_generate(

        req.query,

        system,

        model
    )

    return {

        "success": True,

        "text": text,

        "model": model,

        "sources": [

            d.get("name", "")

            for d in knowledge_base.values()
        ]
    }

# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket
):

    await ws.accept()

    print(
        "✅ WebSocket Connected"
    )

    try:

        while True:

            raw = await ws.receive_text()

            payload = json.loads(raw)

            query_text = payload.get(
                "query",
                ""
            )

            model = payload.get(
                "model",
                OLLAMA_MODEL
            )

            local_context = (
                retrieve_relevant_chunks(
                    query_text
                )
            )

            system = build_system(
                local_context
            )

            await ws.send_text(
                json.dumps({

                    "type":
                        "stream_start"
                })
            )

            async for chunk in ollama_stream(

                query_text,

                system,

                model
            ):

                await ws.send_text(
                    json.dumps({

                        "type":
                            "stream_chunk",

                        "text":
                            chunk
                    })
                )

            await ws.send_text(
                json.dumps({

                    "type":
                        "stream_end"
                })
            )

    except WebSocketDisconnect:

        print(
            "❌ WebSocket disconnected"
        )

# ─────────────────────────────────────────────
# NGROK
# ─────────────────────────────────────────────

def start_ngrok():

    if not NGROK_AUTHTOKEN:

        print(
            "\n⚠️ NGROK disabled\n"
        )

        return

    try:

        from pyngrok import ngrok

        ngrok.kill()

        ngrok.set_auth_token(
            NGROK_AUTHTOKEN
        )

        tunnel = ngrok.connect(
            addr=PORT,
            bind_tls=True
        )

        print(
            f"\n🌐 Public URL:\n"
            f"{tunnel.public_url}\n"
        )

    except Exception as e:

        print(
            f"\n❌ ngrok error:\n"
            f"{e}\n"
        )

# ─────────────────────────────────────────────
# START NGROK
# ─────────────────────────────────────────────

def run_ngrok_delayed():

    time.sleep(3)

    start_ngrok()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print(
        f"\n🚀 NovaRAG running on "
        f"http://localhost:{PORT}"
    )

    print(
        f"🤖 WebSocket:\n"
        f"ws://localhost:{PORT}/ws"
    )

    print(
        f"🧠 Model: "
        f"{OLLAMA_MODEL}\n"
    )

    threading.Thread(
        target=run_ngrok_delayed,
        daemon=True
    ).start()

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )