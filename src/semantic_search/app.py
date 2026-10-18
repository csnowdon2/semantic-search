"""
FastAPI HTTP server for exposing access to the search stack.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from .search import RetrievalStack
from .load import load_codebase
from pathlib import Path
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before starting server
    biencoder_model = os.getenv("BIENCODER_MODEL")
    crossencoder_model =os.getenv("CROSSENCODER_MODEL")
    code_root = Path('./src/semantic_search')

    corpus = load_codebase(code_root, chunk_size=256)
    app.state.search = RetrievalStack(corpus, biencoder_model, crossencoder_model)
    print(f"Codebase loaded with {len(app.state.search.chunks)} chunks")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"status": "ready"}

@app.get("/search")
def search(query: str, request: Request, k: int = 10):
    return request.app.state.search.retrieve(query, k=k)
