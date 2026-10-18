"""Construct retrieval stack to pre-download models"""

from .search import RetrievalStack
from .load import load_codebase
from pathlib import Path
import os

if __name__ == "__main__":
    biencoder_model = os.getenv("BIENCODER_MODEL")
    crossencoder_model =os.getenv("CROSSENCODER_MODEL")
    
    code_root = Path('./src/semantic_search')
    corpus = load_codebase(code_root, 512)
    stack = RetrievalStack(corpus, biencoder_model, crossencoder_model)
