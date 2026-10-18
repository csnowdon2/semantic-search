"""
Methods for loading and chunking documents for input to the semantic
search stack.
"""

from chonkie import CodeChunker, Chunk, OverlapRefinery
from pathlib import Path
from itertools import chain

def chunk_documents(chunker: CodeChunker, overlap: OverlapRefinery, files: list[Path]) -> list[Chunk]:
    """Chunk each file in a list, and return the combined list of chunks"""
    chunked_documents = []
    for file in files:
        text = file.read_text(encoding='utf-8')
        chunks = overlap(chunker.chunk(text))
        for chunk in chunks:
            # Add filepath to chunk metadata so we can map from the chunk back to the file
            chunk.metadata["filepath"] = file

        chunked_documents.append(chunks)
    return list(chain.from_iterable(chunked_documents))


def load_codebase(root_dir: Path, chunk_size: int) -> list[Chunk]:
    """
    Enumerate the python files in a given directory, and break each into chunks.
    Returns the combined list of chunks.
    """
    chunker = CodeChunker(
        language="python",
        tokenizer="character",
        chunk_size=chunk_size,
        include_nodes=True
    )

    overlap = OverlapRefinery(
        tokenizer="character",
        context_size=0.25,
        method="prefix",
        merge=True
    )

    files = list(root_dir.rglob("*.py"))
    corpus = chunk_documents(chunker, overlap, files)
    return corpus

