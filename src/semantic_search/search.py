from chonkie import CodeChunker, Chunk
from fastembed import TextEmbedding
from pathlib import Path
import numpy as np
import BM25

# Documents -> chunking -> embedding -> model

class EmbeddingRetrieval:
    def __init__(self, corpus: list[Chunk]):
        self.model = TextEmbedding(model='jinaai/jina-embeddings-v2-base-code')
        self.chunks = corpus

        embeddings = list(self.model.embed([chunk.text for chunk in corpus]))
        self.embeddings = np.stack(embeddings)

    def retrieve(self, prompt: str, k=20) -> list[int]:
        v = next(self.model.embed(prompt))
        scores = v @ self.embeddings.T
        ranking = sorted(zip(range(len(self.chunks)), scores), key=lambda x: x[1], reverse=True)[:k]
        return [x[0] for x in ranking]

class BM25Retrieval:
    def __init__(self, corpus: list[Chunk]):
        self.chunks = corpus
        self.model = BM25.index([chunk.text for chunk in corpus])

    def retrieve(self, prompt: str, k=20) -> list[int]:
        results = self.model.search([prompt], k=k)[0]
        return [item['id']-1 for item in results]

class RetrievalStack:
    def __init__(self, corpus: list[Chunk]):
        self.chunks = corpus
        self.embedding_model = EmbeddingRetrieval(corpus)
        self.BM25_model = BM25Retrieval(corpus)

    def retrieve(self, prompt: str, k=20):
        xs = self.embedding_model.retrieve(prompt,k)
        ys = self.BM25_model.retrieve(prompt,k)
        print(self.reciprocal_rank_fusion(xs, ys))

    def reciprocal_rank_fusion(self, xs: list[int], ys: list[int]):
        k = len(xs)
        all_values = set(xs).union(set(ys))
        scores = {}
        print(xs,ys)
        for val in all_values:
            score = 0.0
            for ranking in [xs,ys]:
                if val in ranking:
                    score += 1/(60+ranking.index(val))
            scores[val] = score

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]

def load_documents(files: list[Path]) -> list[str]:
    corpus = []
    for file in files:
        with open(file,'r') as f:
            corpus.append(f.read())
    return corpus

def chunk_documents(chunker: CodeChunker, corpus: list[str]) -> list[Chunk]:
    chunks = []
    for code in corpus:
        chunks += chunker.chunk(code)
    return chunks

chunker = CodeChunker(
    language="python",
    tokenizer="character",
    chunk_size=512,
    include_nodes=True
)

files = [Path('./src/semantic_search/search.py')]

corpus = chunk_documents(chunker, load_documents(files))


stack = RetrievalStack(corpus)
stack.retrieve("inference part", k=5)
# just did RRF, next cross-encoder
