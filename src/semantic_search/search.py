from chonkie import CodeChunker, Chunk
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pathlib import Path
import numpy as np
import BM25

# Documents -> chunking -> embedding -> model

class EmbeddingRetrieval:
    def __init__(self, corpus: list[Chunk]):
        self.model = TextEmbedding(model='jinaai/jina-embeddings-v2-base-code')
        embeddings = list(self.model.embed([chunk.text for chunk in corpus]))
        self.embeddings = np.stack(embeddings)

    def retrieve(self, prompt: str, k=20) -> list[int]:
        v = next(self.model.embed(prompt))
        scores = v @ self.embeddings.T
        ranking = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)[:k]
        return [x[0] for x in ranking]


class BM25Retrieval:
    def __init__(self, corpus: list[Chunk]):
        self.model = BM25.index([chunk.text for chunk in corpus])

    def retrieve(self, prompt: str, k=20) -> list[int]:
        results = self.model.search([prompt], k=k)[0]
        print(results)
        return [item['id']-1 for item in results]


class Reranker:
    def __init__(self):
        self.model = TextCrossEncoder(model_name="BAAI/bge-reranker-base")

    def rerank(self, prompt: str, ranking: list[Chunk]):
        return self.model.rerank(prompt, [chunk.text for chunk in ranking])


class RetrievalStack:
    def __init__(self, corpus: list[Chunk]):
        self.chunks = corpus
        self.embedding_model = EmbeddingRetrieval(corpus)
        self.BM25_model = BM25Retrieval(corpus)

    def retrieve(self, prompt: str, k=20):
        xs = self.embedding_model.retrieve(prompt,k)
        ys = self.BM25_model.retrieve(prompt,k)
        reranked = self.reciprocal_rank_fusion(xs, ys)
        return [self.chunks[rank] for rank in reranked]


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

reranker = Reranker()

stack = RetrievalStack(corpus)
prompt = 'inference part'
ranking = stack.retrieve(prompt, k=5)
new_scores = reranker.rerank(prompt, ranking)
ranking = sorted(zip(ranking, new_scores), key=lambda x: x[1], reverse = True)
ranking = [x[0] for x in ranking]
for s in ranking:
    print("CHUNK")
    print(s)
