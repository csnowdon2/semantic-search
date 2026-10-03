from chonkie import CodeChunker, Chunk
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pathlib import Path
import numpy as np
import BM25


class EmbeddingRetrieval:
    def __init__(self, corpus: list[Chunk]):
        print("Embedding")
        self.model = TextEmbedding(model='jinaai/jina-embeddings-v2-base-code')
        print("Do embed")
        embeddings = list(self.model.embed([chunk.text for chunk in corpus]))
        self.embeddings = np.stack(embeddings)
        print("Done")

    def retrieve(self, prompt: str, k=20) -> list[int]:
        v = next(self.model.embed(prompt))
        scores = v @ self.embeddings.T
        ranking = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)[:k]
        return [x[0] for x in ranking]


class BM25Retrieval:
    def __init__(self, corpus: list[Chunk]):
        print("BM25")
        self.model = BM25.index([chunk.text for chunk in corpus])
        print("Done")

    def retrieve(self, prompt: str, k=20) -> list[int]:
        results = self.model.search([prompt], k=k)[0]
        return [item['id']-1 for item in results]


class CrossEncoderReranker:
    def __init__(self):
        print("Cross encoder")
        self.model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        
        #self.model = TextCrossEncoder(model_name="jinaai/jina-reranker-v1-turbo-en")
        #self.model = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        print("Done")

    def rerank(self, prompt: str, ranking: list[Chunk]):
        new_scores = self.model.rerank(prompt, [chunk.text for chunk in ranking])
        ranking = sorted(zip(ranking, new_scores), key=lambda x: x[1], reverse = True)
        ranking = [x[0] for x in ranking]
        return ranking


class RetrievalStack:
    def __init__(self, corpus: list[Chunk]):
        self.chunks = corpus
        self.initial_rankers = [
                EmbeddingRetrieval(corpus),
                #BM25Retrieval(corpus),
        ]
        self.reranker = CrossEncoderReranker()

        self.retrieve('warmup', 5)


    def retrieve(self, prompt: str, k=20):
        rankings = [ranker.retrieve(prompt,k) for ranker in self.initial_rankers]
        reranked = self.reciprocal_rank_fusion(rankings)
        ranking = [self.chunks[rank] for rank in reranked]
        return self.reranker.rerank(prompt, ranking)


    def reciprocal_rank_fusion(self, rankings: list[list[int]]):
        all_values = set()
        for ranking in rankings:
            all_values = all_values.union(set(ranking))

        scores = {}
        for val in all_values:
            score = 0.0
            for ranking in rankings:
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
    chunk_size=768,
    include_nodes=True
)

#files = [Path('./src/semantic_search/search.py')]
requests_root = Path('.venv/lib64/python3.14/site-packages/requests')
files = list(requests_root.glob("*.py"))

#for model in TextCrossEncoder.list_supported_models():
#    print(model['model'], model['size_in_GB'])
#exit()


prompts = [
        'add arbitrary HTTP header to request',
        'connection pooling',
        'streaming downloads', 
        'file upload',
        'TLS verification',
        'chunked HTTP requests',
        ]
corpus = chunk_documents(chunker, load_documents(files))
print("Chunks:", len(corpus))

print("Make stack")
stack = RetrievalStack(corpus)
print("Done")

for prompt in prompts:
    ranking = stack.retrieve(prompt, k=5)
    print(prompt)
    print(ranking[0])
    print()

