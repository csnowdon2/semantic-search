from chonkie import Chunk
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import numpy as np
import BM25


class EmbeddingRetrieval:
    """
    Semantic embedding biencoder search ranker.
    Initially, performs a semantic embedding on each text chunk.
    To lookup, embeds the query string and finds the closest matching
    chunk embedding.
    """
    def __init__(self, model_name: str, corpus: list[Chunk]):
        self.model = TextEmbedding(model=model_name)
        print("Do embed")
        embeddings = list(self.model.embed([chunk.text for chunk in corpus], batch_size = 128))
        self.embeddings = np.stack(embeddings)
        print("Done")

    def retrieve(self, query: str, k=20) -> list[int]:
        v = next(self.model.embed(query))
        # This matrix product calculates the dot product of the query
        # embedding against each chunk embedding. Each dot product is in
        # the range [-1,1], where higher is better.
        scores = v @ self.embeddings.T
        ranking = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)[:k]
        return [x[0] for x in ranking]


class BM25Retrieval:
    """
    BM25 Keyword-based search ranker.
    """
    def __init__(self, corpus: list[Chunk]):
        print("BM25")
        self.model = BM25.index([chunk.text for chunk in corpus])
        print("Done")

    def retrieve(self, query: str, k=20) -> list[int]:
        results = self.model.search([query], k=k)[0]
        return [item['id']-1 for item in results]


class CrossEncoderReranker:
    """
    Semantic embedding crossencoder search re-ranker.
    More expensive but more accurate than pure semantic embedding.
    Used to re-rank an initial guess from the cheaper models.
    """
    def __init__(self, model_name: str):
        print("Cross encoder")
        self.model = TextCrossEncoder(model_name=model_name)
        print("Done")

    def rerank(self, query: str, ranking: list[Chunk]):
        new_scores = self.model.rerank(query, [chunk.text for chunk in ranking])
        ranking = sorted(zip(ranking, new_scores), key=lambda x: x[1], reverse = True)
        ranking = [x[0] for x in ranking]
        return ranking


class RetrievalStack:
    """
    Composition of the ML components into a full semantic search
    capability.

    Initially, ranks the query against each chunk via semantic embedding
    and BM25. The two rankings are combined with Reciprocal Rank Fusion.
    The results are then re-ranked using a more expensive cross-encoder
    ranker.
    """
    def __init__(self, corpus: list[Chunk], biencoder_model_name: str, crossencoder_model_name: str):
        self.chunks = corpus
        self.initial_rankers = [
                EmbeddingRetrieval(biencoder_model_name, corpus),
                BM25Retrieval(corpus),
        ]
        self.reranker = CrossEncoderReranker(crossencoder_model_name)

        self.retrieve('warmup', 5)


    def retrieve(self, query: str, k=20):
        rankings = [ranker.retrieve(query,k) for ranker in self.initial_rankers]
        reranked = self.reciprocal_rank_fusion(rankings)
        ranking = [self.chunks[rank] for rank in reranked]
        return self.reranker.rerank(query, ranking)


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


