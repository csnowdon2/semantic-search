## Overview

This project contains a simple ML-enabled semantic code search app. Semantic code search is a search method which aims 
to incorporate the semantics of a search query rather than just matching based on keywords. For example, you might 
want the search query "attach a header to an HTTP request" to find the function which does that, rather than having 
to know the name of the function to find it. Conventional search methods based on keyword matching are bad at this.
This is achieved through a multi-stage pipeline utilizing semantic text embedding Machine Learning models. 

This project contains a hand-rolled semantic search stack which is served via a FastAPI HTTP server.

### AI Disclaimer

Claude Opus 5.5 was used to help draft the CI and Dockerfile. Everything else was written by me.

## Demo Usage

A demo container image is available on GHCR. The image launches a server which performs semantic code search over 
this codebase. The ML models are baked into the image to reduce startup overhead, but it still takes some time to 
perform the initial embedding. You may get "connection refused" errors while the container is starting up.

```bash
$ podman pull ghcr.io/csnowdon2/semantic-search:1.0.0
$ podman run -d --name semsearch-demo -p 8000:8000 ghcr.io/csnowdon2/semantic-search:1.0.0
# May take a minute for the server to be ready
$ curl http://0.0.0.0:8000/search?query=http%20server
$ podman stop semsearch-demo
```

Sample output:
```json
[{"id":"chnk_b50aa60157ba4fde8c45843fb2e98630","text":"\"\"\"\nFastAPI HTTP server for exposing access to the search stack.\n\"\"\"\n\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI, Request\n\nfrom .search import RetrievalStack\nfrom .load import load_codebase\nfrom pathlib import Path\nimport os\n\n\n","start_index":0,"end_index":255,"token_count":255,"context":null,"embedding":null,"metadata":{"filepath":"/app/src/semantic_search/app.py"}},...]
```

## Source Overview

The source code is in the `src/` directory, and is laid out as follows. The mechanics of semantic search are explained 
in the last section of this README.
- `app.py`: a simple FastAPI HTTP server which serves the search
- `search.py`: a hand-rolled semantic search stack, using `fastembed` for inference
- `load.py`: helper functions for loading and chunking source files, using `chonkie`
- `warmup.py`: a helper which just builds the search stack, used in the Dockerfile to download the models at build time

There is a Dockerfile for building the demo image, and CI for running the build and pushing the image.

## Design Choices

The code has been kept very simple for demonstration purposes.
- The server simply indexes the codebase at startup, and has no functionality for re-indexing at run-time.
- Inference is done with `fastembed`, primarily to avoid having to build `torch` into the container image.
- The demo image uses the `ms-marco-MiniLM-L-6-v2` and `jina-embeddings-v2-base-code` embedding models by default.
  No significant thought went into this choice besides the fact that the models are small, to keep image size down.

If this were to be productionized, there are some major outstanding points for improvement:
- Unit tests and integration tests, built into CI
- An evaluation suite for comparing model choices
- A more robust API for the server

## Semantic Search Theory

Semantic search operates as follows. The broad idea is that we use rough, cheap Machine Learning methods to narrow 
down our search, then we run an expensive, accurate pass over the smaller search space to produce the results.

Firstly, the search corpus is split into chunks. Search queries will return the chunks which best match the query. 
The chunks are pre-processed by semantic embedding, which uses a Machine Learning model to convert each chunk into 
a vector which is intended to represent the semantic meaning of the text.

At search time, the same semantic embedded is done to the query string, and we looks for the chunk embeddings which 
are closest (as vectors) the query embedding. This aims to find the chunks which have the closest meaning to the 
query. However, semantic embedding alone does a poor job of matching queries. For example the query "embedding class" 
might closely match the string "university course", as semantic embeddings lose the context of the actual underlying 
words. To close this gap we also perform a separate search based on keyword-matching with the BM25 algorithm, which 
will do a poor job of matching semantics but a good job of matching exact text. The results of these two searches 
are combined via Reciprocal Rank Fusion, a method of combining rankings which arise from incomparable metrics. This 
approach is used to find the best `k` matches out of all of the chunks. This process is cheap, requiring only 
a single semantic embedding, a matrix product, and a BM25 call, and it provides a good first approximate answer.

Finally, the initial ranking is re-ranked using a semantic embedding cross-encoder model. The cross-encoder scores
the query along with each of the candidate chunks to produce a more accurate evaluation. The re-ranked list is 
returned as the search results.
