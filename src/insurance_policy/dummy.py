from abc import ABC

import numpy as np

from langchain_core.embeddings.embeddings import Embeddings

VECTOR_SIZE = 4096

def dummy_embeddings(size: int):
    return np.random.random(size=(size,)).tolist()


class DummyEmbeddings(Embeddings):

    def aembed_query(self, text: str) -> list[float]:
        return dummy_embeddings(VECTOR_SIZE)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            out.append(dummy_embeddings(VECTOR_SIZE))

        return out

    def embed_query(self, text: str) -> list[float]:
        return dummy_embeddings(VECTOR_SIZE)

    def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            out.append(dummy_embeddings(VECTOR_SIZE))

        return out