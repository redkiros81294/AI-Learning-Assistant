from typing import List, Dict
import os


try:
    from langchain.embeddings import OpenAIEmbeddings
    USE_OPENAI = True
except ImportError:
    from langchain.embeddings import SentenceTransformer
    USE_OPENAI = False

class Embedder:
    def __init__(self, openai_api_key: str = None):
        if USE_OPENAI and openai_api_key:
            self.client = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.client = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embeddings for the given texts."""
        if USE_OPENAI:
            return self.client.embed_documents(texts)
        else:
            return self.client.encode(texts, show_progress_bar=False)
        

    def embed_documents(self, docs: List[Dict]) -> List[Dict]:
        """Given  docs with a 'text' field, and an 'embedding' field to each doc,"""
        texts = [d["text"] for d in docs]
        embeddings = self.embed_texts(texts)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb
        return docs