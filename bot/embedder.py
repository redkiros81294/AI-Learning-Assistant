import os, gc, json, hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import faiss
import logging
import spacy

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(
        self,
        cache_dir: str = "data/embeddings",
        chunk_cache: str = "data/chunk_cache",
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(chunk_cache, exist_ok=True)
        self.cache_dir = cache_dir
        self.chunk_cache = chunk_cache
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_path = os.path.join(cache_dir, "faiss_metadata.json")
        
        logger.info(f"Initializing model on {self.device}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()
        
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_path = os.path.join(cache_dir, "faiss_flat.index")
        self.nlp = spacy.load("en_core_web_sm")
        self._init_faiss()

    def _init_faiss(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing index")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r") as f:
                self.index_to_doc = {int(k): tuple(v) for k, v in json.load(f).items()}
        else:
            logger.info("Creating new index")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index_to_doc = {}

    def _chunk_text(self, text: str) -> list:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_len = 0
        max_len = 512
        overlap = 64
        
        for sent in sentences:
            words = sent.split()
            if current_len + len(words) > max_len:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap else []
                current_len = len(current_chunk)
            current_chunk.extend(words)
            current_len += len(words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def embed_and_index(self, doc_id: int, text: str):
        chunks = self._chunk_text(text)
        logger.info(f"Processing document {doc_id} with {len(chunks)} chunks")
        
        for cid, chunk in enumerate(chunks):
            chash = hashlib.md5(chunk.encode()).hexdigest()
            cache_file = os.path.join(self.chunk_cache, f"doc{doc_id}_chunk{cid}_{chash}.npy")
            
            if os.path.exists(cache_file):
                emb = np.load(cache_file)
            else:
                emb = self.model.encode([chunk], normalize_embeddings=True)
                emb = emb.astype("float32")
                np.save(cache_file, emb)
            
            emb = emb.reshape(1, -1)
            vid = self.index.ntotal
            self.index.add(emb)
            self.index_to_doc[vid] = (doc_id, cid)
            
            text_path = os.path.join(self.chunk_cache, f"doc{doc_id}_chunk{cid}.txt")
            Path(text_path).write_text(chunk)
            
            del emb
            gc.collect()
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump({str(k): v for k, v in self.index_to_doc.items()}, f)

    def query(self, question: str, top_k: int = 3) -> list:
        question = self._expand_query(question)
        emb = self.model.encode([question], normalize_embeddings=True)
        emb = emb.astype("float32").reshape(1, -1)
        
        distances, indices = self.index.search(emb, top_k)
        return [self.index_to_doc[idx] for idx in indices[0] if idx in self.index_to_doc]

    def _expand_query(self, query: str) -> str:
        expansions = {
            "DFS": "Depth-First Search",
            "BFS": "Breadth-First Search",
            "CSP": "Constraint Satisfaction Problem",
             "AI": "Artificial Intelligence",
            "ML": "Machine Learning"
        }
        for short, long in expansions.items():
            query = query.replace(short, f"{short} ({long})")
        return query

    def get_chunk_text(self, doc_id: int, chunk_id: int) -> str:
        path = os.path.join(self.chunk_cache, f"doc{doc_id}_chunk{chunk_id}.txt")
        return Path(path).read_text() if Path(path).exists() else ""