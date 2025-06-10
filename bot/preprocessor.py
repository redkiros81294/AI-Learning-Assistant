# bot/preprocessor.py
import spacy
from typing import List, Generator, Dict
import gc

nlp = spacy.load("en_core_web_sm")

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_words = sent.split()
        if current_len + len(sent_words) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Carryover overlap
            current_len = len(current_chunk)
        current_chunk.extend(sent_words)
        current_len += len(sent_words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def preprocess_documents(docs: Generator[Dict, None, None], batch_size: int = 10) -> Generator[Dict, None, None]:
    batch = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield from _process_batch(batch)
            batch = []
            gc.collect()
    
    if batch:
        yield from _process_batch(batch)
        gc.collect()

def _process_batch(batch: List[Dict]) -> Generator[Dict, None, None]:
    for doc in batch:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            yield {
                'source': doc['source'],
                'page_number': doc['page_number'],
                'chunk_id': idx,
                'text': chunk
            }