import itertools
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 10) -> List[str]:
    """ 
    split text into chunks of up to 'chunk_size' tokens (approx. words),
    with 'overlap' words repeated between chunks.
    """

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start = end - overlap
    return chunks

def preprocess_documents(docs: List[dict]) -> List[dict]:
    """
   Given a list of {'source', 'page_number', 'text'},
   return a list of {'source', 'page_number', 'chunk_id', 'text'}.
    """
    processed = []
    for doc in docs:
        chunks = chunk_text(doc("text"))
        for idx, chunk in enumerate(chunks):
            processed.append({
                'source': doc['source'],
                'page_number': doc['page_number'],
                'chunk_id': idx,
                'text': chunk
            })
    return processed