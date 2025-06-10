# bot/nlp.py
import spacy
from typing import List, Tuple

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    doc = nlp(text.lower())
    freq = {}
    for chunk in doc.noun_chunks:
        freq[chunk.text] = freq.get(chunk.text, 0) + 1
    return [kw for kw, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def summarize_text(text: str, max_sentences: int = 3) -> str:
    doc = nlp(text)
    sentences = list(doc.sents)
    return " ".join([sent.text for sent in sentences[:max_sentences]])

def extract_entities(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]