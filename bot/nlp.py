import spacy
from typing import List, Dict

_nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract top_k noun-chunks as keywords.
    """
    doc = _nlp(text.lower())
    freq = {}
    for chunk in doc.noun_chucks:
        freq[chunk.text] = freq.get(chunk.text, 0) + 1
    sorted_chunks = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_chunks[:top_k]]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    return the first 'max_sentences' sentences as a simple summary.
    """
    doc = _nlp(text)
    sentences = list(doc.sents)
    return " ".join([sent.text for sent in sentences[:max_sentences]])
   

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """ Return list of (entity_text, entity_label) for named entities."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]