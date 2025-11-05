import spacy
import re
import jieba
from typing import List, Dict, Any

# Load English spaCy model once
try:
    NLP_EN = spacy.load("en_core_web_sm", disable=["ner"])
    print("spaCy English model loaded successfully.")
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

def analyze_text_en(utterance: str, turn_id: int) -> List[Dict[str, Any]]:
    """
    Tokenize and analyze mixed English–Chinese utterances with POS tagging.
    Uses spaCy for English and jieba for Chinese. Handles code-switching gracefully.
    """
    if not utterance:
        return []

    # Normalize spacing (ensures consistent splitting)
    utterance = re.sub(r"\s+", " ", utterance.strip())

    # Detect any CJK characters
    contains_cjk = re.search(r"[\u4e00-\u9fff]", utterance) is not None

    results = []
    if contains_cjk:
        # Add spaces around English words/numbers so jieba doesn’t merge them with Chinese
        text_spaced = re.sub(r"([A-Za-z0-9_]+)", r" \1 ", utterance)
        # Tokenize into words and punctuation (CJK + Latin + digits + punctuation)
        pattern = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)
        tokens = pattern.findall(text_spaced)

        # Combine results from jieba + spaCy
        english_buffer = []
        for tok in tokens:
            # CJK run → segment with jieba
            if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
                # Flush pending English tokens first
                if english_buffer:
                    english_doc = NLP_EN(" ".join(english_buffer))
                    for t in english_doc:
                        results.append({
                            "Turn ID": turn_id,
                            "Token": t.text,
                            "Lemma": t.lemma_,
                            "POS Tag": t.pos_,
                            "Syntactic Dependency": t.dep_,
                            "Is Stopword": t.is_stop,
                            "Lang": "en"
                        })
                    english_buffer.clear()

                for sub in jieba.cut(tok):
                    results.append({
                        "Turn ID": turn_id,
                        "Token": sub,
                        "Lemma": sub,
                        "POS Tag": "CJK",
                        "Syntactic Dependency": "",
                        "Is Stopword": False,
                        "Lang": "zh"
                    })
            elif re.fullmatch(r"[A-Za-z]+", tok):
                english_buffer.append(tok)
            elif re.fullmatch(r"\d+", tok):
                results.append({
                    "Turn ID": turn_id,
                    "Token": tok,
                    "Lemma": tok,
                    "POS Tag": "NUM",
                    "Syntactic Dependency": "",
                    "Is Stopword": False,
                    "Lang": "other"
                })
            else:
                # punctuation or symbol
                results.append({
                    "Turn ID": turn_id,
                    "Token": tok,
                    "Lemma": tok,
                    "POS Tag": "PUNCT",
                    "Syntactic Dependency": "",
                    "Is Stopword": False,
                    "Lang": "other"
                })

        # Process any leftover English tokens
        if english_buffer:
            english_doc = NLP_EN(" ".join(english_buffer))
            for t in english_doc:
                results.append({
                    "Turn ID": turn_id,
                    "Token": t.text,
                    "Lemma": t.lemma_,
                    "POS Tag": t.pos_,
                    "Syntactic Dependency": t.dep_,
                    "Is Stopword": t.is_stop,
                    "Lang": "en"
                })

        return results

    # Pure English case
    doc = NLP_EN(utterance)
    for token in doc:
        results.append({
            "Turn ID": turn_id,
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS Tag": token.pos_,
            "Syntactic Dependency": token.dep_,
            "Is Stopword": token.is_stop,
            "Lang": "en"
        })
    return results
