import json
from pathlib import Path
from typing import List, Dict, Optional
import re
from datetime import date

def create_readme(path, df):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# Processed Dataset\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Version: 1.0\n\n")
        f.write("## Prompting Strategies Used\n")
        f.write("- Persona-based\n")
        f.write("- Few-shot\n")
        f.write("- Chain-of-Thought\n\n")
        f.write(f"## Number of utterances collected: {len(df)}\n\n")
        f.write("## Preprocessing Notes\n")
        f.write("- JSON normalized into flat tabular structure\n")
        f.write("- Added linguistic features: tokens and POS tags using spaCy\n")
        f.write("- CSV columns: id, utterance, tokens, pos_tags, generation_strategy, model_name, prompt_text, source\n")

    print(f"README.md saved to: {path}")

def countTokens(df):
    total_tokens = []
    total_distinct_tokens = set()
    for tokens in df["tokens"]:
        for token in tokens:
            total_tokens.append(token)
            total_distinct_tokens.add(token)
    return total_tokens, total_distinct_tokens

def countUtterances(df):
    total_utt = 0
    for dialogue in df["utterance"]:
        count_turns = dialogue.count("\n") + 1
        total_utt += count_turns
    return total_utt

def detect_langs(text):
    """Return a list of language labels detected in the text. Currently detects 'zh' (CJK) and 'en' (Latin letters)."""
    if not isinstance(text, str):
        return []
    langs = set()
    if re.search(r"[\u4e00-\u9fff]", text):
        langs.add('zh')
    if re.search(r"[A-Za-z]", text):
        langs.add('en')
    return list(langs)

def count_en_zh_tokens(tokens):
    """Return counts of English and Chinese tokens from a token list."""
    en_count = 0
    zh_count = 0
    for token in tokens:
        if isinstance(token, str):
            if re.search(r"[\u4e00-\u9fff]", token):  # Chinese characters
                zh_count += 1
            elif re.search(r"[A-Za-z]", token):       # English letters
                en_count += 1
    return en_count, zh_count

def compute_mixed_utterance_rate(df):
    """Compute and print the percentage of examples containing both 'en' and 'zh' labels.

    Expects a DataFrame with a `labels` column where each entry is an iterable of language labels.
    """
    if 'labels' not in df.columns:
        raise ValueError("DataFrame does not contain a 'labels' column.")
    mixed = 0
    total = len(df)
    for langs in df['labels']:
        # treat None or non-iterable
        if not langs:
            continue
        lset = set(langs)
        if 'en' in lset and 'zh' in lset:
            mixed += 1
    rate = mixed / total if total > 0 else 0.0
    print(f"Mixed-utterance rate: {rate*100:.2f}%")
    
def change_ids_in_json(file_path: str, start_id: int = 0, output_path: Optional[str] = None) -> List[Dict]:
    """Read a JSON file containing a list of objects and rewrite their 'id' fields.

    Args:
        file_path: path to input JSON file (expected a top-level list of dicts).
        start_id: integer id to assign to the first element (default 649).
        output_path: if provided, write the modified list to this path; otherwise overwrite the input file.

    Returns:
        The modified list of dicts.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {file_path}, got {type(data)}")

    new_id = int(start_id)
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            # skip non-dict entries but still advance the id counter
            data[i] = item
            new_id += 1
            continue
        item['id'] = new_id
        new_id += 1

    out_p = Path(output_path) if output_path else p
    with out_p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} records to {out_p} with ids starting at {start_id}")
    return data


if __name__ == '__main__':
    
    try:
        change_ids_in_json('data/raw/generated_dialogues_monolingual.json', start_id=339)
    except Exception as e:
        print(f"Error: {e}")
