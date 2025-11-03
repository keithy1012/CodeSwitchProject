import json
from pathlib import Path
from typing import List, Dict, Optional


def change_ids_in_json(file_path: str, start_id: int = 649, output_path: Optional[str] = None) -> List[Dict]:
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


# Example usage when running this script directly
if __name__ == '__main__':
    
    try:
        change_ids_in_json('data/raw/code_switch_data.json', start_id=1)
    except Exception as e:
        print(f"Error: {e}")
