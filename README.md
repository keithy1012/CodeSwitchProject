# CodeSwitchProject

A compact pipeline for analyzing English–Mandarin code-switching in conversational dialogs with a focus on tourist and travel contexts.

## Summary

This repository contains tools to:

- Normalize raw bilingual dialog JSON into a per-turn tabular dataset.
- Tokenize and POS-tag mixed-language text (spaCy for English, jieba for Chinese when available).
- Build token-level cumulative contexts and labels for predicting language switch points at token boundaries.
- Train lightweight TF-IDF + classifiers (Logistic Regression, Random Forest) to detect switch points and apply them to dialogues.

Intended use: research experiments, dataset generation and preprocessing, baseline modeling for switch-point detection.

## Quick start

1. Create and activate a virtual environment (Python 3.8+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux (use the appropriate command on Windows)
```

2. Install core dependencies (add extras as needed):

```bash
pip install -r requirements.txt
# if you don't have a requirements.txt, at minimum install:
pip install pandas scikit-learn spacy joblib
python -m spacy download en_core_web_sm
# Optional (recommended for better Chinese segmentation):
pip install jieba
```

3. Run the end-to-end pipeline:

```bash
python scripts/main.py
```

This will read dialogues from `data/raw/code_switch_data.json`, produce a per-turn processed CSV under `data/processed/<date>/processed_dataset.csv`, train models (or use pre-split train/test if present), and write outputs and visualizations to the processed directory.

## What the pipeline does

- Loads raw JSON dialogues and normalizes them into a DataFrame.
- Splits multi-turn dialogues into single-turn rows (adds `dialogue_text`, `turn_index`, `speaker`, `utterance`).
- Runs tokenization and POS tagging using `scripts/pos_extraction.py` (spaCy + jieba fallback).
- Builds token-level cumulative contexts and heuristic switch labels (change in token language between adjacent tokens).
- Performs train/test split (by default inside `main.py`) and trains both Logistic Regression and Random Forest classifiers.
- Applies the trained models to the dataset and saves predictions and evaluation columns (`predicted_switch_probs`, `predicted_switches`, `actual_switches`, `switch_match_rate`, etc.).
- Produces summary visualizations under the processed `figures/` directory.

## Files & structure

- `data/raw/` — Source JSON files (e.g. `code_switch_data.json`, generated dialogues).
- `data/processed/` — Outputs from the pipeline: processed CSVs, train/test splits, model output CSVs, and figures.
- `scripts/` — Main code and helpers:
  - `main.py` — End-to-end driver that preprocesses, builds contexts, splits data, trains models, applies predictions, and writes outputs.
  - `pos_extraction.py` — Tokenization/POS utilities. Uses spaCy for English and jieba for Chinese (if installed).
  - `logistic_regression.py` — Prepares contexts/labels and trains a TF-IDF + Logistic Regression model. New API allows passing pre-split train/test contexts and labels.
  - `random_forest.py` — Same as above but trains a Random Forest classifier.
  - `utils.py` — Small helper utilities (token counts, saving README, etc.).

## Model labels and outputs

- Token labels: heuristic assignment using character-based regex (`zh` for CJK, `en` for A–Z tokens, `other` otherwise).
- Switch label: 1 when adjacent tokens differ in language, 0 otherwise.
- In `main.py` after model application you will find per-row columns:
  - `predicted_switch_probs` — float probabilities for switch at each token boundary.
  - `predicted_switches` — binary predictions derived by threshold (default 0.5).
  - `actual_switches` — ground-truth 0/1 list derived from `Language` (or fallback rules).
  - `switch_match_rate` — fraction of boundaries where predicted == actual for that row.

## Tips & recommendations

- Install `jieba` for better Chinese segmentation — without it the pipeline falls back to character-level tokenization for CJK runs.
- Persist trained models with `joblib.dump()` if you plan to reuse models without retraining. Consider adding a CLI flag to `main.py` to load persisted models.
- Ensure tokenization used during training matches inference exactly (the pipeline now constructs contexts in `main.py` and passes splits to the training functions to guarantee consistency).

## Reproducibility

- `main.py` saves `train_contexts.csv` and `test_contexts.csv` under the processed directory for reproducible train/eval runs.
- If you need deterministic LLM outputs for data generation, set a fixed random seed in generation scripts or record the RNG/model variants used.

## Troubleshooting

- spaCy model errors: run `python -m spacy download en_core_web_sm`.
- Missing packages: install the dependencies listed above or create a `requirements.txt` from your environment.
- Chinese tokens appear as single characters: install `jieba`.
- If model predictions are poor: check tokenization alignment between training and inference, inspect `switch_match_rate` and label balance, and consider training with more data.

## Next steps (ideas)

- Add model persistence and a `--load-model` flag in `main.py`.
- Expand QA and manual annotation steps for higher-quality ground truth.
- Add unit tests around tokenization and label creation.

## Contact

Project lead: Keith Yao — keith@example.com

License: MIT (adjust as needed)
