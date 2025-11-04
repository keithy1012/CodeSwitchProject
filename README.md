# CodeSwitchProject

## Project Overview

This repository contains an experimental pipeline for studying code-switching in English/Mandarin dialogues within tourist contexts in China. The project analyzes bilingual conversations involving travel logistics, local food, directions, transportation, and cultural etiquette.

**Team:** Keith Yao (Project Lead), Arav Goyal (Data Engineer)

## Language Pair & Context

**Language Pair:** English ↔ Mandarin (Simplified Chinese)

**Justification:**
Mandarin is the primary language spoken in China, while English is widely used by international tourists. China is a growing tourist destination with diverse attractions including tech hotspots and natural landmarks. Additionally, China's growing influence in technology, energy, and finance makes this language pair particularly relevant for cross-cultural communication.

**Typical Code-Switching Contexts:**

- Tourist attractions and travel logistics
- Local food, menus, and dining experiences
- Directions and transportation
- Cultural etiquette and polite expressions (greetings, customs)
- Accommodation and safety tips

The dialogues include both **intersentential** switching (entire sentences in one language) and **intrasentential** switching (Mandarin phrases embedded within English sentences).

## Dataset Design

**Data Scope & Structure:**
The dataset consists of 200–300 conversations, with each conversation containing 10–15 utterances. The content focuses on domains relevant to tourists, including:

- Travel logistics (transport and accommodation)
- Local food and drinks
- Safety tips and cultural practices

Each token is labeled with its language (EN or CH), and speaker turns are annotated to distinguish between User and Agent.

**Data Generation Methodology:**
The dataset is generated using a combination of:

- **ChatGPT API** with prompt engineering (adopting a bilingual tour guide persona)
- **Kaggle** resources
- **Hugging Face ASCEND dataset** (https://huggingface.co/datasets/CAiRE/ASCEND)

Prompt engineering includes few-shot examples for each domain and explicitly encourages natural code-switching with Mandarin phrases embedded within English sentences where appropriate. Language balance is enforced by limiting the percentage of Mandarin within English sentences.

**Quality Assurance:**

- Manual review by bilingual speakers (fluent in both Mandarin and English)
- Consistency checks on language tags and code-switching patterns
- Verification of tourist locations, local foods, and cultural practices
- Cultural accuracy spot checks

## Pipeline Overview

The pipeline performs the following steps:

1. Load raw JSON dialogues from `data/raw/code_switch_data.json`
2. Normalize JSON into tabular form and extract linguistic features (tokens and POS tags) using spaCy (`en_core_web_sm`)
3. Build token-level dataset with cumulative contexts and labels indicating language switches at token boundaries
4. Train a TF-IDF + Logistic Regression model to predict token-level switch points

This baseline model provides a lightweight, interpretable approach to detecting switch points based on cumulative token context.

## Repository Layout

- `data/raw/` - Raw JSON data sources (`code_switch_data.json`, plus generated dialogue files)
- `data/processed/` - Processed dataset CSV (`processed_dataset.csv`) produced by the pipeline, with a small README
- `scripts/` - Main scripts and utilities:
  - `main.py` - End-to-end pipeline: loads JSON → extracts tokens/POS → saves processed CSV → trains/applies logistic regression model and prints summary statistics
  - `logistic_regression.py` - Prepares token-level context-label pairs, trains TF-IDF + LogisticRegression model and prints evaluation metrics. Exposes `train_logreg_model(csv_path)` which returns `(clf, vectorizer)`
  - `pos_extraction.py` - Utilities for tokenization and POS tagging using spaCy (provides `analyze_text_en(utterance, turn_id)`)
  - `random_forest.py`, `utils.py` - Additional helper scripts for related experiments (see docstrings for details)

## Requirements and Environment

**Recommended:** Python 3.8 or newer. Create a virtual environment and install dependencies.

**PowerShell (Windows) Quick Setup:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If you have a requirements.txt, use it. If not, install likely dependencies used by the scripts:
pip install pandas scikit-learn spacy joblib
# Download spaCy English model (required by pos_extraction.py and main.py):
python -m spacy download en_core_web_sm
```

**Notes / Assumptions:**

- The repository does not include a `requirements.txt` by default. If you prefer reproducibility, create one (e.g., `pip freeze > requirements.txt`) after installing packages.
- The minimal set of packages the scripts use are: `pandas`, `spacy`, `scikit-learn` (and `numpy`, which is usually pulled in by those packages).

## Running the Pipeline (Quick Start)

From the project root folder run (PowerShell):

```powershell
# Activate your virtual environment first (see setup above)
python scripts/main.py
```

**What `main.py` does:**

- Loads `data/raw/code_switch_data.json`
- Normalizes it into a DataFrame and writes `data/processed/processed_dataset.csv`
- Uses `pos_extraction.analyze_text_en(...)` to produce `tokens` and `pos_tags` columns in the CSV
- Trains a logistic regression switch-prediction model by calling `train_logreg_model("data/processed/processed_dataset.csv")` from `logistic_regression.py` (the model uses TF-IDF on cumulative token context)
- Applies the trained model to the dataset and prints dataset-level summary stats (predicted mixed-utterance rate, distinct tokens, etc.)

**To train/evaluate the logistic regression model separately:**

```powershell
python scripts/logistic_regression.py
```

This will run `train_logreg_model("data/processed/processed_dataset.csv")` and print evaluation metrics (accuracy, precision/recall/F1, ROC-AUC).

## Files of Interest

- `data/raw/code_switch_data.json` - Source dialogues (edit or expand to add more data)
- `data/processed/processed_dataset.csv` - Produced by `scripts/main.py`. Contains columns such as `id`, `utterance`, `tokens`, `pos_tags`, `labels` (heuristic language labels), and model predictions like `predicted_switch_probs` and `predicted_switches`
- `scripts/pos_extraction.py` - Tokenization + POS tagging utilities. `main.py` imports `analyze_text_en` from here
- `scripts/logistic_regression.py` - Data conversion and model training logic. Returns `clf, vectorizer` so the model can be reused in other scripts

## Evaluation Plan & Metrics

The evaluation focuses on several key criteria:

**Language Balance:**

- Percentage of English versus Mandarin used in conversations
- Natural and proportional code-switching patterns

**Code-Switch Quality:**

- Frequency and naturalness of code-switching
- Appropriate embedding of Mandarin phrases within English sentences and vice versa

**Model Performance:**

- Accuracy, precision/recall/F1 scores
- ROC-AUC for switch prediction
- Predicted mixed-utterance rate
- Distinct token counts

**Domain Coverage:**

- Completeness and relevance of responses for travel logistics, local food and drinks, and cultural practices
- Realistic and informative interactions

**Validation:**

- Manual review of random sample conversations by bilingual speakers
- Coherence of LLM-generated output
- Accuracy of domain-specific information

## Ethical & Responsible AI Considerations

**Risks:**

- Bias or stereotyping of cultural practices
- Incorrect translations leading to miscommunication
- Potential leakage of personal data in prompts

**Mitigation Strategies:**

- Content filters for sensitive content
- Cultural sensitivity checks by native speakers
- Verification of all tourist locations, local foods, and cultural practices
- Anonymization of any real user inputs
- Consistency checks using seed prompts and deterministic LLM responses

## Troubleshooting

- **spaCy model not found error:** Run `python -m spacy download en_core_web_sm` while your virtual environment is active
- **File not found errors:** Ensure you're running commands from the project root (the directory that contains `scripts/` and `data/`)
- **Memory / performance:** The TF-IDF vectorizer is configured with `max_features=5000` in `logistic_regression.py`. For very large datasets, adjust or stream data
