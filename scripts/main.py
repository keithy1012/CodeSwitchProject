import json
import pandas as pd
import spacy
from pathlib import Path
from datetime import date
import re
from logistic_regression import train_logreg_model
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', 500)
# Load spaCy English model
# Install first: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Step 4: Load JSON and convert to CSV
json_path = 'data/raw/code_switch_data.json'
csv_path = f'data/processed/{date.today().strftime("%Y-%m-%d")}/processed_dataset.csv'
processed_dir = Path(f'data/processed/{date.today().strftime("%Y-%m-%d")}')
processed_dir.mkdir(parents=True, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Step 5: Add initial linguistic features (tokens + POS) using pos_extraction utilities
import pos_extraction

tokens_list = []
pos_tags_list = []

for i, utterance in enumerate(df['utterance']):
    try:
        analysis = pos_extraction.analyze_text_en(utterance, i+1)
        # analysis is a list of dicts with keys: 'Token', 'Lemma', 'POS Tag', etc.
        tokens = [item.get('Token') for item in analysis]
        pos_tags = [item.get('POS Tag') for item in analysis]
    except Exception as e:
        print(f"Warning: POS extraction failed for row {i} (id={df.loc[i,'id']}): {e}")
        tokens = []
        pos_tags = []

    tokens_list.append(tokens)
    pos_tags_list.append(pos_tags)

df['tokens'] = tokens_list
df['pos_tags'] = pos_tags_list

# Step 6: Save to CSV
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"Processed CSV saved to: {csv_path}")

# Step 7: Verify a few examples manually
#print(df[['id', 'utterance', 'tokens', 'pos_tags']].head(5))

# Step 8: Create README
Path(processed_dir).mkdir(parents=True, exist_ok=True)

readme_path = processed_dir / 'README.md'
with open(readme_path, 'w', encoding='utf-8') as f:
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

print(f"README.md saved to: {readme_path}")

# Count totals
total_tokens = []
total_distinct_tokens = set()
for tokens in df["tokens"]:
    for token in tokens:
        total_tokens.append(token)
        total_distinct_tokens.add(token)
print(f"Total number of tokens in dataset: {len(total_tokens)}")
print(f"Total number of distinct tokens in dataset: {len(total_distinct_tokens)}")
# Count total utterances
total_utt = 0
for dialogue in df["utterance"]:
    count_turns = dialogue.count("\n") + 1
    total_utt += count_turns
print(f"Total utterances in dataset: {total_utt}")

# Create `labels` column by simple script-based language detection (Chinese vs English)
def _detect_langs(text):
    """Return a list of language labels detected in the text. Currently detects 'zh' (CJK) and 'en' (Latin letters)."""
    if not isinstance(text, str):
        return []
    langs = set()
    if re.search(r"[\u4e00-\u9fff]", text):
        langs.add('zh')
    if re.search(r"[A-Za-z]", text):
        langs.add('en')
    return list(langs)

df['labels'] = df['utterance'].apply(_detect_langs)
# Mixed utterance rate
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

compute_mixed_utterance_rate(df)

# Apply trained logistic regression switch-prediction model to our dataset
def apply_switch_model_to_df(df, clf, vectorizer, threshold=0.5):
    """Apply a token-level switch predictor to the DataFrame.

    Produces three new columns on the DataFrame:
    - predicted_switch_probs: list of probabilities for switch at each token boundary (len = n_tokens-1)
    - predicted_switches: list of binary predictions (0/1) using threshold
    - predicted_mixed_pred: boolean; True if any predicted switch==1 in the dialogue
    """
    probs_list = []
    preds_list = []
    mixed_pred_list = []

    for tokens in df['tokens']:
        if not isinstance(tokens, list) or len(tokens) < 2:
            probs_list.append([])
            preds_list.append([])
            mixed_pred_list.append(False)
            continue
        contexts = [" ".join(tokens[:i+1]) for i in range(len(tokens)-1)]
        X = vectorizer.transform(contexts)
        probs = clf.predict_proba(X)[:, 1].tolist()
        preds = [1 if p >= threshold else 0 for p in probs]
        probs_list.append(probs)
        preds_list.append(preds)
        mixed_pred_list.append(any(preds))

    df['predicted_switch_probs'] = probs_list
    df['predicted_switches'] = preds_list
    df['predicted_mixed_pred'] = mixed_pred_list
    return df


# Train or load model
print("Training logistic regression switch model ...")
clf, vectorizer = train_logreg_model(processed_dir / "processed_dataset.csv")

# Apply model
df = apply_switch_model_to_df(df, clf, vectorizer, threshold=0.5)
# Summary statistics
total_pred_mixed = df['predicted_mixed_pred'].sum()
predicted_mixed_rate = total_pred_mixed / len(df) if len(df) > 0 else 0.0
print(f"Predicted mixed-utterance rate (model): {predicted_mixed_rate*100:.2f}% ({total_pred_mixed}/{len(df)})")


# VISUALIZATIONS

# Create figures directory
fig_dir = processed_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Language proportion chart ---
lang_counts = {'en_only': 0, 'zh_only': 0, 'mixed': 0}
for langs in df['labels']:
    if not langs:
        continue
    lset = set(langs)
    if 'en' in lset and 'zh' in lset:
        lang_counts['mixed'] += 1
    elif 'en' in lset:
        lang_counts['en_only'] += 1
    elif 'zh' in lset:
        lang_counts['zh_only'] += 1

plt.figure(figsize=(6, 6))
plt.pie(lang_counts.values(), labels=lang_counts.keys(), autopct='%1.1f%%', colors=['#4C72B0', '#55A868', '#C44E52'])
plt.title("Language Composition of Utterances")
plt.savefig(fig_dir / "language_proportion_pie.png", dpi=300)
plt.close()

print(f"Saved language proportion pie chart to {fig_dir / 'language_proportion_pie.png'}")

# --- 2. Histogram of switch-point locations ---
switch_positions = []
for probs, preds in zip(df['predicted_switch_probs'], df['predicted_switches']):
    for i, pred in enumerate(preds):
        if pred == 1:
            switch_positions.append(i+1)  # token boundary index

plt.figure(figsize=(8, 5))
sns.histplot(switch_positions, bins=20, kde=False, color='#C44E52')
plt.title("Distribution of Predicted Switch Positions (Token Index)")
plt.xlabel("Token Index of Switch")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(fig_dir / "switch_point_histogram.png", dpi=300)
plt.close()

print(f"Saved switch-point histogram to {fig_dir / 'switch_point_histogram.png'}")

# --- 3. Distribution of utterance lengths ---
df['utterance_length'] = df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0)
plt.figure(figsize=(8, 5))
sns.histplot(df['utterance_length'], bins=30, kde=True, color='#4C72B0')
plt.title("Distribution of Utterance Lengths")
plt.xlabel("Number of Tokens")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(fig_dir / "utterance_length_distribution.png", dpi=300)
plt.close()

print(f"Saved utterance length distribution to {fig_dir / 'utterance_length_distribution.png'}")

# --- 4. Optional: Boxplot of predicted switch probabilities ---
all_probs = [p for probs in df['predicted_switch_probs'] for p in probs]
plt.figure(figsize=(6, 4))
sns.boxplot(x=all_probs, color='#55A868')
plt.title("Distribution of Predicted Switch Probabilities")
plt.xlabel("Switch Probability")
plt.tight_layout()
plt.savefig(fig_dir / "switch_prob_boxplot.png", dpi=300)
plt.close()

print(f"Saved predicted switch probability boxplot to {fig_dir / 'switch_prob_boxplot.png'}")

print(f"All visualizations saved in {fig_dir}")

# --- 5. Token-level language proportion pie chart ---
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

total_en = 0
total_zh = 0

for tokens in df['tokens']:
    en_c, zh_c = count_en_zh_tokens(tokens)
    total_en += en_c
    total_zh += zh_c

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie([total_en, total_zh],
        labels=['English Tokens', 'Chinese Tokens'],
        autopct='%1.1f%%',
        colors=['#4C72B0', '#C44E52'])
plt.title("Token-Level Language Proportion (English vs Chinese)")
plt.savefig(fig_dir / "token_language_pie.png", dpi=300)
plt.close()

print(f"Saved token-level language proportion pie chart to {fig_dir / 'token_language_pie.png'}")
