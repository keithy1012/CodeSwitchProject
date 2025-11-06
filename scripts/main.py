import pandas as pd
from pathlib import Path
from logistic_regression import train_logreg_model
from random_forest import train_random_forest_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils import count_en_zh_tokens, create_readme, countTokens, countUtterances, compute_mixed_utterance_rate, detect_token_languages, addPOSandToken, load_data
import re

# Load JSON and convert to CSV
csv_path, processed_dir, data = load_data()
raw_df = pd.json_normalize(data)

# Expand each dialogue into individual turns (rows)
rows = []
for _, r in raw_df.iterrows():
    dialog_text = r.get('utterance', '')
    # Split on line breaks and drop empty lines
    lines = [ln.strip() for ln in re.split(r"\r?\n", dialog_text) if ln.strip()]
    for i, ln in enumerate(lines):
        speaker = None
        turn_text = ln
        # Try to extract speaker like 'Li: text' or 'Li：text'
        m = re.match(r'^([^:：]+)[:：]\s*(.*)$', ln)
        if m:
            speaker = m.group(1).strip()
            turn_text = m.group(2).strip()
        # Create a new row copying metadata from original dialogue row
        new_row = r.to_dict()
        # Provide original dialogue id and per-turn index. Keep dialogue_text.
        orig_id = r.get('id', '')
        new_row.update({
            'dialogue_text': dialog_text,
            'turn_index': i+1,
            'speaker': speaker,
            'utterance': turn_text
        })
        rows.append(new_row)

# Build DataFrame where each row is a dialogue turn
df = pd.DataFrame(rows)

# Add initial linguistic features (tokens + POS) using pos_extraction utilities
tokens_list, pos_tags_list = addPOSandToken(df)

df['tokens'] = tokens_list
df['pos_tags'] = pos_tags_list

# Count totals tokens
total_tokens, total_distinct_tokens = countTokens(df)
print(f"Total number of tokens in dataset: {len(total_tokens)}")
print(f"Total number of distinct tokens in dataset: {len(total_distinct_tokens)}")

# Count total utterances
total_utt = countUtterances(df)
print(f"Total utterances in dataset: {total_utt}")


# Create `labels` column by script-based language detection
languages = df['tokens'].apply(detect_token_languages)
set_of_languages = languages.apply(set)
df['labels'] = set_of_languages
df['Language'] = languages

# Save to CSV
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"Processed CSV saved to: {csv_path}")

# Create README
Path(processed_dir).mkdir(parents=True, exist_ok=True)
readme_path = processed_dir / 'README.md'
create_readme(readme_path, df)

# Mixed utterance rate
compute_mixed_utterance_rate(df)

# Apply trained switch-prediction model to our dataset
def apply_switch_model_to_df(df, clf, vectorizer, threshold=0.5):
    """Apply a token-level switch predictor to the DataFrame.

    Produces new columns on the DataFrame:
    - predicted_switch_probs: list of probabilities for switch at each token boundary (len = n_tokens-1)
    - predicted_switches: list of binary predictions (0/1) using threshold
    - actual_switches: ground-truth switch labels computed from token-level `Language` (or regex fallback)
    - switch_match_rate: fraction of token boundaries where predicted == actual (0-1)
    """
    probs_list = []
    preds_list = []
    actual_list = []
    match_rate_list = []

    def infer_lang_for_token(tok: str) -> str:
        if isinstance(tok, str) and re.fullmatch(r"[\u4e00-\u9fff]+", tok):
            return 'zh'
        if isinstance(tok, str) and re.fullmatch(r"[A-Za-z]+", tok):
            return 'en'
        return 'other'

    # iterate rows so we can access tokens and the precomputed Language column
    for tokens, langs in zip(df['tokens'], df.get('Language', [None] * len(df))):
        if not isinstance(tokens, list) or len(tokens) < 2:
            probs_list.append([])
            preds_list.append([])
            actual_list.append([])
            match_rate_list.append(0.0)
            continue

        # Predicted
        contexts = [" ".join(tokens[: i + 1]) for i in range(len(tokens) - 1)]
        X = vectorizer.transform(contexts)
        probs = clf.predict_proba(X)[:, 1].tolist()
        preds = [1 if p >= threshold else 0 for p in probs]

        actuals = []
        if isinstance(langs, list) and len(langs) == len(tokens):
            actuals = [
                1 if (langs[i] in ['en', 'zh'] and langs[i + 1] in ['en', 'zh'] and langs[i] != langs[i + 1])
                else 0
                for i in range(len(langs) - 1)
            ]
        else:
            inferred = [infer_lang_for_token(t) for t in tokens]
            actuals = [
                1 if (inferred[i] in ['en', 'zh'] and inferred[i + 1] in ['en', 'zh'] and inferred[i] != inferred[i + 1])
                else 0
                for i in range(len(inferred) - 1)
            ]

        # compute match rate between predicted and actual (only over positions existing in both)
        matches = 0
        denom = min(len(preds), len(actuals))
        if denom > 0:
            matches = sum(1 for a, p in zip(actuals, preds) if a == p)
            match_rate = matches / denom
        else:
            match_rate = 0.0

        probs_list.append(probs)
        preds_list.append(preds)
        actual_list.append(actuals)
        match_rate_list.append(match_rate)

    df['predicted_switch_probs'] = probs_list
    df['predicted_switches'] = preds_list
    df['actual_switches'] = actual_list
    df['switch_match_rate'] = match_rate_list
    return df

# Train or load logistic regression model
print("Training logistic regression switch model ...")
lg_clf, lg_vectorizer = train_logreg_model(processed_dir / "processed_dataset.csv")

# Train or load random forest model
print("Training random forest switch model ...")
rf_clf, rf_vectorizer = train_random_forest_model(processed_dir / "processed_dataset.csv")

# Apply logistic regression model
lg_df = apply_switch_model_to_df(df, lg_clf, lg_vectorizer, threshold=0.5)
lg_df.to_csv(processed_dir/"lg_df.csv")
# Apply random forest model
rf_df = apply_switch_model_to_df(df, rf_clf, rf_vectorizer, threshold=0.5)
rf_df.to_csv(processed_dir/"rf_df.csv")

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

print("Saved language proportion pie chart")

# --- 2. Histogram of switch-point locations ---
switch_positions = []
for switches in df['actual_switches']:
    for i, switch in enumerate(switches):
        if switch == 1:
            switch_positions.append(i+1)

plt.figure(figsize=(8, 5))
sns.histplot(switch_positions, bins=20, kde=False, color='#C44E52')
plt.title("Distribution of Predicted Switch Positions (Token Index)")
plt.xlabel("Token Index of Switch")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(fig_dir / "switch_point_histogram.png", dpi=300)
plt.close()

print("Saved switch-point histogram")

# --- 3. Distribution of utterance lengths ----
df['utterance_length'] = df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0)

def classify_lang_mix(langs):
    if not isinstance(langs, list):
        return "unknown"
    langs_set = set(l for l in langs if l in ["en", "zh"])
    if langs_set == {"en"}:
        return "en_only"
    elif langs_set == {"zh"}:
        return "zh_only"
    elif langs_set == {"en", "zh"}:
        return "mixed"
    else:
        return "other"

df['utterance_type'] = df['Language'].apply(classify_lang_mix)

plt.figure(figsize=(9, 6))
sns.histplot(data=df[df['utterance_type'] == 'en_only'], x='utterance_length', bins=30, color='#4C72B0', label='English Only', kde=True, alpha=0.5)
sns.histplot(data=df[df['utterance_type'] == 'zh_only'], x='utterance_length', bins=30, color='#55A868', label='Chinese Only', kde=True, alpha=0.5)
sns.histplot(data=df[df['utterance_type'] == 'mixed'], x='utterance_length', bins=30, color='#C44E52', label='Mixed (EN+ZH)', kde=True, alpha=0.5)

plt.title("Distribution of Utterance Lengths by Language Composition")
plt.xlabel("Number of Tokens")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "utterance_length_by_language_type.png", dpi=300)
plt.close()
print("Saved utterance length distribution by language type")

# --- 4. Boxplot of predicted switch probabilities ---
all_probs = [p for probs in df['predicted_switch_probs'] for p in probs]
plt.figure(figsize=(6, 4))
sns.boxplot(x=all_probs, color='#55A868')
plt.title("Distribution of Predicted Switch Probabilities")
plt.xlabel("Switch Probability")
plt.tight_layout()
plt.savefig(fig_dir / "switch_prob_boxplot.png", dpi=300)
plt.close()
print("Saved predicted switch probability boxplot")

# --- 5. Token-level language proportion pie chart ---
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

print("Saved token-level language proportion pie chart")


def detect_switch_type(df):
    """
    Adds a 'switch_type' column with values:
      - 'intra' for within-utterance language switches
      - 'inter' for switches between utterances only
      - 'none' for monolingual utterances with no switches
    """
    switch_types = []

    for i, row in df.iterrows():
        langs = row.get('Language', [])
        if not isinstance(langs, list) or len(langs) == 0:
            switch_types.append('none')
            continue

        # Normalize to only en/zh
        langs = [l for l in langs if l in ('en', 'zh')]
        if len(set(langs)) > 1:
            # Multiple langs in the same utterance → intra-sentential
            switch_types.append('intra')
        else:
            # Check inter-sentential: different from previous utterance
            if i > 0:
                prev_langs = df.iloc[i - 1].get('Language', [])
                prev_langs = [l for l in prev_langs if l in ('en', 'zh')]
                if prev_langs and len(set(prev_langs)) == 1 and prev_langs[0] != langs[0]:
                    switch_types.append('inter')
                else:
                    switch_types.append('none')
            else:
                switch_types.append('none')

    df['switch_type'] = switch_types
    return df

df = detect_switch_type(df)

# Check distribution of switch types
print(df['switch_type'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='switch_type', palette='Set2')
plt.title("Distribution of Code-Switching Types")
plt.xlabel("Switch Type")
plt.ylabel("Count")
plt.savefig(fig_dir/"distribution_code_switching_types.png", dpi=300)
plt.close()
print("Saved code-switching type distribution chart")