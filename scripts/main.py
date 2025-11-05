import pandas as pd
from pathlib import Path
from logistic_regression import train_logreg_model
from random_forest import train_random_forest_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils import count_en_zh_tokens, create_readme, countTokens, countUtterances, compute_mixed_utterance_rate, detect_token_languages, addPOSandToken, load_data

# Load JSON and convert to CSV
csv_path, processed_dir, data = load_data()
df = pd.json_normalize(data)

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

# Train or load logistic regression model
print("Training logistic regression switch model ...")
lg_clf, lg_vectorizer = train_logreg_model(processed_dir / "processed_dataset.csv")

# Train or load random forest model
print("Training random forest switch model ...")
rf_clf, rf_vectorizer = train_random_forest_model(processed_dir / "processed_dataset.csv")

# Apply logistic regression model
lg_df = apply_switch_model_to_df(df, lg_clf, lg_vectorizer, threshold=0.5)

# Apply random forest model
rf_df = apply_switch_model_to_df(df, rf_clf, rf_vectorizer, threshold=0.5)

# Summary statistics
total_pred_mixed = df['predicted_mixed_pred'].sum()
predicted_mixed_rate = total_pred_mixed / len(df) if len(df) > 0 else 0.0
print(f"Predicted mixed-utterance rate (model): {predicted_mixed_rate*100:.2f}% ({total_pred_mixed}/{len(df)})")

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

print("Saved switch-point histogram")

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
print("Saved utterance length distribution")

# --- 4. Optional: Boxplot of predicted switch probabilities ---
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
