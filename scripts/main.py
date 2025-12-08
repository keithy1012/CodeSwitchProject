import pandas as pd
from pathlib import Path
from logistic_regression import train_logreg_model
from random_forest import train_random_forest_model
from bilstm_model import train_bilstm_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils import count_en_zh_tokens, create_readme, countTokens, countUtterances, compute_mixed_utterance_rate, detect_token_languages, addPOSandToken, load_data, detect_switch_type, classify_lang_mix
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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

# Train BiLSTM model
print("Training BiLSTM switch model ...")
bilstm_clf, bilstm_vectorizer = train_bilstm_model(processed_dir / "processed_dataset.csv",epochs=15)

# Apply logistic regression model
lg_df = apply_switch_model_to_df(df, lg_clf, lg_vectorizer, threshold=0.5)
lg_df.to_csv(processed_dir/"lg_df.csv")
# Apply random forest model
rf_df = apply_switch_model_to_df(df, rf_clf, rf_vectorizer, threshold=0.5)
rf_df.to_csv(processed_dir/"rf_df.csv")
# Apply BiLSTM model (tokenizer.transform returns padded arrays; wrapper.predict_proba accepts them)
bilstm_df = apply_switch_model_to_df(df, bilstm_clf, bilstm_vectorizer, threshold=0.5)
bilstm_df.to_csv(processed_dir/"bilstm_df.csv")

# Create figures directory
fig_dir = processed_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Language proportion chart ---
def language_proportion_by_utterance():
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
language_proportion_by_utterance()

# --- 2. Histogram of switch-point locations ---
def switch_point_histogram():
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

switch_point_histogram()
# --- 3. Distribution of utterance lengths ----
def utterance_length_distribution():
    df['utterance_length'] = df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0)

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
utterance_length_distribution()
# --- 4. Boxplot of predicted switch probabilities ---
def predicted_switch_probability_boxplot():
    all_probs = [p for probs in df['predicted_switch_probs'] for p in probs]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=all_probs, color='#55A868')
    plt.title("Distribution of Predicted Switch Probabilities")
    plt.xlabel("Switch Probability")
    plt.tight_layout()
    plt.savefig(fig_dir / "switch_prob_boxplot.png", dpi=300)
    plt.close()
    print("Saved predicted switch probability boxplot")
predicted_switch_probability_boxplot()
# --- 5. Token-level language proportion pie chart ---
def language_proportion_by_token():
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
language_proportion_by_token()

# --- 6. Count of Code Switching ---
def code_switch_type():
    new_df = detect_switch_type(df)
    plt.figure(figsize=(6,4))
    sns.countplot(data=new_df, x='switch_type', palette='Set2')
    plt.title("Distribution of Code-Switching Types")
    plt.xlabel("Switch Type")
    plt.ylabel("Count")
    plt.savefig(fig_dir/"distribution_code_switching_types.png", dpi=300)
    plt.close()
    print("Saved code-switching type distribution chart")
code_switch_type()

# Utility: safe sum of switches per utterance
df['num_switches'] = df['actual_switches'].apply(lambda lst: int(np.sum(lst)) if isinstance(lst, (list, np.ndarray)) else 0)
df['utterance_length'] = df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['utterance_type'] = df['Language'].apply(classify_lang_mix)

# 1) Dataset Class Balance (utterance-level and token-level)
def dataset_class_balance():
    # utterance-level
    ut_counts = {'en_only': 0, 'zh_only': 0, 'mixed': 0}
    for langs in df['labels']:
        if not langs:
            continue
        lset = set(langs)
        if 'en' in lset and 'zh' in lset:
            ut_counts['mixed'] += 1
        elif 'en' in lset:
            ut_counts['en_only'] += 1
        elif 'zh' in lset:
            ut_counts['zh_only'] += 1

    plt.figure(figsize=(7,5))
    sns.barplot(x=list(ut_counts.keys()), y=list(ut_counts.values()), palette=['#4C72B0','#55A868','#C44E52'])
    plt.title("Utterance-Level Class Balance")
    plt.ylabel("Number of Utterances")
    plt.xlabel("Utterance Type")
    plt.tight_layout()
    plt.savefig(fig_dir/"dataset_class_balance_utterance.png", dpi=300)
    plt.close()
    print("Saved utterance-level class balance chart")

    # token-level: en tokens, zh tokens, switch-boundaries (as a proxy for code-switch tokens)
    total_en = total_zh = total_switch_boundaries = 0
    for langs, switches in zip(df['Language'], df['actual_switches']):
        if isinstance(langs, list):
            total_en += sum(1 for l in langs if l == 'en')
            total_zh += sum(1 for l in langs if l == 'zh')
        if isinstance(switches, list):
            total_switch_boundaries += sum(int(x) for x in switches)
    plt.figure(figsize=(6,6))
    sns.barplot(x=['English tokens','Chinese tokens','Switch boundaries (token-level)'],
                y=[total_en, total_zh, total_switch_boundaries],
                palette=['#4C72B0','#C44E52','#C49442'])
    plt.xticks(rotation=15, ha='right')
    plt.title("Token-Level Class Balance (tokens & switch boundaries)")
    plt.tight_layout()
    plt.savefig(fig_dir/"dataset_class_balance_token.png", dpi=300)
    plt.close()
    print("Saved token-level class balance chart")

dataset_class_balance()

# 2) Code-Switching Frequency Distribution (number of code-switched tokens per utterance)
def code_switching_frequency_distribution():
    # bins: 0, 1-2, 3+
    bins = [0,1,3, np.max(df['num_switches'])+1]
    labels = ['0','1-2','3+']
    binned = pd.cut(df['num_switches'], bins=bins, right=False, labels=labels)
    freq = binned.value_counts().reindex(labels).fillna(0)

    plt.figure(figsize=(7,5))
    sns.barplot(x=freq.index, y=freq.values, palette=['#4C72B0','#55A868','#C44E52'])
    plt.xlabel("Number of Switches per Utterance")
    plt.ylabel("Utterance Count")
    plt.title("Code-Switching Frequency per Utterance")
    plt.tight_layout()
    plt.savefig(fig_dir/"code_switching_frequency_distribution.png", dpi=300)
    plt.close()
    print("Saved code-switching frequency distribution chart")

    # Also save a histogram of raw counts for reference
    plt.figure(figsize=(8,4))
    sns.histplot(df['num_switches'], bins=range(0, max(5, df['num_switches'].max()+2)), kde=False, color='#C44E52')
    plt.xlabel("Number of switch boundaries in utterance")
    plt.ylabel("Count")
    plt.title("Histogram: Raw Count of Switch Boundaries per Utterance")
    plt.tight_layout()
    plt.savefig(fig_dir/"code_switching_raw_histogram.png", dpi=300)
    plt.close()
    print("Saved raw histogram of switch counts")

code_switching_frequency_distribution()

# 3) Language Biases in Syntactic Features (compare POS frequencies)
def pos_language_biases(top_n=3):
    # We'll map spaCy and jieba tags into coarse categories: NOUN, VERB, ADJ
    def coarse_pos(tag):
        if not isinstance(tag, str):
            return None
        tag = tag.upper()
        # common spaCy tags: NOUN, VERB, ADJ; jieba: n (noun), v (verb), a (adj), nr/ns/nr etc
        if tag.startswith('N') or tag in ('NR','NS','NZ','N'):
            return 'NOUN'
        if tag.startswith('V') or tag in ('V','VD','VG'):
            return 'VERB'
        if tag.startswith('A') or tag in ('ADJ','A'):
            return 'ADJ'
        return 'OTHER'

    counts = {'en': {'NOUN':0,'VERB':0,'ADJ':0,'OTHER':0}, 'zh': {'NOUN':0,'VERB':0,'ADJ':0,'OTHER':0}}
    # iterate rows and aggregate POS per language token
    for langs, pos_tags in zip(df['Language'], df['pos_tags']):
        if not isinstance(langs, list) or not isinstance(pos_tags, list):
            continue
        for l, p in zip(langs, pos_tags):
            if l not in ('en','zh'):
                continue
            cp = coarse_pos(p)
            if cp:
                counts[l][cp] = counts[l].get(cp,0) + 1

    # Build DataFrame for plotting
    pos_df = pd.DataFrame(counts).T[['NOUN','VERB','ADJ','OTHER']]
    pos_df = pos_df.reset_index().melt(id_vars='index', var_name='POS', value_name='Count').rename(columns={'index':'Language'})

    plt.figure(figsize=(8,5))
    sns.barplot(data=pos_df, x='POS', y='Count', hue='Language', palette=['#4C72B0','#C44E52'])
    plt.title("POS Frequency by Language (coarse categories)")
    plt.ylabel("Token Count")
    plt.tight_layout()
    plt.savefig(fig_dir/"pos_language_biases.png", dpi=300)
    plt.close()
    print("Saved POS language bias chart")

pos_language_biases()

# 4) Sentence Length vs. Code-Switching (scatter + boxplot and correlation)
def sentence_length_vs_csi():
    # compute correlation (Pearson) - safe handling for constant arrays
    try:
        corr = df['utterance_length'].corr(df['num_switches'])
    except Exception:
        corr = None

    # scatter plot with jitter for clarity
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df.sample(frac=1.0, replace=False), x='utterance_length', y='num_switches', alpha=0.4)
    plt.xlabel("Utterance Length (tokens)")
    plt.ylabel("Number of Switch Boundaries")
    plt.title(f"Utterance Length vs. Code-Switching (corr={corr:.3f} )" if corr is not None else "Utterance Length vs. Code-Switching")
    plt.tight_layout()
    plt.savefig(fig_dir/"length_vs_switch_scatter.png", dpi=300)
    plt.close()
    print("Saved utterance length vs code-switching scatter")

    # boxplot: distribution of utterance lengths grouped by 0-switch vs >=1 switches
    df['has_switch'] = df['num_switches'].apply(lambda x: 'no_switch' if x==0 else 'has_switch')
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x='has_switch', y='utterance_length', palette=['#4C72B0','#C44E52'])
    plt.title("Utterance Length Distribution: no_switch vs has_switch")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fig_dir/"length_vs_switch_boxplot.png", dpi=300)
    plt.close()
    print("Saved boxplot of length vs code-switching")

sentence_length_vs_csi()

# Print quick stats to console for inclusion in analysis
print("Class balance (utterance-level):")
print(df['utterance_type'].value_counts().to_dict())
print("Token-level totals (en, zh, switch boundaries):")
total_en = sum(1 for langs in df['Language'] if isinstance(langs, list) for l in langs if l=='en')
# (printing approximate counts above may be noisy due to shape; user can compute exact numbers as needed)


def histogram_code_switches_per_utterance():
    """
    Plot a histogram showing how many switch boundaries each utterance has.
    """
    plt.figure(figsize=(8,5))
    sns.histplot(
        df['num_switches'],
        bins=range(0, df['num_switches'].max() + 2),
        kde=False,
        color="#4C72B0"
    )
    plt.xlabel("Number of Code-Switches in Utterance")
    plt.ylabel("Number of Utterances")
    plt.title("Histogram of Code-Switch Counts per Utterance")
    plt.tight_layout()
    plt.savefig(fig_dir / "hist_code_switches_per_utterance.png", dpi=300)
    plt.close()
    print("Saved histogram of code-switch counts per utterance")

histogram_code_switches_per_utterance()
