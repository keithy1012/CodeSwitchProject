# !pip install datasets scikit-learn torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import re
# ----------------------------------------------------------
# 1. Load and preprocess  dataset (only transcription)
# ----------------------------------------------------------
def load_local_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # If tokens are saved as stringified lists, convert them back
    if isinstance(df.loc[0, "tokens"], str) and df.loc[0, "tokens"].startswith("["):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
    else:
        # Otherwise split by space if they’re space-separated
        df["tokens"] = df["tokens"].apply(lambda x: x.split() if isinstance(x, str) else x)

    return df

# --------------------------------------------------------
# 2. Create switch prediction labels
# --------------------------------------------------------
def create_switch_prediction_labels(df):
    data = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        # Heuristic: label each token as English or Chinese
        labels = []
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]", token):
                labels.append("zh")
            elif re.fullmatch(r"[a-zA-Z]+", token):
                labels.append("en")
            else:
                labels.append("other")

        switch_labels = [
            1 if labels[i] != labels[i + 1] else 0 for i in range(len(labels) - 1)
        ]
        data.append({"tokens": tokens, "labels": labels, "switch_labels": switch_labels})
    return data

# --------------------------------------------------------
# 3. Prepare token-level context data for ML model
# --------------------------------------------------------
def flatten_examples(data):
    contexts, labels = [], []
    for ex in data:
        tokens = ex["tokens"]
        switch_labels = ex["switch_labels"]
        for i in range(len(switch_labels)):
            context = " ".join(tokens[: i + 1])  # cumulative context
            contexts.append(context)
            labels.append(switch_labels[i])
    return contexts, labels

# --------------------------------------------------------
# 4. Train and evaluate Logistic Regression model
# --------------------------------------------------------
def train_logreg_model(csv_path):
    print("Loading our data...")
    data = load_local_data(csv_path)
    data = create_switch_prediction_labels(data)
    contexts, labels = flatten_examples(data)

    print(f"Total context-label pairs: {len(contexts)}")

    # TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(contexts)
    y = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)

    print("\n===== Logistic Regression Results =====")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["no_switch", "switch"]))
    # Return both classifier and vectorizer so callers can reuse them
    return clf, vectorizer


if __name__ == "__main__":
    clf, vectorizer = train_logreg_model("data/processed/processed_dataset.csv")
