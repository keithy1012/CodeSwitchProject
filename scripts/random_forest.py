# !pip install datasets scikit-learn torch

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import numpy as np
import re
import torch
from typing import List, Dict


# --------------------------------------------------------
# 1. Load and preprocess ASCEND dataset (only transcription)
# --------------------------------------------------------
def load_ascend_data(debug=False, num_samples=None):
    if debug:
        data = [
            {"tokens": ["你好", "hello", "world", "再见"], "labels": ["zh", "en", "en", "zh"]},
            {"tokens": ["This", "is", "a", "test"], "labels": ["en", "en", "en", "en"]},
            {"tokens": ["测试", "is", "done"], "labels": ["zh", "en", "en"]},
        ]
        return data

    dataset = load_dataset("CAiRE/ASCEND", split="train")

    # Tokenize transcription into words and label each token as en/zh
    def tokenize_and_label(text):
        pattern = re.compile(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+|[^\w\s]')
        tokens = pattern.findall(text)
        labels = []
        for token in tokens:
            if re.fullmatch(r'[\u4e00-\u9fff]', token):
                labels.append('zh')
            elif re.fullmatch(r'[a-zA-Z]+', token):
                labels.append('en')
            else:
                labels.append('other')
        return tokens, labels

    data = []
    for ex in dataset:
        tokens, labels = tokenize_and_label(ex['transcription'])
        if len(tokens) > 1:
            data.append({'tokens': tokens, 'labels': labels})
    if num_samples:
        data = data[:num_samples]
    return data


# --------------------------------------------------------
# 2. Create switch prediction labels
# --------------------------------------------------------
def create_switch_prediction_labels(data: List[Dict]) -> List[Dict]:
    for example in data:
        switch_labels = []
        for i in range(len(example['labels']) - 1):
            switch_labels.append(1 if example['labels'][i] != example['labels'][i + 1] else 0)
        example["switch_labels"] = switch_labels
    return data


# --------------------------------------------------------
# 3. Prepare token-level context data for ML model
# --------------------------------------------------------
def flatten_examples(data: List[Dict]):
    contexts, labels = [], []
    for ex in data:
        tokens = ex['tokens']
        switch_labels = ex['switch_labels']
        for i in range(len(switch_labels)):
            context = " ".join(tokens[:i+1])  # context up to current token
            contexts.append(context)
            labels.append(switch_labels[i])
    return contexts, labels


# --------------------------------------------------------
# 4. Train and evaluate Random Forest model
# --------------------------------------------------------
def train_random_forest_model(debug=False):
    print("Loading ASCEND data...")
    data = load_ascend_data(debug=debug, num_samples=200 if debug else 10000)
    data = create_switch_prediction_labels(data)
    contexts, labels = flatten_examples(data)

    print(f"Total context-label pairs: {len(contexts)}")

    # TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(contexts)
    y = np.array(labels)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)

    print("\n===== Random Forest Results =====")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["no_switch", "switch"]))


if __name__ == "__main__":
    train_random_forest_model(debug=True)
