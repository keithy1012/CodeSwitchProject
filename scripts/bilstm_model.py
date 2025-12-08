import ast
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Simple tokenizer that builds vocab from context strings (space-separated tokens)
class SimpleTokenizer:
    def __init__(self, oov_token="<OOV>", pad_token="<PAD>"):
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.word_index = {pad_token: 0, oov_token: 1}
        self.index_word = {0: pad_token, 1: oov_token}
        self.vocab_size = 2

    def fit_on_texts(self, texts):
        for txt in texts:
            for w in str(txt).split():
                if w not in self.word_index:
                    idx = len(self.word_index)
                    self.word_index[w] = idx
                    self.index_word[idx] = w
        self.vocab_size = len(self.word_index)

    def texts_to_sequences(self, texts):
        seqs = []
        for txt in texts:
            seqs.append([self.word_index.get(w, self.word_index[self.oov_token]) for w in str(txt).split()])
        return seqs

    def transform(self, texts, maxlen=None):
        seqs = self.texts_to_sequences(texts)
        if maxlen is None:
            maxlen = max((len(s) for s in seqs), default=0)
        arr = np.full((len(seqs), maxlen), fill_value=self.word_index[self.pad_token], dtype=np.int64)
        for i, s in enumerate(seqs):
            L = min(len(s), maxlen)
            arr[i, :L] = s[:L]
        return arr

# PyTorch dataset wrapper
class SeqDataset(Dataset):
    def __init__(self, X_array, y_array):
        self.X = torch.from_numpy(X_array).long()
        self.y = torch.from_numpy(y_array).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# BiLSTM model with improved architecture
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        
        # Add dropout and attention to final layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        
        # Use mean pooling + last hidden state (better than just last token)
        last = out[:, -1, :]
        mean = out.mean(dim=1)
        combined = last + mean  # Combine both representations
        combined = self.dropout(combined)
        
        logits = self.fc(combined).squeeze(-1)
        return logits

# Wrapper to expose scikit-like predict_proba
class TorchClassifierWrapper:
    def __init__(self, model, device='cpu', batch_size=256):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

    def predict_proba(self, X_array):
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(np.array(X_array)).long().to(self.device)
            probs = []
            for i in range(0, X.size(0), self.batch_size):
                batch = X[i:i+self.batch_size]
                logits = self.model(batch)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
            probs = np.concatenate(probs, axis=0)
            # return shape (n_samples, 2) like sklearn: prob for class 0 and 1
            probs = np.stack([1 - probs, probs], axis=1)
            return probs

# Data helpers
def load_local_data(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return df
    if isinstance(df.loc[0, "tokens"], str) and df.loc[0, "tokens"].startswith("["):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
    else:
        df["tokens"] = df["tokens"].apply(lambda x: x.split() if isinstance(x, str) else x)
    return df

def create_switch_prediction_labels(df):
    data = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        labels = []
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]", token):
                labels.append("zh")
            elif re.fullmatch(r"[a-zA-Z]+", token):
                labels.append("en")
            else:
                labels.append("other")
        switch_labels = [1 if labels[i] != labels[i + 1] else 0 for i in range(len(labels) - 1)]
        data.append({"tokens": tokens, "labels": labels, "switch_labels": switch_labels})
    return data

def flatten_examples(data):
    contexts, labels = [], []
    for ex in data:
        tokens = ex["tokens"]
        switch_labels = ex["switch_labels"]
        for i in range(len(switch_labels)):
            context = " ".join(tokens[: i + 1])
            contexts.append(context)
            labels.append(switch_labels[i])
    return contexts, np.array(labels, dtype=np.int64)

# ====== CONFUSION MATRIX & HEATMAP ======
def plot_confusion_matrix_heatmap(y_true, y_pred, model_name="BiLSTM", save_dir="figures"):
    """Generate and save confusion matrix heatmap."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Compute rates
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['No Switch', 'Switch'],
        yticklabels=['No Switch', 'Switch'],
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        cbar=True,
        linewidths=2,
        linecolor='black'
    )
    
    # Labels and title
    ax.set_ylabel('True Label', fontsize=12, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix\n(Test Set: n={cm.sum()} samples)', 
                 fontsize=14, weight='bold', pad=20)
    
    # Add rates as text
    fig.text(0.5, 0.02, 
             f'Specificity (TNR): {specificity:.1%} | Sensitivity (TPR): {sensitivity:.1%}',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = os.path.join(save_dir, "confusion_matrix_bilstm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm, tn, fp, fn, tp, specificity, sensitivity

def print_confusion_matrix_report(y_true, y_pred, model_name="BiLSTM", save_dir="figures"):
    """Print detailed confusion matrix analysis."""
    
    cm, tn, fp, fn, tp, specificity, sensitivity = plot_confusion_matrix_heatmap(
        y_true, y_pred, model_name=model_name, save_dir=save_dir
    )
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - CONFUSION MATRIX ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):   {tn:6d}  (correctly predicted no_switch)")
    print(f"  False Positives (FP):  {fp:6d}  (incorrectly predicted switch)")
    print(f"  False Negatives (FN):  {fn:6d}  (incorrectly predicted no_switch)")
    print(f"  True Positives (TP):   {tp:6d}  (correctly predicted switch)")
    print(f"  {'─'*60}")
    print(f"  Total:                 {cm.sum():6d}")
    
    print(f"\nError Rates:")
    print(f"  False Positive Rate (Type I):  {fp/(tn+fp)*100:.1f}%  (out of {tn+fp} actual no_switch)")
    print(f"  False Negative Rate (Type II): {fn/(tp+fn)*100:.1f}%  (out of {tp+fn} actual switch)")
    
    print(f"\nQuality Metrics:")
    print(f"  Specificity (TNR):  {specificity:.1%}  (correctly identify no_switch)")
    print(f"  Sensitivity (TPR):  {sensitivity:.1%}  (correctly identify switch)")
    print(f"  Balance:            {'✓ Balanced' if abs(specificity - sensitivity) < 0.05 else '⚠ Imbalanced'}")
    print(f"{'='*70}\n")

# Training entrypoint with improved learning dynamics
def train_bilstm_model(csv_path, device=None, epochs=30, batch_size=256, embed_dim=128, hidden_dim=256):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    df = load_local_data(csv_path)
    data = create_switch_prediction_labels(df)
    contexts, labels = flatten_examples(data)
    if len(contexts) == 0:
        raise ValueError("No training data found in CSV tokens column")

    print(f"Total training examples: {len(contexts)}")
    print(f"Class distribution: {np.bincount(labels)} (0=no_switch, 1=switch)")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(contexts, labels, test_size=0.2, random_state=42, stratify=labels)

    # tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(X_train)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # determine maxlen from training contexts
    maxlen = min(128, max((len(s.split()) for s in X_train), default=1))
    print(f"Max sequence length: {maxlen}")
    
    X_train_arr = tokenizer.transform(X_train, maxlen=maxlen)
    X_test_arr = tokenizer.transform(X_test, maxlen=maxlen)

    # create datasets
    train_ds = SeqDataset(X_train_arr, np.array(y_train, dtype=np.float32))
    test_ds = SeqDataset(X_test_arr, np.array(y_test, dtype=np.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # model with improved architecture
    model = BiLSTMClassifier(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=2, dropout=0.3)
    model.to(device)
    
    # Use AdamW with weight decay (better generalization)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    # Compute pos_weight for class imbalance
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"Pos weight (for class imbalance): {pos_weight.item():.3f}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    patience_counter = 0

    # training loop with early stopping
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            opt.step()
            epoch_loss += loss.item() * Xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        epoch_loss /= len(train_loader.dataset)
        acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                p = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(p)
                val_labels.append(yb.numpy())
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = 0.0
        
        print(f"Epoch {ep+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {acc:.4f} - Val AUC: {val_auc:.4f}")
        
        # Early stopping based on AUC
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {ep+1}")
                model.load_state_dict(best_model_state)
                break

    # evaluation on test set
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            p = torch.sigmoid(logits).cpu().numpy()
            preds = (p >= 0.5).astype(int)
            all_probs.append(p)
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')

    print("\n===== BiLSTM Results =====")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["no_switch", "switch"], zero_division=0))

    # Generate confusion matrix heatmap
    print("\nGenerating confusion matrix heatmap...")
    print_confusion_matrix_report(all_labels, all_preds, model_name="BiLSTM", save_dir="figures")

    # wrapper and return
    wrapper = TorchClassifierWrapper(model, device=device, batch_size=batch_size)
    return wrapper, tokenizer

if __name__ == "__main__":
    clf, tokenizer = train_bilstm_model("data/processed/2025-11-20/processed_dataset.csv", epochs=30)