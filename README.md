# Code-Switching Detection in Bilingual Dialogue

A deep learning project for detecting language-switch boundaries in English-Chinese bilingual conversations using BiLSTM neural networks.

## 🎯 Project Overview

**Code-switching** occurs when bilingual speakers alternate between languages within a single conversation. This project tackles **token-level code-switch boundary detection**—predicting whether a language switch occurs between consecutive tokens in bilingual dialogue.

### Key Results

| Model               | Accuracy  | Precision | Recall    | F1-score  | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.711     | 0.583     | 0.766     | 0.662     | 0.795     |
| Random Forest       | 0.763     | 0.649     | 0.778     | 0.708     | 0.831     |
| **BiLSTM (Final)**  | **0.834** | **0.748** | **0.828** | **0.786** | **0.915** |

## 📊 Dataset

- **Total examples**: 118,965 token-boundary classification instances
- **Train/test split**: 80/20 (stratified)
- **Class distribution**: 63% no-switch, 37% switch (balanced)
- **Vocabulary size**: 3,094 unique tokens
- **Max sequence length**: 50 tokens

### Data Generation

Synthetic dialogues generated using **OpenAI GPT-3.5/GPT-4** with:

- **V5 prompt template** with parameter sweeps
- **4 domains**: Tech/Professional (25%), Casual/Social (30%), Family/Intimate (25%), Narrative (20%)
- **5 persona pairs**: colleagues, friends, siblings, family, romantic
- **Agentic validation pipeline**: Language ratio checks, switch frequency validation, naturalness scoring
- **~80% acceptance rate** after quality filtering

See [`data_generation_v5.py`](#data-generation-script) for details.

## 🏗️ Architecture

### BiLSTM Classifier

```
Input: [token_1, token_2, ..., token_50]
  ↓
Embedding Layer (vocab_size=3094 → embed_dim=128)
  ↓
BiLSTM (2 layers, hidden_dim=256, bidirectional, dropout=0.3)
  ↓
Pooling: Mean + Last Hidden State (combined)
  ↓
FC Layers: [Dense(256→128) + ReLU + Dropout] → [Dense(128→1)]
  ↓
Sigmoid Output: probability ∈ [0, 1]
```

### Why BiLSTM?

- **Sequential modeling**: Captures token order (unlike bag-of-words)
- **Bidirectionality**: Reads context left-to-right AND right-to-left
- **Long-range dependencies**: Handles distant context relationships
- **Better than baselines**: +7.1% accuracy over Random Forest

### Training

**Hyperparameters:**

- Learning rate: 1e-3 (AdamW optimizer)
- Batch size: 256
- Loss function: BCEWithLogitsLoss with pos_weight=1.72 (class imbalance)
- Gradient clipping: max_norm=1.0
- Early stopping: patience=5 epochs

**Training dynamics:**

- Peak validation AUC: 0.9202 (Epoch 5)
- Final test ROC-AUC: 0.915
- Training time: ~20 minutes (15 epochs with early stopping)

## 📁 Project Structure

```
.
├── README.md                           # This file
├── scripts/
│   ├── data_generation_v5.py           # Synthetic dialogue generation (V5 template + parameter sweep)
│   ├── logistic_regression.py          # Baseline: TF-IDF + LogReg
│   ├── random_forest.py                # Baseline: Random Forest
│   ├── bilstm_model.py                 # Main model: BiLSTM with heatmap
│   └── main.py                         # End-to-end pipeline + visualization
├── data/
│   ├── raw/
│   │   └── generated_dialogues_v5.json # Generated synthetic data (118K examples)
│   └── processed/
│       ├── processed_dataset.csv        # Tokenized, flattened examples
│       ├── lg_df.csv                   # Logistic Regression predictions
│       ├── rf_df.csv                   # Random Forest predictions
│       └── bilstm_df.csv               # BiLSTM predictions
├── figures/
│   ├── confusion_matrix_bilstm.png      # BiLSTM confusion matrix heatmap
│   ├── language_proportion_pie.png      # Language composition
│   ├── switch_point_histogram.png       # Switch location distribution
│   ├── utterance_length_by_language_type.png
│   ├── switch_prob_boxplot.png          # Predicted probabilities
│   ├── token_language_pie.png           # Token-level EN vs ZH
│   ├── distribution_code_switching_types.png
│   ├── dataset_class_balance_utterance.png
│   ├── dataset_class_balance_token.png
│   ├── code_switching_frequency_distribution.png
│   ├── code_switching_raw_histogram.png
│   ├── pos_language_biases.png          # POS tag analysis
│   ├── length_vs_switch_scatter.png     # Utterance length vs switches
│   └── length_vs_switch_boxplot.png
└── requirements.txt                    # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/code-switching-bilstm.git
cd code-switching-bilstm

# Install dependencies
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
# Requires OpenAI API key in .env file
export API_KEY="sk-..."

python scripts/data_generation_v5.py
```

This generates ~2,000 synthetic dialogues with parameter sweeps across domains and personas.

### Train Models

```bash
# End-to-end pipeline: data processing + train all 3 models + generate visualizations
python scripts/main.py

# Or train individual models:
python scripts/logistic_regression.py
python scripts/random_forest.py
python scripts/bilstm_model.py
```

**Output:**

- Trained model metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix heatmap saved to `figures/confusion_matrix_bilstm.png`
- Predictions on test set saved to `data/processed/`
- 8+ analytical visualizations in `figures/`

## 📈 Key Findings

### Model Performance

- **BiLSTM achieves 91.5% ROC-AUC**, outperforming simpler baselines
- Bidirectional context crucial: Random Forest (bag-of-words) misses 5.0% more switches
- Early stopping at Epoch 5 prevented overfitting (val AUC plateau at 0.9202)

### Error Analysis

**False Positives (16.1%)**: High vocabulary diversity in mono-lingual utterances

- Example: "我们今天讨论工程师的职责和团队的目标" (all Chinese, no switch)
- Model incorrectly predicts switch due to diverse tokens

**False Negatives (17.0%)**: Subtle, natural technical term switches

- Example: "我们需要 deploy 这个功能" (Mandarin + English technical term)
- Model misses contextually smooth switches

### Linguistic Insights

- **Language balance**: 63% Chinese, 37% English tokens (realistic code-switching ratio)
- **Switch frequency**: 37% of token boundaries are switches (high code-mixing)
- **Domain variation**: Tech contexts default to English; casual contexts more balanced
- **POS patterns**: Nouns and verbs have distinct language preferences

## 🔬 Ethical Analysis

### Detected Bias

**Domain Confinement**: 68% of generated data was tech/professional domain

- Risk: Model overfits to formal language, fails on casual speech
- Mitigation: V5 prompt enforces domain stratification (25% each for 4 domains)

**English Authority Skew**: Technical terms default to English

- Reflects real-world bias but risks overfitting
- Addressed via diverse persona generation and casualness checks

### Mitigation Strategy

1. **Prompt refinement**: V2→V5 added explicit domain percentages
2. **Data filtering**: Vocabulary audits, casualness checks, manual validation
3. **Rebalancing**: Oversampled informal/family dialogues to 40% of dataset
4. **Post-mitigation**: Domain diversity improved from 58% → 78%

## 🔮 Future Work

### Short-term

1. **Multilingual embeddings**: Replace random embeddings with mBERT

   - Expected improvement: +5% ROC-AUC
   - Captures cross-lingual semantic relationships

2. **Threshold optimization**: Tune decision boundary per domain

   - Logistics: Different domains may need different thresholds

3. **Real data validation**: Test on authentic bilingual dialogue
   - Current: Synthetic data only

### Long-term

1. **More language pairs**: Extend to Hindi-English, Spanish-English, etc.
2. **Phonetic code-switching**: Handle pronunciation-driven switches
3. **Contextual semantics**: Why do speakers switch? (pragmatic analysis)
4. **Dialect variation**: Account for regional/generational differences

## 📚 References

### Data & Code

- OpenAI GPT API for synthetic data generation
- PyTorch for BiLSTM implementation
- scikit-learn for baselines and evaluation

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 👥 Authors

Keith Yao & Arav Goyal

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## 📞 Contact

For questions or feedback:

- Open an issue on GitHub
- Email: yao.kei@northeastern.edu

---

**Last Updated**: December 2025  
**Dataset Version**: V5 (2,000 synthetic dialogues, 4-domain stratified)  
**Model Version**: BiLSTM (2-layer, 256-dim hidden, ROC-AUC 0.915)
