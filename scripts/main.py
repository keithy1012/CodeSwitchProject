import json
import pandas as pd
import spacy
from pathlib import Path
from datetime import date

# Load spaCy English model
# Install first: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Step 4: Load JSON and convert to CSV
json_path = 'data/raw/code_switch_data.json'
csv_path = 'data/processed/processed_dataset.csv'
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Step 5: Add initial linguistic features (tokens + POS) using spaCy
tokens_list = []
pos_tags_list = []

for i, utterance in enumerate(df['utterance']):
    doc = nlp(utterance)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    
    if len(tokens) != len(pos_tags):
        print(f"Warning: Token-POS length mismatch in row {i} (id={df.loc[i,'id']})")
    
    tokens_list.append(tokens)
    pos_tags_list.append(pos_tags)

df['tokens'] = tokens_list
df['pos_tags'] = pos_tags_list

# Step 6: Save to CSV
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"Processed CSV saved to: {csv_path}")

# Step 7: Verify a few examples manually
print(df[['id', 'utterance', 'tokens', 'pos_tags']].head(5))

# Step 8: Create README
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

# Count total tokens in dataset
total_tokens = sum(len(tokens) for tokens in df['tokens'])
print(f"Total number of tokens in dataset: {total_tokens}")