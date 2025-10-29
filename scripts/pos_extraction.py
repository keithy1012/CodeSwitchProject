import spacy
import pandas as pd
from typing import List, Dict, Tuple, Any

# --- Configuration and Setup ---

# Load the English spaCy model. This line assumes the model 'en_core_web_sm' is downloaded.
try:
    NLP_EN = spacy.load('en_core_web_sm')
    print(" spaCy English model loaded successfully.")
except OSError:
    print(" ERROR: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    # Exit or use a mock function if the model is crucial. For this example, we proceed with the assumption it loads.

# --- Raw Dialogue Input ---

RAW_DIALOGUE = """
Analyst: Good morning, team. We need to analyze the Q3 sales data. What are the current figures?
Manager: I have the preliminary report. Total revenue is up by 12% year-over-year.
Analyst: That is excellent news! Where is the growth concentrated geographically?
Manager: Primarily in the Asian markets. Their aggressive marketing strategy worked perfectly.
Analyst: We must replicate that success next quarter. Please send me the detailed regional breakdown.
"""

# --- Core Functions ---

def parse_dialogue(raw_text: str) -> List[Dict[str, str]]:
    """
    Organizes the raw, multi-turn dialogue into a structured list of turns.
    Assumes a format like 'Speaker: Utterance' on a new line.
    """
    structured_dialogue = []
    lines = raw_text.strip().split('\n')
    turn_id = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Splits at the first colon
            speaker, utterance = line.split(':', 1)
            structured_dialogue.append({
                'turn_id': turn_id,
                'speaker': speaker.strip(),
                'utterance': utterance.strip()
            })
            turn_id += 1
        except ValueError:
            # Handle lines that don't match the 'Speaker: Utterance' format
            print(f" Skipping unparsable line: {line}")
            continue

    return structured_dialogue

def analyze_text_en(utterance: str, turn_id: int) -> List[Dict[str, Any]]:
    """
    Applies tokenization and English POS tagging using spaCy.
    """
    # Process the text with the loaded spaCy model
    doc = NLP_EN(utterance)
    analysis_results = []

    for token in doc:
        analysis_results.append({
            'Turn ID': turn_id,
            'Token': token.text,
            'Lemma': token.lemma_, # Base form of the word
            'POS Tag': token.pos_,  # Universal POS Tag (e.g., NOUN, VERB)
            'Syntactic Dependency': token.dep_, # Grammatical relationship
            'Is Stopword': token.is_stop # Common words (a, the, in, etc.)
        })

    return analysis_results



def discuss_linguistic_patterns(results_df: pd.DataFrame) -> str:
    """
    Analyzes the extracted tokens and POS tags to discuss linguistic patterns.
    """
    total_tokens = len(results_df)
    noun_count = len(results_df[results_df['POS Tag'] == 'NOUN'])
    verb_count = len(results_df[results_df['POS Tag'] == 'VERB'])
    adj_count = len(results_df[results_df['POS Tag'] == 'ADJ'])
    proper_nouns = results_df[results_df['POS Tag'] == 'PROPN']['Token'].tolist()

    discussion = (
        "\n--- Linguistic Pattern Analysis ---\n"
        f"This is a business-focused dialogue containing {total_tokens} tokens.\n"
        "Key observations based on Part-of-Speech (POS) tagging:\n"
        f"1.  **High Noun-to-Verb Ratio:** With {noun_count} Nouns and {verb_count} Verbs, the language is highly descriptive and transactional. "
        "The density of Nouns reflects a focus on concepts and data (e.g., 'revenue', 'figures', 'breakdown', 'strategy', 'markets').\n"
        f"2.  **Specific Terminology:** Proper Nouns detected include: {', '.join(set(proper_nouns)) if proper_nouns else 'None'}. "
        "These tokens often carry the most important context (e.g., specific reports, quarters, or regions).\n"
        f"3.  **Command/Request Structure:** The dialogue frequently uses imperative or modal verbs to guide action ('We **need** to analyze', 'Please **send** me...'). "
        "These structures, tagged as VERB or AUX, drive the conversational goal (task completion)."
    )
    return discussion


def main():
    """
    Main function to execute the dialogue structuring and analysis workflow.
    """
    print("Starting Dialogue Analyzer Workflow...\n")

    # 1. Structure the raw dialogue
    structured_dialogue = parse_dialogue(RAW_DIALOGUE)

    print("1. Dialogue Turns Extracted:")
    turns_df = pd.DataFrame(structured_dialogue)
    print(turns_df.to_markdown(index=False))

    # 2. & 3. Apply Tokenization and POS Tagging
    all_analysis_data = []
    for turn in structured_dialogue:
        # We perform English analysis for the provided dialogue
        analysis_data = analyze_text_en(turn['utterance'], turn['turn_id'])
        all_analysis_data.extend(analysis_data)

    if not all_analysis_data:
        print("\nAnalysis could not be performed due to model loading error.")
        return

    # 4. Organize and Display Results
    analysis_df = pd.DataFrame(all_analysis_data)

    # Display only the most relevant columns for the final analysis table
    print("\n\n2. Tokenization and POS Tagging Results (English):")
    final_display_df = analysis_df[['Turn ID', 'Token', 'POS Tag', 'Lemma', 'Is Stopword']]
    print(final_display_df.to_markdown(index=False, numalign="left", stralign="left"))


    # 5. Discuss Linguistic Patterns
    linguistic_discussion = discuss_linguistic_patterns(analysis_df)
    print(linguistic_discussion)



if __name__ == "__main__":
    main()