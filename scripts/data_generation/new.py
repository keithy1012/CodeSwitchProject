import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict

# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)

# ====== CONFIG ======
OUTPUT_PATH = "data/raw/generated_dialogues_v5.json"
N_PER_COMBO = 2  # number of dialogues per domain-persona-model combo
MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]  # Reduced for cost
MAX_RETRIES = 3  # Retry rejected dialogues
# ====================

# ====== PARAMETER SWEEP DEFINITION ======
domains = {
    "tech": {
        "name": "Tech/Professional",
        "description": "Project meetings, technical discussions, startup talk",
        "target_percentage": 0.25
    },
    "casual": {
        "name": "Casual/Social",
        "description": "Friend hangouts, phone calls, social media vibes",
        "target_percentage": 0.30
    },
    "family": {
        "name": "Family/Intimate",
        "description": "Parent-child, sibling banter, emotional moments",
        "target_percentage": 0.25
    },
    "narrative": {
        "name": "Narrative/Storytelling",
        "description": "Storytelling, past events, jokes",
        "target_percentage": 0.20
    }
}

personas = [
    {"name_a": "Li", "name_b": "Sarah", "relationship": "colleagues", "role": "frontend engineer from Shanghai, backend engineer from SF"},
    {"name_a": "Wei", "name_b": "Jake", "relationship": "friends", "role": "college friends, one in tech, one in finance"},
    {"name_a": "Chen", "name_b": "Alex", "relationship": "siblings", "role": "older sibling (Chen) and younger sibling (Alex)"},
    {"name_a": "Mom (王女士)", "name_b": "Son (王明)", "relationship": "family", "role": "mother and son, casual family chat"},
    {"name_a": "Lily", "name_b": "Marcus", "relationship": "romantic", "role": "boyfriend and girlfriend, long-distance"},
]

# ====== V5 PROMPT TEMPLATE ======
def create_v5_prompt(domain_key, domain_config, persona, code_switch=True):
    """Generate V5 prompt with injected parameters."""
    
    domain_name = domain_config["name"]
    domain_desc = domain_config["description"]
    persona_a = persona["name_a"]
    persona_b = persona["name_b"]
    relationship = persona["relationship"]
    role_desc = persona["role"]
    
    if code_switch:
        # Code-switching version
        prompt = f"""You are a bilingual English-Chinese dialogue writer generating natural conversations.

TASK: Create a realistic dialogue where speakers code-switch naturally (mix EN & ZH mid-sentence).

CONSTRAINTS:

1. Dialogue Type: {domain_name}
   - Description: {domain_desc}

2. Speaker Setup: {persona_a} and {persona_b}
   - Relationship: {relationship}
   - Context: {role_desc}
   - Both are native bilingual speakers (NOT learners)
   - Authority: Equal peers (no power dynamics)

3. Code-Switching Rules:
   - Target: ~37% English tokens, ~63% Chinese tokens (overall ratio)
   - Switches per utterance: 2-5 language transitions (intra-sentential)
   - Motivation examples:
     * Technical terms default to English (project, algorithm, database, deploy, build, bug)
     * Emotional expressions in native language (好吧, seriously, 天哪, 不行)
     * Code-switching for emphasis: "这太 crazy 了" (that's too crazy)
     * Natural language mixing, NOT forced or artificial

4. Format Output:
   [Speaker A]: [utterance with tokens naturally mixed]
   [Speaker B]: [response with natural switches]
   [Speaker A]: [next turn]
   ...
   (Generate 8-12 dialogue turns)

5. Quality Checks:
   ✓ No repetitive patterns or templates
   ✓ Each switch has pragmatic motivation (not random)
   ✓ Grammar correct in both languages
   ✓ No English romanization of Chinese (use proper 汉字)
   ✓ Conversational & realistic (natural tone for {relationship})
   ✓ Include at least one trigger word: deploy, build, bug, deadline, merge, 项目, 截止

OUTPUT: Return only the dialogue, no additional text.
"""
    else:
        # Monolingual English version
        prompt = f"""You are an expert dataset generator for workplace English dialogues.

TASK: Create a realistic conversation between {persona_a} and {persona_b}.

CONSTRAINTS:

1. Dialogue Type: {domain_name}
   - Description: {domain_desc}

2. Speaker Setup: {persona_a} and {persona_b}
   - Relationship: {relationship}
   - Context: {role_desc}
   - Keep the entire conversation in English (no Chinese characters)

3. Language Rules:
   - 100% English, no code-switching
   - Technical tone appropriate for {domain_name}
   - Natural conversational phrasing
   - Include at least one of these trigger words: deploy, build, bug, deadline, merge

4. Format Output:
   [Speaker A]: [utterance]
   [Speaker B]: [response]
   [Speaker A]: [next turn]
   ...
   (Generate 8-12 dialogue turns)

5. Quality Checks:
   ✓ No Chinese characters or mixing
   ✓ Grammar is natural and conversational
   ✓ Tone appropriate for {relationship}
   ✓ Realistic dialogue flow

OUTPUT: Return only the dialogue, no additional text.
"""
    
    return prompt

# ====== VALIDATORS ======
def count_language_tokens(utterance):
    """Count English and Chinese tokens in utterance."""
    en_tokens = len(re.findall(r'[a-zA-Z]+', utterance))
    zh_tokens = len(re.findall(r'[\u4e00-\u9fff]', utterance))
    total_tokens = en_tokens + zh_tokens
    return en_tokens, zh_tokens, total_tokens

def count_switches(utterance):
    """Count language switches in utterance (ZH→EN or EN→ZH)."""
    tokens = re.split(r'\s+', utterance.strip())
    switch_count = 0
    prev_lang = None
    
    for token in tokens:
        has_en = bool(re.search(r'[a-zA-Z]', token))
        has_zh = bool(re.search(r'[\u4e00-\u9fff]', token))
        
        if has_zh and not has_en:
            curr_lang = "zh"
        elif has_en and not has_zh:
            curr_lang = "en"
        else:
            curr_lang = None  # Mixed token, skip
        
        if curr_lang and prev_lang and curr_lang != prev_lang:
            switch_count += 1
        
        if curr_lang:
            prev_lang = curr_lang
    
    return switch_count

def validate_dialogue(dialogue_text, code_switch=True):
    """Validate generated dialogue against constraints."""
    
    utterances = [line.strip() for line in dialogue_text.split('\n') if ':' in line]
    
    if len(utterances) < 6:
        return False, "Too few utterances (need ≥6)"
    
    if len(utterances) > 14:
        return False, "Too many utterances (max 12)"
    
    # Check for trigger words
    trigger_words = ["deploy", "build", "bug", "deadline", "merge", "项目", "截止"]
    has_trigger = any(word in dialogue_text.lower() for word in trigger_words)
    if not has_trigger:
        return False, "Missing trigger words"
    
    if code_switch:
        # Code-switching validation
        total_en = 0
        total_zh = 0
        total_switches = 0
        
        for utterance in utterances:
            en, zh, _ = count_language_tokens(utterance)
            total_en += en
            total_zh += zh
            total_switches += count_switches(utterance)
        
        total_tokens = total_en + total_zh
        if total_tokens < 20:
            return False, "Too few tokens overall"
        
        en_ratio = total_en / total_tokens if total_tokens > 0 else 0
        if not (0.32 <= en_ratio <= 0.42):
            return False, f"EN ratio {en_ratio:.2f} out of range [0.32, 0.42]"
        
        avg_switches = total_switches / len(utterances)
        if avg_switches < 1.0:
            return False, f"Insufficient switches (avg {avg_switches:.2f} per utterance, need ≥1.0)"
    
    else:
        # Monolingual English validation
        total_en = 0
        total_zh = 0
        
        for utterance in utterances:
            en, zh, _ = count_language_tokens(utterance)
            total_en += en
            total_zh += zh
        
        if total_zh > 0:
            return False, "Found Chinese characters in English-only dialogue"
        
        if total_en < 20:
            return False, "Too few English tokens"
    
    return True, "Valid"

def score_naturalness(dialogue_text):
    """Heuristic scoring (1-5 stars) for dialogue naturalness."""
    
    score = 3.0  # Base score
    
    # Positive signals
    if "..." in dialogue_text or "hmm" in dialogue_text.lower():
        score += 0.5  # Natural hesitation markers
    
    if any(word in dialogue_text for word in ["yeah", "ok", "sure", "好的", "嗯", "对"]):
        score += 0.5  # Natural acknowledgments
    
    # Negative signals
    if dialogue_text.count('\n') > 25:
        score -= 0.5  # Too long/repetitive
    
    if dialogue_text.count('Example') > 0 or dialogue_text.count('few-shot') > 0:
        score -= 1.0  # Model repeated prompt
    
    return min(5.0, max(1.0, score))

# ====== MAIN GENERATION LOOP ======
results = []
counter = 1
rejection_reasons = defaultdict(int)
coverage_tracker = defaultdict(int)

print("="*60)
print("PARAMETER SWEEP DATA GENERATION (V5)")
print("="*60)

for domain_key, domain_config in domains.items():
    for persona in personas:
        for model_choice in MODELS:
            for attempt in range(N_PER_COMBO):
                retry_count = 0
                dialogue_text = None
                valid = False
                
                while retry_count < MAX_RETRIES and not valid:
                    # Randomly decide code-switching (80% code-switch, 20% monolingual)
                    code_switch = (attempt % 5) != 0  # ~80% code-switched
                    
                    # Generate prompt
                    prompt = create_v5_prompt(domain_key, domain_config, persona, code_switch=code_switch)
                    
                    print(f"\n[{counter}] Domain={domain_key}, Persona={persona['name_a']}-{persona['name_b']}, " \
                          f"Model={model_choice}, CodeSwitch={code_switch}, Retry={retry_count}")
                    
                    try:
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.8,  # Higher temp for diversity
                        )
                        dialogue_text = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"  ❌ API Error: {e}")
                        rejection_reasons["api_error"] += 1
                        retry_count += 1
                        continue
                    
                    # Validate
                    valid, reason = validate_dialogue(dialogue_text, code_switch=code_switch)
                    
                    if not valid:
                        print(f"  ❌ Validation failed: {reason}")
                        rejection_reasons[reason] += 1
                        retry_count += 1
                        continue
                    
                    # Score naturalness
                    naturalness = score_naturalness(dialogue_text)
                    print(f"  ✓ Valid | Naturalness: {naturalness:.1f}/5.0")
                    
                    if naturalness < 2.5 and retry_count < MAX_RETRIES - 1:
                        print(f"  ⚠ Low naturalness score, retrying...")
                        rejection_reasons["low_quality"] += 1
                        retry_count += 1
                        continue
                
                if valid:
                    results.append({
                        "id": counter,
                        "dialogue": dialogue_text,
                        "domain": domain_key,
                        "speakers": [persona["name_a"], persona["name_b"]],
                        "relationship": persona["relationship"],
                        "code_switched": code_switch,
                        "model": model_choice,
                        "naturalness_score": naturalness,
                        "prompt_template": "V5",
                        "generation_strategy": ["persona-based", "parameter-sweep"],
                    })
                    coverage_tracker[domain_key] += 1
                    counter += 1
                    print(f"  ✅ SAVED (id={counter-1})")
                else:
                    print(f"  ❌ REJECTED after {MAX_RETRIES} retries")
                    counter += 1

# Save results
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("GENERATION SUMMARY")
print("="*60)
print(f"Total dialogues generated: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"\nDomain coverage:")
total_dialogues = len(results)
for domain_key, count in sorted(coverage_tracker.items()):
    target = domains[domain_key]["target_percentage"] * len(results)
    pct = (count / total_dialogues * 100) if total_dialogues > 0 else 0
    print(f"  {domain_key:12s}: {count:3d} dialogues ({pct:5.1f}% | target {domains[domain_key]['target_percentage']*100:.0f}%)")

print(f"\nRejection breakdown:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
    pct = (count / (count + len(results)) * 100) if (count + len(results)) > 0 else 0
    print(f"  {reason:40s}: {count:3d} ({pct:.1f}%)")