import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# ====== CONFIG ======
OUTPUT_PATH = "data/raw/generated_dialogue.json"
N = 5  # number of dialogues per prompt per model
MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o-mini"]
# ====================

# 6 DIVERSE PROMPTS: Each combines two prompting techniques
prompts = [
    {
        "prompt_text": """You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. 
Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, 
and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.
Few-shot examples:
Example 1:
A: "Can you push the latest styles to staging?"
B: "可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird."
Example 2:
A: "There's a bug in the login flow after the OAuth change."
B: "看一下 redirect URI, scope 好像没有 properly 设置."
Example 3:
A: "We need to ship before Friday's deadline."
B: "好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint."
Now: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.
Requirements:
Use **English–Mandarin code-switching** at the **phrase level**.
Include at least one trigger word: **deploy, build, bug, deadline, merge**.
Target ~35–50% Mandarin tokens per dialogue.
Use natural engineering register (issue, PR, deploy, QA, staging, workaround).
Keep turns short-to-medium.
Output only the dialogues. Do not repeat the few-shot examples.""",
        "generation_strategy": ["few-shot", "persona-based"]
    },
    {
        "prompt_text": """Simulate two bilingual software engineers (Li and Sarah) working remotely across time zones. 
Use reasoning (chain of thought) to plan a natural English–Mandarin dialogue about resolving a deployment bug, 
then output only the conversation (5–7 turns). 
Mix Mandarin and English at the phrase level, like real bilingual speech.
Include at least one of: deploy, build, bug, deadline, merge.
Use casual engineering tone, short to medium turns, 35–50% Mandarin content.""",
        "generation_strategy": ["chain-of-thought", "persona-based"]
    },
    {
        "prompt_text": """You are an expert bilingual dialogue writer. Use a few-shot reasoning approach to write 
a 5–7 turn English–Mandarin code-switched conversation between Li and Sarah debugging a front-end issue. 
Reference QA, staging, and deploy naturally.
Include at least one trigger word (deploy, build, bug, deadline, merge).
Switch languages mid-sentence, around technical phrases. 
Output only the dialogue.""",
        "generation_strategy": ["few-shot", "chain-of-thought"]
    },
    {
        "prompt_text": """Adopt the persona of two full-stack developers (Li and Sarah). You’re casually discussing 
a merge conflict and upcoming deadline. Use English–Mandarin code-switching naturally and fluently. 
Reference tools like GitHub, PRs, staging, and deploy. 
Include one trigger word: deploy, build, bug, deadline, or merge. 
Keep turns concise (5–7 exchanges total). Output only the dialogue.""",
        "generation_strategy": ["persona-based", "chain-of-thought"]
    },
    {
        "prompt_text": """You are creating realistic bilingual training data. Generate a short 5–7 turn dialogue between 
Li and Sarah, two engineers reviewing a PR before a deploy. 
Use code-switching between Mandarin and English at phrase level (within sentences).
At least one of these words must appear: deploy, build, bug, deadline, merge. 
Follow a few-shot reasoning mindset and make tone authentic, like teammates chatting over Slack.""",
        "generation_strategy": ["few-shot", "chain-of-thought"]
    },
    {
        "prompt_text": """Write a natural bilingual (English–Mandarin) conversation between two engineers (Li and Sarah) 
preparing for a Friday deploy. You are a persona-based dataset creator focused on realistic technical speech. 
Include at least one of: deploy, build, bug, deadline, merge.
Target ~40% Mandarin content, switch languages mid-sentence around technical terms.
Each dialogue should be 5–7 turns. Keep the tone informal but technical. Output only the dialogue.""",
        "generation_strategy": ["persona-based", "few-shot"]
    }
]

# === Generation loop ===
results = []
counter = 1

for prompt in prompts:
    for model_choice in MODELS:
        for _ in range(N):
            print(f"Generating id={counter} with model={model_choice}...")
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": prompt["prompt_text"]}],
                )
                text_output = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error on id={counter}: {e}")
                text_output = None

            results.append({
                "id": counter,
                "utterance": text_output,
                "source": "LLM",
                "generation_strategy": prompt["generation_strategy"],
                "model_name": model_choice,
                "prompt_text": prompt["prompt_text"]
            })
            counter += 1
            print(text_output)

# Save to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} generated dialogues to {OUTPUT_PATH}")
