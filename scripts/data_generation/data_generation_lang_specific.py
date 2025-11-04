import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)

# ====== CONFIG ======
OUTPUT_PATH = "data/raw/generated_dialogues_monolingual.json"
N = 2 # number of dialogues per prompt per model
MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o-mini"]
# ====================

'''
    {
        "prompt_text": """你是一位擅长生成中文职场对话的数据集专家。请模拟两个软件工程师（李伟 和 王珊）的真实对话。
整个对话必须使用中文（不要夹杂任何英文单词或代码混合）。
语气自然，像同事在公司聊天。讨论内容包括部署、构建、修复漏洞或赶项目截止日期。
Few-shot 示例：
A: “你那边构建完了吗？我准备合并到主分支。”
B: “还没，我先跑一下测试，等会儿再部署。”
现在生成一个 5–7 轮的中文对话，只输出对话内容。
""",
        "generation_strategy": ["few-shot", "persona-based"]
    },
    {
        "prompt_text": """你是资深后端工程师李伟，正在和前端同事王珊聊天。
整个对话请保持纯中文（不要出现任何英文技术词汇），语气要真实自然，像同事在讨论工作问题。
话题可以包括构建、部署、修复漏洞、合并代码或赶项目截止日期。
生成一个 5–7 轮的中文对话，只输出对话内容。
""",
        "generation_strategy": ["chain-of-thought", "persona-based"]
    },
    {
        "prompt_text": """生成一个完全中文的工程师工作对话，人物是李伟（后端）和王珊（前端）。
语气自然、真实、有一点职场紧迫感。
讨论内容可以包括部署、合并、修复漏洞、赶截止日期等。
不要使用任何英文单词或混合语言，所有技术词汇请用中文表达。
生成 5–7 轮对话，只输出对话

""",
        "generation_strategy": ["chain-of-thought", "few-shot"]
    }
'''
# PROMPTS — monolingual only (no code switching)
prompts = [
    {
        "prompt_text": """You are an expert dataset generator for workplace English dialogues between software engineers. 
Act as Sarah, a frontend engineer in San Francisco, chatting with Li, a backend engineer from Shanghai who speaks fluent English. 
Keep the entire conversation in English (no Chinese characters). 
The tone should sound like Slack or hallway conversation — short, technical turns about debugging, deploying, or merging code.
Include at least one of these trigger words: deploy, build, bug, deadline, merge.
Few-shot examples:
A: “Can you push the fix to staging?”
B: “Yeah, I’ll deploy right after lunch.”
A: “Cool, I’ll run QA on the build then.”
Now: produce a 5–7 turn dialogue between Li and Sarah, fully in English.
Output only the dialogue.""",
        "generation_strategy": ["persona-based", "few-shot"]
    },
    {
        "prompt_text": """You are a senior software engineer roleplaying natural English workplace chat. 
Li and Sarah are discussing a merge conflict and deployment blockers. 
Keep the tone pragmatic and realistic — no Chinese text at all. 
Use natural conversational phrasing like "staging build" or "hotfix".
Each dialogue should be 5–7 turns long and include one of: deploy, build, bug, deadline, merge.
Output only the conversation.
""",
        "generation_strategy": ["persona-based", "chain-of-thought"]
    },
    {
        "prompt_text": """Generate a purely English conversation between Li and Sarah, two engineers working to meet a Friday deployment deadline.
Keep tone friendly but work-focused.
Include words like deploy, merge, bug, or build naturally in context.
Each dialogue should have 5–7 turns.
Output only the dialogue.
""",
        "generation_strategy": ["few-shot", "chain-of-thought"]
    },
    {
        "prompt_text": """You are a senior software engineer roleplaying natural English workplace chat. 
Li and Sarah are discussing a merge conflict and deployment blockers. 
Keep the tone pragmatic and realistic — no Chinese text at all. 
Use natural conversational phrasing like "staging build" or "hotfix".
Each dialogue should be 5–7 turns long and include one of: deploy, build, bug, deadline, merge.
Output only the conversation.
""",
        "generation_strategy": ["chain-of-thought", "persona-based"]
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
