from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)

prompts = [{
    "id": 1,
    "utterance": "Li: \"昨天的 nightly build 挂了，你有看到吗?\"\nSarah: \"Yeah, log 里面说是一个 dependency 没有 properly 安装.\"\nLi: \"我试过 rerun build, 还是 fail. 可能需要 pin version.\"\nSarah: \"好，我开个 PR 去 lock version, 然后我们再 merge 一次.\"\nLi: \"OK，merge 完之后我会再跑一遍 CI.\"\nSarah: \"等确认 pass 了，再准备 deploy 到 staging.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
}
,{
    "id": 2,
    "utterance": "Sarah: \"这个 checkout flow 有个 bug, 用户在 Safari 上卡住.\"\nLi: \"我 reproduce 了一下，好像是 localStorage 没 properly update.\"\nSarah: \"那我们要不要加个小 workaround?\"\nLi: \"可以，写个 fallback logic. 我晚点 push 到 branch.\"\nSarah: \"行，等你 push 完我来 review PR.\"\nLi: \"Review 后尽快 merge，不然 QA 没法继续测.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
}
, {
    "id": 3,
    "utterance": "Li: \"我们这周的 deadline 快到了，dashboard 还没 ready.\"\nSarah: \"对，前端部分我还在 refactor chart component.\"\nLi: \"要不要先 cut 一个简单版？deploy 上去给 PM 看.\"\nSarah: \"好，我可以先把 basic filters 留着，其他 advance 的先砍掉.\"\nLi: \"这样至少能 meet deadline，剩下的功能下个 sprint 加.\"\nSarah: \"Sounds good，我马上 push 到 staging branch.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
}
,{
    "id": 4,
    "utterance": "Sarah: \"昨天的 merge 把 config 搞坏了，production deploy fail 了.\"\nLi: \"啊？我以为 QA 都 pass 了才 merge 的.\"\nSarah: \"QA pass 了，但 prod config file 不一样.\"\nLi: \"那我们要 hotfix 吗？\"\nSarah: \"Yeah，先开个小 patch branch，改 config 再 redeploy.\"\nLi: \"我来处理，你帮忙 double-check 环境变量.\"\nSarah: \"OK, 我会在 Slack 上 confirm 一下.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
}
,{
    "id": 5,
    "utterance": "Li: \"Login 页面 layout 在 Android 上 broken, user report 一个 bug.\"\nSarah: \"我看了下，是 CSS flex 在 old Chrome version 不支持.\"\nLi: \"那要不要写个 polyfill, 或者直接改成 grid?\"\nSarah: \"Grid 更 stable, 我可以马上 build 一版.\"\nLi: \"Build 完 push 到 feature branch, 我来 test.\"\nSarah: \"好，等你确认没问题，我们 merge 后 deploy staging.\"\nLi: \"OK，赶紧搞定，不然 QA 会卡住.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
}
,   {
    "id": 6,
    "utterance": "Chen: \"We saw latency spikes after 11 PM.\"\nJohn: \"Yes, cache miss rate went up sharply.\"\nChen: \"这是昨天晚上 deploy 引起的.\"\nJohn: \"Right, we should rollback 那个 microservice.\"\nChen: \"好的，我准备 rollback plan 并通知 infra.\"\nJohn: \"I'll add monitoring 指标 for cache hits.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats."
  }, {
    "id": 7,
    "utterance": "Chen: \"The timeout logic is fine, but circuit breaker 配置太 aggressive.\"\nJohn: \"It trips after two failures, 然后导致级联错误.\"\nChen: \"我们应该加 threshold 并且 implement exponential backoff.\"\nJohn: \"Right — client retries 现在是 sync, 要改成 async.\"\nChen: \"我来 patch client lib 并更新 doc.\"\nJohn: \"Perfect, 那我们就能 benchmark 修复效果.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats."
  },
    {
    "id": 8,
    "utterance": "Chen: \"This API contract is incompatible, 对吧?\"\nJohn: \"对，他们忘了加 version.\"\nChen: \"We need a migration path, 是不是?\"\nJohn: \"Exactly — deprecate 老接口然后加个 shim.\"\nChen: \"我写 migration plan, maintain dual writes 两个 sprint.\"\nJohn: \"我联系 SDK team 确保他们 clients 不会 break.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats."
  },
    {
    "id": 9,
    "utterance": "Chen: \"There's a payment bug after tokenization.\"\nJohn: \"Token service fails 在某些卡类型.\"\nChen: \"先 hotfix, long-term 再重构.\"\nJohn: \"我加 logging 并且启用 A/B test 来验证.\"\nChen: \"Staging pass 后 we deploy to prod.\"\nJohn: \"好，我也准备 rollback steps 以防万一.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats."
  },
  {
    "id": 10,
    "utterance": "A: \"I'll deploy 新 feature branch this afternoon.\"\nB: \"好，别忘了 to run smoke tests.\"\nA: \"If pass, 我就 tag release 并通知 QA.\"\nB: \"Perfect, 然后我们就 deploy.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: verb-boundary."
  },
  {
    "id": 11,
    "utterance": "A: \"这个 hotfix 只是权宜之计, but it avoids outage.\"\nB: \"是的，下个 sprint 我们必须 refactor 那部分.\"\nA: \"Agree, uptime first, refactor later.\"\nB: \"我加一个 ticket 把 refactor 细节写清楚.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: idiom + switch."
  },
  {
    "id": 12,
    "utterance": "A: \"If cache misses, fallback kicks in.\"\nB: \"如果 fallback 触发, increase timeout 然后 log event.\"\nA: \"Add metrics 来衡量 hit rate.\"\nB: \"有了数据我们就能调整 policy.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: phrasal-trigger."
  },
  {
    "id": 13,
    "utterance": "A: \"Checkout flow 有个 bug when address fails.\"\nB: \"对，validator 不支持本地字符.\"\nA: \"I'll add normalization 然后跑 tests.\"\nB: \"我补 QA cases 来避免 regression.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: phrasal-trigger."
  },
  {
    "id": 14,
    "utterance": "A: \"我要 bump version 然后 publish.\"\nB: \"好，update changelog 在 release 前.\"\nA: \"Ok, publish 后我会开 PR.\"\nB: \"Great, 那我们可以 deploy.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: verb-boundary."
  },
  {
    "id": 15,
    "utterance": "A: \"Merge 完成了, 可是 staging 有个 bug.\"\nB: \"那就先 revert, 然后开新 branch 修复.\"\nA: \"好的，我晚上写 patch.\"\nB: \"别忘了加 unit tests.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: idiom + switch."
  },
  {
    "id": 16,
    "utterance": "Li: \"昨天 build 又 fail 了, 可能是新加的 dependency conflict.\"\nSarah: \"对，我看 log, 是版本 mismatch.\"\nLi: \"要不要先 revert 然后再 pin version?\"\nSarah: \"好，我开 PR 修复依赖，然后你再 merge.\"\nLi: \"Merge 后我跑一次 CI，确保 staging pass.\"\nSarah: \"OK, pass 后我们就准备 deploy.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
  },
  {
    "id": 17,
    "utterance": "Sarah: \"这个 checkout 页面在 Safari 上有个 bug.\"\nLi: \"我看了，是 form validation 没 properly trigger.\"\nSarah: \"那我们先加个 fallback?\"\nLi: \"可以，写个 JS snippet 来 handle edge case.\"\nSarah: \"Snippet push 到 branch 后我来 review PR.\"\nLi: \"Review 后 merge, QA 就能继续测试.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
  },
  {
    "id": 18,
    "utterance": "Li: \"我们这周 deadline 快到了, dashboard feature 还没完成.\"\nSarah: \"前端 chart 组件我还在 refactor.\"\nLi: \"要不要 cut 个 MVP 先 deploy 给 PM 看?\"\nSarah: \"可以，我保留 basic filters, advance 部分下个 sprint 加.\"\nLi: \"这样至少 meet deadline, 其他功能下次再加.\"\nSarah: \"Push 到 staging branch, 我们可以 demo.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
  },
  {
    "id": 19,
    "utterance": "Sarah: \"昨天 merge 导致 config 错乱, prod deploy fail.\"\nLi: \"QA 不是都 pass 了吗?\"\nSarah: \"QA pass, 但 prod config 有差异.\"\nLi: \"那我们开个 hotfix branch?\"\nSarah: \"对，改完 config 再 redeploy.\"\nLi: \"我来 patch, 你 double-check 环境变量.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
  },
  {
    "id": 20,
    "utterance": "Li: \"Login 页面在 Android broken, user report bug.\"\nSarah: \"CSS flex 不支持旧版 Chrome.\"\nLi: \"改成 grid 或写 polyfill?\"\nSarah: \"Grid 更 stable, 我 build 一版.\"\nLi: \"Build 完 push 到 feature branch, 我来 test.\"\nSarah: \"Test 没问题就 merge 并 deploy 到 staging.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Few-shot"],
    "model_name": "GPT-4",
    "prompt_text": "You are an expert dataset generator for bilingual English–Mandarin code-switched technical dialogues. Act as a frontend engineer from Shanghai now working in San Francisco. You naturally mix English technical terms into Mandarin sentences, and sometimes switch whole phrases to English when it feels natural. Keep the tone pragmatic, casual, like real engineers chatting.\nFew-shot examples:\nExample 1:\nA: \"Can you push the latest styles to staging?\"\nB: \"可以，我现在就 push. 不过 header 在 mobile 上看起来有点 weird.\"\nExample 2:\nA: \"There's a bug in the login flow after the OAuth change.\"\nB: \"看一下 redirect URI, scope 好像没有 properly 设置.\"\nExample 3:\nA: \"We need to ship before Friday's deadline.\"\nB: \"好，先 prioritize login 和 checkout, 其他功能可以等下个 sprint.\"\nNow: produce a dialogue between two engineers (Li and Sarah). Each dialogue should be 5–7 turns.\nRequirements:\nUse **English-Mandarin code-switching** at the **phrase level** (switch after a technical word or phrase).\nMake sure at least one of the following trigger words appears in each dialogue: **deploy, build, bug, deadline, merge**.\nTarget ~35–50% Mandarin tokens per dialogue.\nUse natural engineering register (issue, PR, deploy, QA, staging, workaround).\nKeep turns short-to-medium.\nOutput only the dialogues. Do not repeat the few-shot examples."
  },
  {
    "id": 21,
    "utterance": "Chen: \"We noticed latency spikes post 11 PM.\"\nJohn: \"Yes, cache miss rate shot up.\"\nChen: \"这是昨晚 deploy 引起的.\"\nJohn: \"We should rollback the microservice quickly.\"\nChen: \"好的，我写 rollback plan 并通知 infra team.\"\nJohn: \"I'll add metrics for cache hits to monitor.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats. Pattern: inter-sentential."
  },
  {
    "id": 22,
    "utterance": "Chen: \"The timeout logic is fine, but circuit breaker 配置太 aggressive.\"\nJohn: \"It trips after two failures, 然后 cascading failure.\"\nChen: \"我们应该增加 threshold 并 implement exponential backoff.\"\nJohn: \"Right — client retries 是 sync, 改成 async 吧.\"\nChen: \"我 patch client lib 并更新文档.\"\nJohn: \"Perfect, benchmark 修复效果后 report.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats. Pattern: intra-sentential."
  },
  {
    "id": 23,
    "utterance": "Chen: \"This API contract 不兼容, 对吧?\"\nJohn: \"对，他们忘记加 version.\"\nChen: \"We need migration path, 是不是?\"\nJohn: \"Exactly — deprecate 老接口并加 shim.\"\nChen: \"我写 migration plan, dual writes 两个 sprint.\"\nJohn: \"联系 SDK team 确保 clients 不会 break.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats. Pattern: tag-switching."
  },
  {
    "id": 24,
    "utterance": "Chen: \"Payment service 出了 bug after tokenization.\"\nJohn: \"Token service fails 在某些卡类型.\"\nChen: \"先 hotfix, long-term 再重构.\"\nJohn: \"我加 logging 并启用 A/B test 验证.\"\nChen: \"Staging pass 後 we deploy to prod.\"\nJohn: \"好，我也准备 rollback steps 防万一.\"",
    "source": "LLM",
    "generation_strategy": ["Persona-based", "Chain of Thought"],
    "model_name": "GPT-4",
    "prompt_text": "You are a senior bilingual engineer (Chen) from Beijing working on distributed systems in Seattle. Write English-Mandarin code-switched dialogues reflecting real engineering reasoning. Chen is precise and professional but mixes Mandarin naturally in work chats. Pattern: lexical-borrowing + inter-sentential."
  },
  {
    "id": 25,
    "utterance": "A: \"I'll deploy 新 feature branch this afternoon.\"\nB: \"好，别忘了 to run smoke tests.\"\nA: \"If pass, 我就 tag release 并通知 QA.\"\nB: \"Perfect, 然后我们就 deploy.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: verb-boundary."
  },
  {
    "id": 26,
    "utterance": "A: \"这个 hotfix 只是权宜之计, but it avoids outage.\"\nB: \"是的，下个 sprint 我们必须 refactor 那部分.\"\nA: \"Agree, uptime first, refactor later.\"\nB: \"我加一个 ticket 把 refactor 细节写清楚.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: idiom + switch."
  },
  {
    "id": 27,
    "utterance": "A: \"If cache misses, fallback kicks in.\"\nB: \"如果 fallback 触发, increase timeout 并 log event.\"\nA: \"Add metrics 来衡量 hit rate.\"\nB: \"有了数据我们可以调整 policy.\"",
    "source": "LLM",
    "generation_strategy": ["Few-shot", "Chain-of-Thought"],
    "model_name": "GPT-4",
    "prompt_text": "Generate English-Mandarin code-switched dialogues showing micro-level switching patterns. Few-shot examples:\nTemplate 1 (verb-boundary):\nA: \"I'm going to deploy 新分支 right now.\"\nB: \"好，记得 to bump version number.\"\nTemplate 2 (idiom + switch):\nA: \"That fix is a 临时方案 but it works.\"\nB: \"是的，不过我们得做 proper refactor.\"\nTemplate 3 (phrasal-trigger):\nA: \"If cache misses, call fallback.\"\nB: \"如果这样, we'll log event 并且 alert team.\"\nPattern: phrasal-trigger."
  }
]

generated_examples = []

for item in prompts:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": item['prompt_text']}],
    )
    # The generated dialogue
    dialogue_text = response.choices[0].message.content
    print(dialogue_text)
    
    # Store new example
    generated_examples.append({
        "id": item['id'],
        "utterance": dialogue_text,
        "source": "LLM",
        "generation_strategy": item['generation_strategy'],
        "model_name": "GPT-4",
        "prompt_text": item['prompt_text']
    })

# Optionally save to JSON
with open("generated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(generated_examples, f, ensure_ascii=False, indent=2)

print(f"Generated {len(generated_examples)} new dialogues.")
