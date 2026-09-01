import os
import json
import time
import requests
import re
from tqdm import tqdm

# ---------------------------
# API 配置
# ---------------------------
API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
REQUEST_DELAY_SECONDS = float(os.getenv("OLLAMA_REQUEST_DELAY_SECONDS", "0.1"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))

# 模型调用参数（根据需要调整）
API_PARAMS = {
    "max_tokens": 4096,  # 最大生成 token 数（可根据实际情况调整）
    "temperature": 0.7,  # 温度控制采样随机性
    "top_p": 0.9,  # nucleus 采样参数
    "stream": False  # 不使用流式输出
}

# ---------------------------
# 目录设置
# ---------------------------
PROC_DATA_DIR = "data/MOOCCubeX/proc_data"
OUTPUT_DIR = "data/MOOCCubeX/knowledge_multilevel_memory"  # 多级记忆增强知识输出目录


# ---------------------------
# 辅助函数：调用 LLM API（Ollama generate 接口）
# ---------------------------
def call_llm(prompt: str) -> str:
    """
    调用本地 Ollama 模型接口，生成回复文本。
    使用 Ollama 的 /api/generate 接口格式。
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,  # Ollama 使用 prompt 而不是 messages
        "stream": False,   # 不使用流式输出
        **API_PARAMS
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        result = response.json()
        # Ollama 的返回格式：{"response": "生成的文本", "done": true, ...}
        generated_text = result.get("response", "")
        return generated_text.strip()
    except requests.exceptions.Timeout:
        print(f"Timeout when calling API for prompt: {prompt[:50]}...")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Request error when calling API for prompt: {prompt[:50]}..., error: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON decode error when calling API for prompt: {prompt[:50]}..., error: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error when calling API for prompt: {prompt[:50]}..., error: {e}")
        return ""


# ---------------------------
# 辅助函数：去除生成答案中的推理部分
# ---------------------------
def remove_thinking(text: str) -> str:
    """
    利用正则表达式去除文本中形如 <think> ... </think> 块（包括换行符）。
    例如：如果生成文本中存在 "<think>\n ... </think>\n\n"，则将其删除，保留其前后的内容。
    """
    # 使用 DOTALL 模式匹配所有字符，包括换行符
    cleaned_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def load_existing_knowledge(file_path: str) -> dict:
    """加载历史知识文件用于增量复用。"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: failed to load existing knowledge from {file_path}: {e}")
        return {}


def generate_knowledge(prompts: dict, existing_knowledge: dict, desc: str, id_name: str):
    """按 prompt 增量生成知识：相同prompt直接复用，变化prompt才重算。"""
    output = {}
    reused_cnt = 0
    generated_cnt = 0
    failed_cnt = 0

    for sample_id, prompt in tqdm(prompts.items(), desc=desc):
        old = existing_knowledge.get(sample_id, {})
        old_prompt = old.get("prompt")
        old_ans = old.get("ans", "")

        # prompt 未变化且已有有效答案时，直接复用
        if old_prompt == prompt and old_ans:
            output[sample_id] = old
            reused_cnt += 1
            continue

        generated_text = call_llm(prompt)
        if generated_text:
            ans_text = remove_thinking(generated_text)
            output[sample_id] = {"prompt": prompt, "ans": ans_text}
            generated_cnt += 1
        else:
            print(f"Failed to generate {id_name} knowledge for {sample_id}")
            output[sample_id] = {"prompt": prompt, "ans": ""}
            failed_cnt += 1

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    print(
        f"{desc} finished: total={len(prompts)}, reused={reused_cnt}, "
        f"generated={generated_cnt}, failed={failed_cnt}, delay={REQUEST_DELAY_SECONDS}s"
    )
    return output


# ---------------------------
# 主函数：生成多级记忆增强的用户与物品知识
# ---------------------------
def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载多级记忆增强的 prompt 文件
    prompt_hist_multilevel_memory_path = os.path.join(PROC_DATA_DIR, "prompt.hist.multilevel_memory")
    prompt_item_multilevel_memory_path = os.path.join(PROC_DATA_DIR, "prompt.item.multilevel_memory")
    prompt_memory_analysis_path = os.path.join(PROC_DATA_DIR, "prompt.memory_analysis")

    print("Loading multilevel memory enhanced prompt files ...")
    try:
        with open(prompt_hist_multilevel_memory_path, "r", encoding="utf-8") as f:
            user_prompts_multilevel_memory = json.load(f)
        with open(prompt_item_multilevel_memory_path, "r", encoding="utf-8") as f:
            item_prompts_multilevel_memory = json.load(f)
        with open(prompt_memory_analysis_path, "r", encoding="utf-8") as f:
            memory_analysis_prompts = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Multilevel memory prompt file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in multilevel memory prompt file: {e}")
        return

    print(f"Loaded {len(user_prompts_multilevel_memory)} multilevel memory user prompts, "
          f"{len(item_prompts_multilevel_memory)} multilevel memory item prompts, "
          f"and {len(memory_analysis_prompts)} memory analysis prompts")

    user_knowledge_multilevel_memory_path = os.path.join(OUTPUT_DIR, "user_multilevel_memory.klg")
    item_knowledge_multilevel_memory_path = os.path.join(OUTPUT_DIR, "item_multilevel_memory.klg")
    memory_analysis_knowledge_path = os.path.join(OUTPUT_DIR, "memory_analysis.klg")

    # 读取历史结果，后续按prompt进行增量复用
    existing_user_knowledge = load_existing_knowledge(user_knowledge_multilevel_memory_path)
    existing_item_knowledge = load_existing_knowledge(item_knowledge_multilevel_memory_path)
    existing_analysis_knowledge = load_existing_knowledge(memory_analysis_knowledge_path)

    # ---------------------------
    # 生成多级记忆增强的用户知识（User Knowledge with Multilevel Memory）
    # ---------------------------
    print("Generating multilevel memory enhanced user knowledge ...")
    user_knowledge_multilevel_memory = generate_knowledge(
        user_prompts_multilevel_memory,
        existing_user_knowledge,
        "Multilevel Memory User Prompt",
        "user",
    )

    # 保存多级记忆增强用户知识文件
    with open(user_knowledge_multilevel_memory_path, "w", encoding="utf-8") as f:
        json.dump(user_knowledge_multilevel_memory, f, ensure_ascii=False, indent=2)
    print(f"Multilevel memory user knowledge saved to: {user_knowledge_multilevel_memory_path}")

    # ---------------------------
    # 生成多级记忆增强的物品知识（Item Knowledge with Multilevel Memory）
    # ---------------------------
    print("Generating multilevel memory enhanced item knowledge ...")
    item_knowledge_multilevel_memory = generate_knowledge(
        item_prompts_multilevel_memory,
        existing_item_knowledge,
        "Multilevel Memory Item Prompt",
        "item",
    )

    # 保存多级记忆增强物品知识文件
    with open(item_knowledge_multilevel_memory_path, "w", encoding="utf-8") as f:
        json.dump(item_knowledge_multilevel_memory, f, ensure_ascii=False, indent=2)
    print(f"Multilevel memory item knowledge saved to: {item_knowledge_multilevel_memory_path}")

    # ---------------------------
    # 生成多级记忆分析知识（Multilevel Memory Analysis Knowledge）
    # ---------------------------
    print("Generating multilevel memory analysis knowledge ...")
    memory_analysis_knowledge = generate_knowledge(
        memory_analysis_prompts,
        existing_analysis_knowledge,
        "Memory Analysis Prompt",
        "analysis",
    )

    # 保存多级记忆分析知识文件
    with open(memory_analysis_knowledge_path, "w", encoding="utf-8") as f:
        json.dump(memory_analysis_knowledge, f, ensure_ascii=False, indent=2)
    print(f"Multilevel memory analysis knowledge saved to: {memory_analysis_knowledge_path}")

    # 打印统计信息
    successful_users_multilevel_memory = sum(1 for v in user_knowledge_multilevel_memory.values() if v["ans"])
    successful_items_multilevel_memory = sum(1 for v in item_knowledge_multilevel_memory.values() if v["ans"])
    successful_analysis = sum(1 for v in memory_analysis_knowledge.values() if v["ans"])

    print("\n" + "="*60)
    print("🎉 多级记忆增强知识生成完成！")
    print("="*60)
    print(f"✅ 用户多级记忆知识: {successful_users_multilevel_memory}/{len(user_prompts_multilevel_memory)} 成功生成")
    print(f"✅ 课程多级记忆知识: {successful_items_multilevel_memory}/{len(item_prompts_multilevel_memory)} 成功生成")
    print(f"✅ 多级记忆分析知识: {successful_analysis}/{len(memory_analysis_prompts)} 成功生成")
    print("\n生成的知识文件:")
    print(f"  📁 {user_knowledge_multilevel_memory_path}")
    print(f"  📁 {item_knowledge_multilevel_memory_path}")
    print(f"  📁 {memory_analysis_knowledge_path}")
    print("\n🧠 认知心理学框架:")
    print("  🔥 感觉记忆 (Sensory Memory): 即时需求和最近浏览")
    print("  ⚡ 工作记忆 (Working Memory): 当前会话行为模式")
    print("  🏗️ 长期记忆 (Long-Term Memory): 职业发展方向")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    if args.proc_dir:
        PROC_DATA_DIR = args.proc_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    main()


# nohup python knowledge-generating/llm_generating_multilevel_memory.py > knowledge-generating/llm_generating_multilevel_memory.log 2>&1 &
# tail -f knowledge-generating/llm_generating_multilevel_memory.log
