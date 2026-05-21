import argparse
import json
import os
import re
import time

import requests
from tqdm import tqdm


DEFAULT_API_URL = os.getenv("MEMREC_LLM_API_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL_NAME = os.getenv("MEMREC_LLM_MODEL", "llama3.1:8b")
REQUEST_DELAY_SECONDS = float(os.getenv("MEMREC_LLM_DELAY_SECONDS", "0.1"))


def call_llm(prompt: str, api_url: str, model_name: str, timeout: int = 120) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as exc:
        print(f"LLM request failed: {exc}")
        return ""
    except json.JSONDecodeError as exc:
        print(f"LLM response is not valid JSON: {exc}")
        return ""


def remove_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL).strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_knowledge(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}
    try:
        data = load_json(file_path)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_knowledge(prompts: dict, existing_knowledge: dict, desc: str, id_name: str, api_url: str, model_name: str):
    output = {}
    for sample_id, prompt in tqdm(prompts.items(), desc=desc):
        old = existing_knowledge.get(sample_id, {})
        if old.get("prompt") == prompt and old.get("ans"):
            output[sample_id] = old
            continue

        generated_text = call_llm(prompt, api_url, model_name)
        output[sample_id] = {
            "prompt": prompt,
            "ans": remove_thinking(generated_text) if generated_text else "",
        }
        if not generated_text:
            print(f"Failed to generate {id_name} knowledge for {sample_id}")
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
    return output


def main(proc_dir: str, output_dir: str, api_url: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)

    prompt_paths = {
        "user": os.path.join(proc_dir, "prompt.hist.multilevel_memory"),
        "item": os.path.join(proc_dir, "prompt.item.multilevel_memory"),
        "analysis": os.path.join(proc_dir, "prompt.memory_analysis"),
    }
    output_paths = {
        "user": os.path.join(output_dir, "user_multilevel_memory.klg"),
        "item": os.path.join(output_dir, "item_multilevel_memory.klg"),
        "analysis": os.path.join(output_dir, "memory_analysis.klg"),
    }

    for kind in ("user", "item", "analysis"):
        prompts = load_json(prompt_paths[kind])
        existing = load_existing_knowledge(output_paths[kind])
        result = generate_knowledge(
            prompts,
            existing,
            desc=f"Generating {kind} memory knowledge",
            id_name=kind,
            api_url=api_url,
            model_name=model_name,
        )
        save_json(result, output_paths[kind])
        success = sum(1 for entry in result.values() if entry.get("ans"))
        print(f"{kind} knowledge saved to {output_paths[kind]} ({success}/{len(result)} non-empty)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", default="data/MOOCCubeX/proc_data")
    parser.add_argument("--output_dir", default="data/MOOCCubeX/knowledge_multilevel_memory")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    main(args.proc_dir, args.output_dir, args.api_url, args.model_name)
