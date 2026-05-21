import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(texts, tokenizer, model, device, batch_size=16):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        vecs = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
        all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


def stable_text_hash(text):
    return str(abs(hash(text)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_dir", required=True, help="Directory containing memory_analysis.klg")
    parser.add_argument("--output_path", required=True, help="Output embedding JSON path")
    parser.add_argument("--model_path", default="bert-base-uncased", help="Local or Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    klg_path = os.path.join(args.knowledge_dir, "memory_analysis.klg")
    with open(klg_path, encoding="utf-8") as f:
        klg = json.load(f)

    existing = {}
    hashes = {}
    hash_path = args.output_path + ".hashes.json"
    if os.path.exists(args.output_path) and os.path.exists(hash_path):
        with open(args.output_path, encoding="utf-8") as f:
            existing = json.load(f)
        with open(hash_path, encoding="utf-8") as f:
            hashes = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    to_encode_ids, to_encode_texts = [], []
    for sample_id, entry in klg.items():
        text = entry.get("ans", "")
        text_hash = stable_text_hash(text)
        if sample_id in existing and hashes.get(sample_id) == text_hash:
            continue
        to_encode_ids.append(sample_id)
        to_encode_texts.append(text if text else " ")
        hashes[sample_id] = text_hash

    results = dict(existing)
    if to_encode_ids:
        vecs = encode_texts(to_encode_texts, tokenizer, model, device, args.batch_size)
        for sample_id, vec in tqdm(zip(to_encode_ids, vecs), total=len(to_encode_ids), desc="Writing"):
            results[sample_id] = vec.tolist()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f)

    dim = len(next(iter(results.values()))) if results else 0
    print(f"Encoded {len(results)} entries with dim={dim}")


if __name__ == "__main__":
    main()
