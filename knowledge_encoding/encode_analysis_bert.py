"""Encode all three MemRec knowledge streams with one BERT checkpoint."""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from tqdm import tqdm


BERT_PATH = os.getenv("MEMREC_BERT_PATH", "")
KNOWLEDGE_STREAMS = {
    "hist": "user_multilevel_memory.klg",
    "item": "item_multilevel_memory.klg",
    "analysis": "memory_analysis.klg",
}


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(texts, tokenizer, model, device, batch_size=32):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        vectors = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
        all_vectors.append(vectors.cpu().float().numpy())
    return np.concatenate(all_vectors, axis=0)


def load_stream(path, stream):
    with open(path, encoding="utf-8") as handle:
        knowledge = json.load(handle)
    texts = {}
    missing_answers = []
    for key, entry in knowledge.items():
        answer = entry.get("ans", "").strip()
        if not answer:
            missing_answers.append(str(key))
            continue
        texts[str(key)] = answer
    if missing_answers:
        raise ValueError(
            f"{stream} knowledge contains {len(missing_answers)} empty answers; "
            f"example={missing_answers[0]}. Regenerate failed LLM outputs first."
        )
    return texts


def encode_stream(text_by_id, output_path, tokenizer, model, device, batch_size):
    existing = {}
    hashes = {}
    hash_path = output_path + ".hashes.json"
    if os.path.exists(output_path) and os.path.exists(hash_path):
        with open(output_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        with open(hash_path, encoding="utf-8") as handle:
            hashes = json.load(handle)

    current_hashes = {
        key: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for key, text in text_by_id.items()
    }
    changed_ids = [
        key
        for key in text_by_id
        if key not in existing or hashes.get(key) != current_hashes[key]
    ]
    results = {key: existing[key] for key in text_by_id if key in existing}
    if changed_ids:
        vectors = encode_texts(
            [text_by_id[key] for key in changed_ids],
            tokenizer,
            model,
            device,
            batch_size=batch_size,
        )
        for key, vector in tqdm(
            zip(changed_ids, vectors), total=len(changed_ids), desc=os.path.basename(output_path)
        ):
            results[key] = vector.tolist()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
    with open(hash_path, "w", encoding="utf-8") as handle:
        json.dump(current_hashes, handle)
    dimension = len(next(iter(results.values()))) if results else 0
    print(
        f"{output_path}: total={len(results)}, changed={len(changed_ids)}, "
        f"reused={len(results) - len(changed_ids)}, dim={dimension}"
    )


def select_jobs(args):
    if args.output_path:
        if args.stream != "analysis":
            raise ValueError("--output_path is only supported with --stream analysis")
        return [("analysis", args.output_path)]
    if not args.data_dir:
        raise ValueError("--data_dir is required when encoding the complete pipeline")
    streams = KNOWLEDGE_STREAMS if args.stream == "all" else {args.stream: KNOWLEDGE_STREAMS[args.stream]}
    return [
        (stream, os.path.join(args.data_dir, f"{args.output_prefix}.{stream}"))
        for stream in streams
    ]


def main():
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge_dir", required=True, help="directory containing the three .klg files")
    parser.add_argument("--data_dir", help="processed-data output directory")
    parser.add_argument("--output_prefix", default="bert_newprompt")
    parser.add_argument("--stream", choices=["all", *KNOWLEDGE_STREAMS], default="all")
    parser.add_argument("--output_path", help="legacy analysis-only output path")
    parser.add_argument("--model_path", default=None, help="local path or Hugging Face checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--offline", action="store_true", help="only use local or cached model files")
    args = parser.parse_args()

    jobs = select_jobs(args)
    inputs = []
    for stream, output_path in jobs:
        knowledge_path = os.path.join(args.knowledge_dir, KNOWLEDGE_STREAMS[stream])
        if not os.path.isfile(knowledge_path):
            raise FileNotFoundError(
                f"missing {stream} knowledge file: {knowledge_path}. "
                "Run the three-stream LLM generation step first."
            )
        inputs.append((stream, output_path, load_stream(knowledge_path, stream)))

    checkpoint = args.model_path or BERT_PATH or "bert-base-uncased"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading encoder: {checkpoint}, device={device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=args.offline)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=args.offline)
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    for stream, output_path, texts in inputs:
        print(f"Encoding {stream}: {len(texts)} records")
        encode_stream(texts, output_path, tokenizer, model, device, args.batch_size)


if __name__ == "__main__":
    main()
