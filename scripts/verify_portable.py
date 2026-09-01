#!/usr/bin/env python3
"""Validate a MemRec migration bundle without loading the full datasets."""

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "torch": "torch",
    "scikit-learn": "sklearn",
    "transformers": "transformers",
    "tqdm": "tqdm",
    "requests": "requests",
    "pandas": "pandas",
}

DATASETS = {
    "mooccubex": ROOT / "data" / "MOOCCubeX" / "proc_data",
    "mooccube": ROOT / "data" / "MOOCCube" / "proc_data",
    "coursera": ROOT / "data" / "coursera" / "proc_data",
}

BASE_FILES = [
    "stat.json",
    "sequential_data.json",
    "item2attributes.json",
    "datamaps.json",
    "rank.train",
    "rank.test",
    "rerank.train",
    "rerank.test",
]

AUGMENT_FILES = [
    "bert_newprompt.hist",
    "bert_newprompt.item",
    "bert_newprompt.analysis",
    "causal_multilevel_memory.json",
    "enhanced_gating_features.json",
    "transition_features.json",
    "sequential_timestamps.json",
    "memory_partition_config.json",
]


def human_size(size):
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def check_syntax():
    failures = []
    checked = 0
    excluded = {".git", "worktrees", "__pycache__", "dist"}
    for path in ROOT.rglob("*.py"):
        relative_path = path.relative_to(ROOT)
        if any(part in excluded for part in relative_path.parts):
            continue
        checked += 1
        try:
            compile(path.read_bytes(), str(path), "exec")
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    return checked, failures


def check_packages():
    missing = []
    for display_name, module_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(display_name)
    return missing


def check_dataset(name, proc_dir):
    result = {"name": name, "path": proc_dir, "missing": [], "details": [], "warnings": []}
    for filename in BASE_FILES + AUGMENT_FILES:
        path = proc_dir / filename
        if not path.is_file():
            result["missing"].append(filename)

    stat_path = proc_dir / "stat.json"
    if stat_path.is_file():
        with stat_path.open(encoding="utf-8") as handle:
            stats = json.load(handle)
        result["details"].append(
            f"items={stats.get('item_num', '?')}, attributes={stats.get('attribute_num', '?')}"
        )

    for split in ("rank.train", "rank.test", "rerank.train", "rerank.test"):
        path = proc_dir / split
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        candidate_counts = [len(sample[2]) for sample in samples]
        candidate_min = min(candidate_counts, default=0)
        candidate_max = max(candidate_counts, default=0)
        result["details"].append(
            f"{split}: samples={len(samples)}, candidates={candidate_min}-{candidate_max}, "
            f"size={human_size(path.stat().st_size)}"
        )
        if not samples:
            result["warnings"].append(f"{split} is empty")
        elif candidate_min != candidate_max:
            result["warnings"].append(
                f"{split} has variable candidate lengths ({candidate_min}-{candidate_max}); loader will pad them"
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    args = parser.parse_args()

    print(f"MemRec root: {ROOT}")
    checked, syntax_failures = check_syntax()
    print(f"Python syntax: checked={checked}, failures={len(syntax_failures)}")
    for failure in syntax_failures:
        print(f"  ERROR {failure}")

    missing_packages = check_packages()
    if missing_packages:
        print("Missing Python packages: " + ", ".join(missing_packages))
        print("Install them with: python -m pip install -r requirements.txt")
    else:
        print("Python packages: OK")

    selected = DATASETS.items() if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]
    data_failures = 0
    for name, proc_dir in selected:
        result = check_dataset(name, proc_dir)
        print(f"Dataset {name}: {proc_dir}")
        for detail in result["details"]:
            print(f"  {detail}")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")
        if result["missing"]:
            data_failures += 1
            print("  Missing: " + ", ".join(result["missing"]))
        else:
            print("  Required training files: OK")

    if syntax_failures or data_failures:
        return 1
    if missing_packages:
        return 2
    print("Portable bundle verification: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
