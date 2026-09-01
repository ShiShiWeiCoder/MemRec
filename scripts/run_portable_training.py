#!/usr/bin/env python3
"""Canonical portable launcher for one MemRec training run."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = {
    "mooccubex": ROOT / "data" / "MOOCCubeX" / "proc_data",
}


def build_command(args):
    data_dir = DATA_DIRS[args.dataset]
    if args.task == "rank":
        entry = "main_rank_multilevel_memory.py"
        script = ROOT / "RS" / "rank" / entry
        algorithm = args.algo or "DIN"
        metrics = "5,10"
    else:
        entry = "main_rerank_multilevel_memory.py"
        script = ROOT / "RS" / "rerank" / entry
        algorithm = args.algo or "DLCM"
        metrics = "1,3,5"

    command = [
        sys.executable,
        "-u",
        str(script),
        f"--data_dir={data_dir}",
        f"--task={args.task}",
        f"--algo={algorithm}",
        f"--epoch_num={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--lr={args.lr}",
        f"--seed={args.seed}",
        f"--metric_scope={metrics}",
        "--patience=3",
        f"--metrics_output={ROOT / 'results' / 'paper_metrics' / f'{args.dataset}_{args.task}_{args.mode}_{algorithm}_seed{args.seed}.json'}",
    ]
    if args.device != "auto":
        command.append(f"--device={args.device}")

    if args.mode == "memrec":
        prefix = "bert_newprompt"
        command.extend(
            [
                "--augment=true",
                f"--aug_prefix={prefix}",
                "--convert_type=MultilevelMemoryHEA",
                "--convert_arch=128,32",
                "--export_num=2",
                "--specific_export_num=3",
                "--memory_mode=true",
                "--enable_memory_attention=true",
                "--reflection_mode",
                "--fusion_mode=film",
            ]
        )
        analysis = data_dir / f"{prefix}.analysis"
        if analysis.is_file():
            command.append(f"--analysis_aug_file={analysis}")
        if (data_dir / "enhanced_gating_features.json").is_file():
            command.append("--enhanced_gating")
        if (data_dir / "transition_features.json").is_file():
            command.append("--transition_feature_dim=9")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATA_DIRS, default="mooccubex")
    parser.add_argument("--task", choices=["rank", "rerank"], default="rank")
    parser.add_argument("--mode", choices=["memrec"], default="memrec")
    parser.add_argument("--algo", help="Default: DIN for Rank, DLCM for Rerank")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", default="1e-3")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(args)
    print("Command:")
    print(" ".join(str(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
