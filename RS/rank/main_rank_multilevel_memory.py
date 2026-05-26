import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.utils.data as Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import AmzDataset
from models import AutoInt, BERT4Rec, BSARec, Caser, DCN, DIEN, DeepFM, DeepInterestNet, FMLPRec, FiBiNet, FiGNN, GRU4Rec, NARM, SASRec, xDeepFM


def str2list(value):
    if isinstance(value, list):
        return value
    return [int(x) for x in str(value).split(",") if x]


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class MultilevelMemoryDataset(AmzDataset):
    def __init__(self, data_path, set="train", task="rerank", max_hist_len=10, augment=False, aug_prefix=None, memory_mode=True, data_file=None, enhanced_gating=False, reflection_mode=False, analysis_aug_file=None, analysis_as_expert=False, no_analysis=False):
        super().__init__(data_path, set, task, max_hist_len, augment, aug_prefix, data_file)
        self.memory_mode = memory_mode
        self.enhanced_gating = enhanced_gating
        self.reflection_mode = reflection_mode
        self.analysis_aug_file = analysis_aug_file
        self.no_analysis = no_analysis
        self.analysis_aug_data = {}
        if augment and not no_analysis:
            default_path = analysis_aug_file or os.path.join(data_path, f"{aug_prefix}.analysis")
            if os.path.exists(default_path):
                self.analysis_aug_data = load_json(default_path)

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if self.analysis_aug_data:
            uid, seq_idx = self.data[index][0], self.data[index][1]
            key = f"{uid}:{seq_idx}"
            out["analysis_vec"] = torch.tensor(self.analysis_aug_data.get(key, [])).float()
        return out


def eval_multilevel_memory(model, test_loader, metric_scope, is_rank=True, compute_auc=True):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            if "loss" in output:
                total_loss += float(output["loss"].item())
                total_batches += 1
    return {"loss": total_loss / max(1, total_batches)}


def load_model_multilevel_memory(args, dataset):
    model_map = {
        "DeepFM": DeepFM,
        "xDeepFM": xDeepFM,
        "DIN": DeepInterestNet,
        "DIEN": DIEN,
        "DCN": DCN,
        "FiBiNet": FiBiNet,
        "FiGNN": FiGNN,
        "AutoInt": AutoInt,
        "GRU4Rec": GRU4Rec,
        "Caser": Caser,
        "SASRec": SASRec,
        "BERT4Rec": BERT4Rec,
        "NARM": NARM,
        "FMLPRec": FMLPRec,
        "BSARec": BSARec,
    }
    if args.algo not in model_map:
        raise ValueError(f"Unsupported rank model: {args.algo}")
    return model_map[args.algo](args, dataset)


def get_optimizer_multilevel_memory(args, model, train_data_num):
    base_params = []
    memrec_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in ["convert", "film", "memory", "expert", "gate"]):
            memrec_params.append(param)
        else:
            base_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": memrec_params, "lr": args.memrec_lr},
        ],
        weight_decay=args.weight_decay,
    )


def train_multilevel_memory(args):
    setup_seed(args.seed)
    train_dataset = MultilevelMemoryDataset(
        args.data_dir,
        set="train",
        task=args.task,
        max_hist_len=args.max_hist_len,
        augment=str(args.augment).lower() == "true",
        aug_prefix=args.aug_prefix,
        memory_mode=str(args.memory_mode).lower() == "true",
        data_file="rank",
        enhanced_gating=args.enhanced_gating,
        reflection_mode=args.reflection_mode,
        analysis_aug_file=args.analysis_aug_file,
        no_analysis=args.no_analysis,
    )
    test_dataset = MultilevelMemoryDataset(
        args.data_dir,
        set="test",
        task=args.task,
        max_hist_len=args.max_hist_len,
        augment=str(args.augment).lower() == "true",
        aug_prefix=args.aug_prefix,
        memory_mode=str(args.memory_mode).lower() == "true",
        data_file="rank",
        enhanced_gating=args.enhanced_gating,
        reflection_mode=args.reflection_mode,
        analysis_aug_file=args.analysis_aug_file,
        no_analysis=args.no_analysis,
    )
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model_multilevel_memory(args, train_dataset).to(args.device)
    optimizer = get_optimizer_multilevel_memory(args, model, len(train_dataset))

    for epoch in range(args.epoch_num):
        model.train()
        start = time.time()
        for batch in train_loader:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            output["loss"].backward()
            optimizer.step()
        metrics = eval_multilevel_memory(model, test_loader, args.metric_scope)
        print(f"epoch={epoch + 1} loss={metrics['loss']:.6f} time={time.time() - start:.1f}s")


def parse_args_multilevel_memory():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/MOOCCubeX/proc_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--timestamp", default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--epoch_num", default=20, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--memrec_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--convert_dropout", default=0.0, type=float)
    parser.add_argument("--metric_scope", default="3,5,10")
    parser.add_argument("--task", default="rerank")
    parser.add_argument("--algo", default="DeepFM")
    parser.add_argument("--augment", default="true")
    parser.add_argument("--aug_prefix", default="bert_newprompt")
    parser.add_argument("--convert_type", default="MultilevelMemoryHEA")
    parser.add_argument("--max_hist_len", default=5, type=int)
    parser.add_argument("--embed_dim", default=32, type=int)
    parser.add_argument("--final_mlp_arch", default="200,80", type=str2list)
    parser.add_argument("--deepfm_deep_arch", default="200,80", type=str2list)
    parser.add_argument("--convert_arch", default="128,32", type=str2list)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--memory_mode", default="true")
    parser.add_argument("--enhanced_gating", action="store_true")
    parser.add_argument("--reflection_mode", action="store_true")
    parser.add_argument("--transition_feature_dim", default=9, type=int)
    parser.add_argument("--analysis_aug_file", default=None)
    parser.add_argument("--analysis_vec_dim", default=0, type=int)
    parser.add_argument("--no_analysis", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_multilevel_memory(parse_args_multilevel_memory())
