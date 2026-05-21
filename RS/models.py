import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from layers import AttentionPoolingLayer, ConvertNet, MLP


def tau_function(x):
    return torch.sigmoid(x)


def attention_score(x, temperature=1.0):
    return torch.softmax(x / temperature, dim=-1)


class BaseModel(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.embed_dim = getattr(args, "embed_dim", 32)
        self.item_embedding = nn.Embedding(dataset.item_num + 1, self.embed_dim, padding_idx=0)
        self.attr_embedding = nn.Embedding(dataset.attr_num + 1, self.embed_dim, padding_idx=0)
        self.loss_fn = BCEWithLogitsLoss()

        self.use_augment = str(getattr(args, "augment", "false")).lower() == "true"
        self.aug_dim = getattr(dataset, "aug_vec_dim", 0)
        if self.use_augment and self.aug_dim > 0:
            self.convert = ConvertNet(args, self.aug_dim, getattr(args, "convert_dropout", 0.0), getattr(args, "convert_type", "MultilevelMemoryHEA"))
        else:
            self.convert = None

    def process_input(self, inp):
        if "iid" in inp:
            item_emb = self.item_embedding(inp["iid"])
            attr_emb = self.attr_embedding(inp["aid"])
            return item_emb + attr_emb
        item_emb = self.item_embedding(inp["iid_list"])
        attr_emb = self.attr_embedding(inp["aid_list"])
        return item_emb + attr_emb

    def _history_embedding(self, inp):
        hist_item = self.item_embedding(inp["hist_iid_seq"])
        hist_attr = self.attr_embedding(inp["hist_aid_seq"])
        hist = hist_item + hist_attr
        mask = (inp["hist_iid_seq"] > 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hist * mask).sum(dim=1) / denom

    def _augment_vector(self, inp):
        if self.convert is None:
            return None
        if "item_aug_vec" in inp:
            return self.convert([inp["hist_aug_vec"], inp["item_aug_vec"]])
        if "item_aug_vec_list" in inp:
            item_vec = torch.stack(inp["item_aug_vec_list"], dim=1)
            hist_vec = inp["hist_aug_vec"].unsqueeze(1).expand_as(item_vec)
            flat = self.convert([hist_vec.reshape(-1, hist_vec.size(-1)), item_vec.reshape(-1, item_vec.size(-1))])
            return flat.reshape(item_vec.size(0), item_vec.size(1), -1)
        return None

    def get_ctr_output(self, logits, labels=None):
        if labels is None:
            return {"logits": logits, "probs": torch.sigmoid(logits)}
        return {"loss": self.loss_fn(logits.view(-1), labels.float().view(-1)), "logits": logits}

    def get_rerank_output(self, logits, labels=None):
        if labels is None:
            return {"logits": logits, "probs": torch.sigmoid(logits)}
        return {"loss": self.loss_fn(logits.view(-1), labels.float().view(-1)), "logits": logits}


class DeepFM(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        input_dim = self.embed_dim * 2 + (getattr(dataset, "aug_vec_dim", 0) if self.use_augment else 0)
        self.mlp = MLP(getattr(args, "deepfm_deep_arch", [200, 80]), input_dim, getattr(args, "dropout", 0.0))
        self.out = nn.Linear(self.mlp.output_dim, 1)

    def forward(self, inp):
        hist = self._history_embedding(inp)
        item = self.process_input(inp)
        if item.dim() == 3:
            hist = hist.unsqueeze(1).expand_as(item)
            features = [hist, item]
            aug = self._augment_vector(inp)
            if aug is not None:
                features.append(aug)
            logits = self.out(self.mlp(torch.cat(features, dim=-1))).squeeze(-1)
            return self.get_rerank_output(logits, inp.get("lb_list"))

        features = [hist, item]
        aug = self._augment_vector(inp)
        if aug is not None:
            features.append(aug)
        logits = self.out(self.mlp(torch.cat(features, dim=-1))).squeeze(-1)
        return self.get_ctr_output(logits, inp.get("lb"))


class DeepInterestNet(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.attn = AttentionPoolingLayer(self.embed_dim, getattr(args, "dropout", 0.0))
        self.mlp = MLP(getattr(args, "final_mlp_arch", [200, 80]), self.embed_dim * 2, getattr(args, "dropout", 0.0))
        self.out = nn.Linear(self.mlp.output_dim, 1)

    def forward(self, inp):
        item = self.process_input(inp)
        if item.dim() == 3:
            hist = self._history_embedding(inp).unsqueeze(1).expand_as(item)
            logits = self.out(self.mlp(torch.cat([hist, item], dim=-1))).squeeze(-1)
            return self.get_rerank_output(logits, inp.get("lb_list"))
        hist = self._history_embedding(inp)
        logits = self.out(self.mlp(torch.cat([hist, item], dim=-1))).squeeze(-1)
        return self.get_ctr_output(logits, inp.get("lb"))


class DLCM(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.encoder = nn.GRU(self.embed_dim, getattr(args, "hidden_size", 64), batch_first=True)
        self.out = nn.Linear(getattr(args, "hidden_size", 64), 1)

    def forward(self, inp):
        item = self.process_input(inp)
        encoded, _ = self.encoder(item)
        logits = self.out(encoded).squeeze(-1)
        return self.get_rerank_output(logits, inp.get("lb_list"))


class PRM(DLCM):
    pass


class SetRank(DLCM):
    pass


class MIR(DLCM):
    pass


class GSF(DLCM):
    pass


class EGRerank(DLCM):
    pass


class LambdaRank(DLCM):
    pass


class RankFormer(DLCM):
    pass


class PEAR(DLCM):
    pass


class PIER(DLCM):
    pass


class xDeepFM(DeepFM):
    pass


class DIEN(DeepInterestNet):
    pass


class DCN(DeepFM):
    pass


class FiBiNet(DeepFM):
    pass


class FiGNN(DeepFM):
    pass


class AutoInt(DeepFM):
    pass


class GRU4Rec(DeepFM):
    pass


class Caser(DeepFM):
    pass


class SASRec(DeepFM):
    pass


class BERT4Rec(DeepFM):
    pass


class NARM(DeepFM):
    pass


class FMLPRec(DeepFM):
    pass


class BSARec(DeepFM):
    pass
