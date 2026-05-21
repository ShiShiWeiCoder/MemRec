import torch
import torch.utils.data as Data


def load_json(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


class AmzDataset(Data.Dataset):
    def __init__(self, data_path, set="train", task="ctr", max_hist_len=10, augment=False, aug_prefix=None, data_file=None):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set

        file_name = data_file if data_file is not None else task
        self.data = load_pickle(f"{data_path}/{file_name}.{set}")
        self.stat = load_json(f"{data_path}/stat.json")
        self.item_num = self.stat["item_num"]
        self.attr_num = self.stat["attribute_num"]
        self.attr_ft_num = self.stat["attribute_ft_num"]
        self.rating_num = self.stat["rating_num"]
        self.dense_dim = self.stat["dense_dim"]
        self.length = len(self.data)
        self.sequential_data = load_json(f"{data_path}/sequential_data.json")
        self.item2attribution = load_json(f"{data_path}/item2attributes.json")

        if task == "rerank":
            self.max_list_len = self.stat.get("rerank_list_len", 10)
            if data_file == "rank" and self.data:
                self.max_list_len = len(self.data[0][2])

        self.aug_vec_dim = 0
        if augment:
            self.hist_aug_data = load_json(f"{data_path}/{aug_prefix}.hist")
            self.item_aug_data = load_json(f"{data_path}/{aug_prefix}.item")
            sample_key = next(iter(self.item_aug_data))
            self.aug_vec_dim = len(self.item_aug_data[sample_key])

    def __len__(self):
        return self.length

    def _first_attribute(self, attr):
        if self.attr_ft_num == 1 and isinstance(attr, list):
            return attr[0] if attr else 0
        return attr

    def _history(self, uid, seq_idx):
        item_seq, rating_seq = self.sequential_data[str(uid)]
        start = max(0, seq_idx - self.max_hist_len)
        hist_items = item_seq[start:seq_idx]
        hist_ratings = rating_seq[start:seq_idx]
        hist_attrs = [self._first_attribute(self.item2attribution[str(item)]) for item in hist_items]
        return hist_items, hist_attrs, hist_ratings, len(hist_items)

    def __getitem__(self, index):
        if self.task == "ctr":
            uid, seq_idx, label = self.data[index]
            item_seq, _ = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_items, hist_attrs, hist_ratings, hist_len = self._history(uid, seq_idx)
            out = {
                "iid": torch.tensor(iid).long(),
                "aid": torch.tensor(self._first_attribute(self.item2attribution[str(iid)])).long(),
                "lb": torch.tensor(label).long(),
                "hist_iid_seq": torch.tensor(hist_items).long(),
                "hist_aid_seq": torch.tensor(hist_attrs).long(),
                "hist_rate_seq": torch.tensor(hist_ratings).long(),
                "hist_seq_len": torch.tensor(hist_len).long(),
            }
            if self.augment:
                out["item_aug_vec"] = torch.tensor(self.item_aug_data.get(str(iid), [0.0] * self.aug_vec_dim)).float()
                out["hist_aug_vec"] = torch.tensor(self.hist_aug_data.get(f"{uid}:{seq_idx}", [0.0] * self.aug_vec_dim)).float()
            return out

        if self.task == "rerank":
            uid, seq_idx, candidates, labels = self.data[index]
            actual_len = len(candidates)
            pad_len = max(0, self.max_list_len - actual_len)
            padded_candidates = candidates + [0] * pad_len
            padded_labels = labels + [0] * pad_len
            candidate_attrs = [
                self._first_attribute(self.item2attribution.get(str(item), [0]))
                for item in padded_candidates
            ]
            hist_items, hist_attrs, hist_ratings, hist_len = self._history(uid, seq_idx)
            out = {
                "iid_list": torch.tensor(padded_candidates).long(),
                "aid_list": torch.tensor(candidate_attrs).long(),
                "lb_list": torch.tensor(padded_labels).long(),
                "hist_iid_seq": torch.tensor(hist_items).long(),
                "hist_aid_seq": torch.tensor(hist_attrs).long(),
                "hist_rate_seq": torch.tensor(hist_ratings).long(),
                "hist_seq_len": torch.tensor(hist_len).long(),
                "list_len": torch.tensor(actual_len).long(),
            }
            if self.augment:
                item_vecs = [
                    torch.tensor(self.item_aug_data.get(str(item), [0.0] * self.aug_vec_dim)).float()
                    if pos < actual_len and item != 0
                    else torch.tensor([0.0] * self.aug_vec_dim).float()
                    for pos, item in enumerate(padded_candidates)
                ]
                out["item_aug_vec_list"] = item_vecs
                out["hist_aug_vec"] = torch.tensor(self.hist_aug_data.get(f"{uid}:{seq_idx}", [0.0] * self.aug_vec_dim)).float()
            return out

        raise NotImplementedError(f"Unsupported task: {self.task}")
