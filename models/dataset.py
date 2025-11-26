import torch
import torch.utils.data as Data
import pickle
import numpy as np
from models.utils import load_json, load_pickle


class MemoryDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='rerank', max_hist_len=10, augment=False, aug_prefix=None, data_file=None):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set
        # data_file参数用于Rank粗排：加载rank数据但task设为rerank（Rank和Rerank模型逻辑相同）
        file_name = data_file if data_file is not None else task
        self.data = load_pickle(data_path + f'/{file_name}.{set}')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            # 根据实际加载的数据文件确定列表长度
            if data_file == 'rank':
                # Rank粗排：从实际数据中动态获取候选数量
                if len(self.data) > 0:
                    sample_candidates = self.data[0][2]  # [uid, seq_idx, candidates, labels]
                    self.max_list_len = len(sample_candidates)
                    print(f'Rank粗排候选数量（动态检测）: {self.max_list_len}')
                else:
                    self.max_list_len = 50  # 默认值
            else:
                # Rerank精排：使用stat.json中的配置
                self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        
        # 初始化缺失计数器
        self.missing_item_aug_count = 0
        self.missing_hist_aug_count = 0
        
        if augment:
            print(f"Loading augment data for {set} set...")
            self.hist_aug_data = load_json(data_path + f'/{aug_prefix}.hist')
            self.item_aug_data = load_json(data_path + f'/{aug_prefix}.item')
            # 获取增强向量的维度，用于创建默认零向量
            sample_key = list(self.item_aug_data.keys())[0]
            self.aug_vec_dim = len(self.item_aug_data[sample_key])
            print(f'Augment vector dimension: {self.aug_vec_dim}')
            print(f'Item augment data size: {len(self.item_aug_data)}, Hist augment data size: {len(self.hist_aug_data)}')
        else:
            self.aug_vec_dim = 0
        
        print(f"Dataset {set} initialized: {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        try:
            if self.task == 'rerank':
                uid, seq_idx, candidates, candidate_lbs = self.data[_id]
                candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
                item_seq, rating_seq = self.sequential_data[str(uid)]
                hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
                hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
                hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
                hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
                
                # 处理候选项属性和历史属性
                if self.attr_ft_num == 1:
                    processed_candidates_attr = []
                    for attr_list in candidates_attr:
                        if isinstance(attr_list, list) and len(attr_list) > 0:
                            processed_candidates_attr.append(attr_list[0])
                        else:
                            processed_candidates_attr.append(attr_list if not isinstance(attr_list, list) else 0)
                    
                    processed_hist_attri_seq = []
                    for attr_list in hist_attri_seq:
                        if isinstance(attr_list, list) and len(attr_list) > 0:
                            processed_hist_attri_seq.append(attr_list[0])
                        else:
                            processed_hist_attri_seq.append(attr_list if not isinstance(attr_list, list) else 0)
                else:
                    processed_candidates_attr = candidates_attr
                    processed_hist_attri_seq = hist_attri_seq
                
                # 对候选项进行 padding 到 max_list_len（确保所有样本的候选项数量一致）
                actual_list_len = len(candidates)
                if actual_list_len < self.max_list_len:
                    # Pad with 0 (padding token)
                    pad_len = self.max_list_len - actual_list_len
                    candidates = candidates + [0] * pad_len
                    processed_candidates_attr = processed_candidates_attr + [0] * pad_len
                    candidate_lbs = candidate_lbs + [0] * pad_len
                
                out_dict = {
                    'iid_list': torch.tensor(candidates).long(),
                    'aid_list': torch.tensor(processed_candidates_attr).long(),
                    'lb_list': torch.tensor(candidate_lbs).long(),
                    'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                    'hist_aid_seq': torch.tensor(processed_hist_attri_seq).long(),
                    'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                    'hist_seq_len': torch.tensor(hist_seq_len).long(),
                    'list_len': torch.tensor(actual_list_len).long()  # 记录实际的候选项数量
                }
                if self.augment:
                    item_aug_vec = []
                    for i, idx in enumerate(candidates):
                        # 对于 padding 的项目（idx=0 或超出实际长度），使用零向量
                        if i >= actual_list_len or idx == 0:
                            item_aug_vec.append(torch.tensor([0.0] * self.aug_vec_dim).float())
                        else:
                            item_key = str(idx)
                            if item_key in self.item_aug_data:
                                item_aug_vec.append(torch.tensor(self.item_aug_data[item_key]).float())
                            else:
                                item_aug_vec.append(torch.tensor([0.0] * self.aug_vec_dim).float())
                    
                    user_key = str(uid)
                    if user_key in self.hist_aug_data:
                        hist_aug_vec = self.hist_aug_data[user_key]
                    else:
                        hist_aug_vec = [0.0] * self.aug_vec_dim
                    
                    out_dict['item_aug_vec_list'] = item_aug_vec
                    out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
            else:
                raise NotImplementedError(f"Task {self.task} not supported. Only 'rerank' is supported.")

            return out_dict
        
        except Exception as e:
            print(f"Error in __getitem__ at index {_id}: {str(e)}")
            print(f"Data sample: {self.data[_id] if _id < len(self.data) else 'Index out of range'}")
            if hasattr(self, 'augment') and self.augment:
                print(f"Augment enabled, aug_vec_dim: {self.aug_vec_dim}")
            raise e

