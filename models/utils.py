import json
import pickle
import os
import random

import torch
from  torch import nn
import torch.utils.data as Data
import numpy as np

AGE_MAPPING = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

GENDER_MAPPING = {'F': 'female', 'M': 'male'}

OCCUPATION_MAPPING = {
    0: "in an unknown occupation",
    1: "an academic/educator",
    2: "an artist",
    3: "in clerical/admin department",
    4: "a college/grad student",
    5: "a customer service staff",
    6: "a doctor/health care",
    7: "an executive/managerial",
    8: "a farmer",
    9: "a homemaker",
    10: "a K-12 student",
    11: "a lawyer",
    12: "a programmer",
    13: "retired",
    14: "in sales/marketing department",
    15: "a scientist",
    16: "self-employed",
    17: "a technician/engineer",
    18: "a tradesman/craftsman",
    19: "unemployed",
    20: "a writer",
}


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def load_data(data_path, lm_vec=None, item_vec=None):
    with open(data_path, 'rb') as r:
        hist, itm_fts, usr_fts, lm_idx, lbs = pickle.load(r)
    if lm_vec:
        hist_dens = np.array(lm_vec)[np.array(lm_idx)]
        itm_dens = [item_vec[itm_ft[0]] for itm_ft in itm_fts]
        # data_size = len(hist)
        # hist_dens = np.random.random((data_size, 4096))
        # itm_dens = np.random.random((data_size, 4096))
        hist_dens = torch.from_numpy(hist_dens).long()
        itm_dens = torch.tensor(itm_dens).long()
    hist_spar = torch.tensor(hist).long()
    itm_spar = torch.tensor(itm_fts).long()
    usr_fts = torch.tensor(usr_fts).long()
    lbs = torch.tensor(lbs).long()
    if lm_vec:
        data_set = Data.TensorDataset(hist_spar, hist_dens, itm_spar, itm_dens, usr_fts, lbs)
    else:
        data_set = Data.TensorDataset(hist_spar, itm_spar, usr_fts, lbs)
    return data_set


def load_train_and_test(train_path, test_path, lm_vec_path=None):
    if lm_vec_path:
        with open(lm_vec_path, 'rb') as r:
            train_hist_vec, test_hist_vec, item_vec_dict = pickle.load(r)
    else:
        train_hist_vec, test_hist_vec, item_vec_dict = None, None, None
    train_set = load_data(train_path, train_hist_vec, item_vec_dict)
    test_set = load_data(test_path, test_hist_vec, item_vec_dict)
    return train_set, test_set


def correct_title(title):
    title = title.strip()
    spl_list = title.split(',')
    # spl_list = [word.strip() for word in spl_list]
    last_word = spl_list[-1].strip().lower()
    if last_word == 'the' or last_word == 'a':
        tmp = ','.join(spl_list[:-1])
        title = spl_list[-1].strip() + ' ' + tmp
    return title


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)

def str2list(s):
    return [int(i.strip()) for i in s.strip().split(',')]
# def obtain_verbalizer_ids(verbalizer, tokenizer):
#     verbalizer_ids = tokenizer.convert_tokens_to_id(list(verbalizer.keys()))
#     index2ids = {1: verbalizer_ids[i] for i in range(len(verbalizer_ids))}
#     return verbalizer_ids, index2ids


def get_paragraph_representation(outputs, mask, pooler='cls', dim=1):
    last_hidden = outputs.last_hidden_state
    hidden_states = outputs.hidden_states

    # Apply different poolers

    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return outputs.pooler_output.cpu()  # chatglm不能用，用于bert
    elif pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == 'len_last':  # 根据padding方式last方式也不一样
        lens = mask.unsqueeze(-1).sum(dim)
        # index = torch.arange(last_hidden.shape[0])
        # print(index)
        pooled_result = [last_hidden[i, lens[i] - 1, :] for i in range(last_hidden.shape[0])]
        pooled_result = torch.cat(pooled_result, dim=0)
        return pooled_result.cpu()
    elif pooler == 'last':
        if dim == 0:
            return last_hidden[-1, :, :]
        else:
            return last_hidden[:, -1, :]
    elif pooler == 'wavg':
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float().to(last_hidden.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            mask
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded * weights, dim=dim)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=dim)

        pooled_result = sum_embeddings / sum_mask
        return pooled_result.cpu()
    else:
        raise NotImplementedError


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def evaluate_rerank(labels, preds, scope_number, is_rank, compute_auc=False):
    """
    评估rerank/rank任务的性能指标
    返回: MAP, NDCG, HR (Hit Rate), MRR (Mean Reciprocal Rank，全局指标), AUC (可选)
    """
    ndcg, map, hr = [[] for _ in range(len(scope_number))], [[] for _ in range(len(scope_number))], \
                    [[] for _ in range(len(scope_number))]
    mrr_list = []  # MRR是全局指标，不分@K
    
    # AUC相关
    all_labels_flat = []
    all_preds_flat = []

    for label, pred in zip(labels, preds):
        if is_rank:
            final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        else:
            final = list(range(len(pred)))
        click = np.array(label)[final].tolist()  # reranked labels
        gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg
        
        # 用于AUC计算（收集所有标签和预测）
        if compute_auc:
            all_labels_flat.extend(label)
            all_preds_flat.extend(pred)

        # 计算MRR（全局，不限定在@K范围内）
        # MRR计算第一个相关项目在整个排序列表中的倒数排名
        _mrr = 0.0
        for j in range(len(click)):
            if click[j] >= 1:
                _mrr = 1.0 / (j + 1)
                break
        mrr_list.append(_mrr)

        for i, scope in enumerate(scope_number):
            ideal_dcg, dcg,  AP_value, AP_count = 0, 0, 0, 0
            cur_scope = min(scope, len(label))
            
            # 计算NDCG和MAP
            for _i, _g, _f in zip(range(1, cur_scope + 1), gold[:cur_scope], final[:cur_scope]):
                dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                if click[_i - 1] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.
            
            # 计算HR@K (Hit Rate): 前K个推荐中是否有相关项目
            # 相关项目定义为label >= 1
            _hr = 1.0 if any(click[j] >= 1 for j in range(cur_scope)) else 0.0

            ndcg[i].append(_ndcg)
            map[i].append(_map)
            hr[i].append(_hr)
    
    # 计算AUC
    auc_score = None
    if compute_auc and len(all_labels_flat) > 0:
        try:
            from sklearn.metrics import roc_auc_score
            # 确保有正负样本
            if len(set(all_labels_flat)) > 1:
                auc_score = roc_auc_score(all_labels_flat, all_preds_flat)
            else:
                auc_score = 0.5  # 只有一个类别时，AUC设为0.5
        except Exception as e:
            print(f"AUC计算失败: {e}")
            auc_score = None
    
    # MRR是全局指标，返回平均值（不是数组）
    mrr_score = np.mean(mrr_list) if mrr_list else 0.0
    
    return (np.mean(np.array(map), axis=-1), 
            np.mean(np.array(ndcg), axis=-1), 
            np.mean(np.array(hr), axis=-1),
            mrr_score,  # MRR是单个值，不是数组
            auc_score)

