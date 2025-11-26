'''
Rank model training with multilevel memory enhancement
'''
# 1.python
import os
import sys
import time

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
import datetime
import copy
import torch
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, log_loss

from models.utils import load_parse_from_json, setup_seed, load_data, weight_init, str2list, evaluate_rerank
from models.rank import DeepFM, DeepInterestNet, DIEN, DCN, FiBiNet, FiGNN, AutoInt
from models.dataset import MemoryDataset
from models.optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup


def load_json(file_path):
    """加载JSON文件的辅助函数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {file_path}, 错误: {e}")
        return {}


class MultilevelMemoryDataset(MemoryDataset):
    def __init__(self, data_path, set='train', task='rerank', max_hist_len=10, augment=False, aug_prefix=None, memory_mode=True, data_file=None):
        super().__init__(data_path, set, task, max_hist_len, augment, aug_prefix, data_file)
        self.memory_mode = memory_mode
        
        if memory_mode and augment:
            analysis_file = data_path + f'/{aug_prefix}.analysis'
            if os.path.exists(analysis_file):
                self.memory_analysis_aug_data = load_json(analysis_file)
            else:
                self.memory_analysis_aug_data = {}
            
            multilevel_memory_file = data_path + '/multilevel_memory.json'
            if os.path.exists(multilevel_memory_file):
                self.multilevel_memory_data = load_json(multilevel_memory_file)
            else:
                self.multilevel_memory_data = {}
    
    def __getitem__(self, _id):
        """重写数据获取方法，添加多级记忆处理"""
        out_dict = super().__getitem__(_id)
        
        if self.memory_mode and self.augment:
            if self.task == 'rerank':
                uid, seq_idx, cands, lb = self.data[_id]
                
                # 添加多级记忆分析向量
                user_key = str(uid)
                if user_key in self.memory_analysis_aug_data:
                    memory_analysis_aug_vec = self.memory_analysis_aug_data[user_key]
                else:
                    memory_analysis_aug_vec = [0.0] * self.aug_vec_dim
                
                out_dict['memory_analysis_aug_vec'] = torch.tensor(memory_analysis_aug_vec).float()
                
                uid_int = int(uid) if isinstance(uid, str) else uid
                if uid_int in self.multilevel_memory_data:
                    memory_data = self.multilevel_memory_data[uid_int]
                    
                    sensory_len = len(memory_data.get('sensory_memory', [{}])[0]) if memory_data.get('sensory_memory') else 0
                    working_len = len(memory_data.get('working_memory', [{}])[0]) if memory_data.get('working_memory') else 0
                    longterm_len = len(memory_data.get('long_term_memory', [{}])[0]) if memory_data.get('long_term_memory') else 0
                    total_len = sensory_len + working_len + longterm_len
                    
                    if total_len > 0:
                        sensory_ratio = sensory_len / total_len
                        working_ratio = working_len / total_len
                        longterm_ratio = longterm_len / total_len
                    else:
                        sensory_ratio = 0.33
                        working_ratio = 0.33
                        longterm_ratio = 0.34
                    
                    out_dict['sensory_memory_ratio'] = torch.tensor(sensory_ratio).float()
                    out_dict['working_memory_ratio'] = torch.tensor(working_ratio).float()
                    out_dict['longterm_memory_ratio'] = torch.tensor(longterm_ratio).float()
                    out_dict['memory_field_num'] = torch.tensor(len(memory_data.get('long_term_fields', []))).long()
                else:
                    out_dict['sensory_memory_ratio'] = torch.tensor(0.33).float()
                    out_dict['working_memory_ratio'] = torch.tensor(0.33).float()
                    out_dict['longterm_memory_ratio'] = torch.tensor(0.34).float()
                    out_dict['memory_field_num'] = torch.tensor(0).long()
        
        return out_dict


def eval_multilevel_memory(model, test_loader, metric_scope, is_rank=True, compute_auc=True):
    model.eval()
    losses = []
    preds = []
    labels = []
    t = time.time()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            outputs = model(data)
            loss = outputs['loss']
            logits = outputs['logits']
            preds.extend(logits.detach().cpu().tolist())
            labels.extend(outputs['labels'].detach().cpu().tolist())
            losses.append(loss.item())
    eval_time = time.time() - t
    res = evaluate_rerank(labels, preds, metric_scope, is_rank, compute_auc)
    return res, np.mean(losses), eval_time


def test_multilevel_memory(args):
    model = torch.load(args.reload_path, weights_only=False)
    test_set = MultilevelMemoryDataset(args.data_dir, 'test', 'rerank', args.max_hist_len, 
                                      args.augment, args.aug_prefix, args.memory_mode, data_file='rank')
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    metric_scope = args.metric_scope
    res, loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, True, compute_auc=True)
    print(f"Test Loss: {loss:.5f}, Time: {eval_time:.5f}")
    for i, scope in enumerate(metric_scope):
        print(f"@{scope}, MAP: {res[0][i]:.5f}, NDCG: {res[1][i]:.5f}, HR: {res[2][i]:.5f}")
    print(f"MRR: {res[3]:.5f}")
    if res[4] is not None:
        print(f"AUC: {res[4]:.5f}")


def load_model_multilevel_memory(args, dataset):
    algo = args.algo
    device = args.device
    
    if hasattr(args, 'memory_mode') and args.memory_mode:
        args.memory_specific_export_num = getattr(args, 'memory_specific_export_num', 3)
        args.memory_fusion_type = getattr(args, 'memory_fusion_type', 'attention')
    
    if algo == 'DeepFM':
        model = DeepFM(args, dataset).to(device)
    elif algo == 'DCN':
        model = DCN(args, 'v1', dataset).to(device)
    elif algo == 'FiBiNet':
        model = FiBiNet(args, dataset).to(device)
    elif algo == 'FiGNN':
        model = FiGNN(args, dataset).to(device)
    elif algo == 'AutoInt':
        model = AutoInt(args, dataset).to(device)
    elif algo == 'DIN':
        model = DeepInterestNet(args, dataset).to(device)
    elif algo == 'DIEN':
        model = DIEN(args, dataset).to(device)
    else:
        raise ValueError(f'Unsupported model: {algo}. Supported: DeepFM, DCN, FiBiNet, FiGNN, AutoInt, DIN, DIEN')
    
    model.apply(weight_init)
    return model


def get_optimizer_multilevel_memory(args, model, train_data_num):
    no_decay = ['bias', 'LayerNorm.weight']
    named_params = [(k, v) for k, v in model.named_parameters()]
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay) and not any(memory_key in n for memory_key in ['memory_', 'multilevel_', 'sensory_', 'working_', 'longterm_'])],
            'weight_decay': args.weight_decay,
            'lr': args.lr
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and not any(memory_key in n for memory_key in ['memory_', 'multilevel_', 'sensory_', 'working_', 'longterm_'])],
            'weight_decay': 0.0,
            'lr': args.lr
        },
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay) and any(memory_key in n for memory_key in ['memory_', 'multilevel_', 'sensory_', 'working_', 'longterm_'])],
            'weight_decay': args.weight_decay,
            'lr': args.lr * 0.5
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and any(memory_key in n for memory_key in ['memory_', 'multilevel_', 'sensory_', 'working_', 'longterm_'])],
            'weight_decay': 0.0,
            'lr': args.lr * 0.5
        }
    ]
    
    beta1, beta2 = args.adam_betas.split(',')
    beta1, beta2 = float(beta1), float(beta2)
    adam_epsilon = float(args.adam_epsilon)
    optimizer = AdamW(optimizer_grouped_parameters, eps=adam_epsilon, betas=(beta1, beta2))

    t_total = int(train_data_num * args.epoch_num)
    t_warmup = int(t_total * args.warmup_ratio)
    if args.lr_sched.lower() == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=t_warmup,
                                                    num_training_steps=t_total)
    elif args.lr_sched.lower() == 'const':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=t_warmup)
    else:
        raise NotImplementedError
    return optimizer, scheduler


def train_multilevel_memory(args):
    train_set = MultilevelMemoryDataset(args.data_dir, 'train', 'rerank', args.max_hist_len, 
                                       args.augment, args.aug_prefix, args.memory_mode, data_file='rank')
    test_set = MultilevelMemoryDataset(args.data_dir, 'test', 'rerank', args.max_hist_len, 
                                      args.augment, args.aug_prefix, args.memory_mode, data_file='rank')
    
    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    model = load_model_multilevel_memory(args, test_set)
    optimizer, scheduler = get_optimizer_multilevel_memory(args, model, len(train_set))

    best_map = 0
    best_model_state = None
    global_step = 0
    patience = 0
    metric_scope = args.metric_scope
    
    res, eval_loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, False, compute_auc=True)
    
    for epoch in range(args.epoch_num):
        t = time.time()
        train_loss = []
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            outputs = model(data)
            loss = outputs['loss']
            
            if hasattr(model, 'get_memory_regularization_loss'):
                memory_reg_loss = model.get_memory_regularization_loss()
                loss = loss + 0.01 * memory_reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            global_step += 1
                
        train_time = time.time() - t
        res, eval_loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, True, compute_auc=True)
        main_k_idx = len(metric_scope) // 2
        current_map = res[0][main_k_idx]
        
        if current_map > best_map:
            best_map = current_map
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    
    main_k = metric_scope[len(metric_scope) // 2]
    print(f"Best MAP@{main_k}: {best_map:.5f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_res, final_loss, final_time = eval_multilevel_memory(model, test_loader, metric_scope, True, compute_auc=True)
    print(f"Final - Loss: {final_loss:.5f}, MAP@{main_k}: {final_res[0][main_k_idx]:.5f}, NDCG@{main_k}: {final_res[1][main_k_idx]:.5f}, HR@{main_k}: {final_res[2][main_k_idx]:.5f}, MRR: {final_res[3]:.5f}")


def parse_args_multilevel_memory():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/mooc/proc_data/')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--output_dim', default=1, type=int, help='output_dim')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))

    parser.add_argument('--epoch_num', default=20, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='l2 loss scale')
    parser.add_argument('--adam_betas', default='0.9,0.999', type=str, help='beta1 and beta2 for Adam optimizer.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=str, help='Epsilon for Adam optimizer.')
    parser.add_argument('--lr_sched', default='cosine', type=str, help='Type of LR schedule method')
    parser.add_argument('--warmup_ratio', default=0.0, type=float, help='linear warmup over warmup_ratio if warmup_steps not set')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--convert_dropout', default=0.0, type=float, help='dropout rate of convert module')
    parser.add_argument('--grad_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--patience', default=5, type=int, help='The patience for early stop (increased for augmented models)')
    parser.add_argument('--metric_scope', default='3,5,10', type=str, help='metric scope (e.g., 3,5,10 for Coursera, 3,5,10,20 for MOOC)')

    parser.add_argument('--task', default='rerank', type=str, help='task type: rerank (用于Rank和Rerank阶段，处理多候选排序)')
    parser.add_argument('--algo', default='DeepFM', type=str, help='model name')
    parser.add_argument('--augment', default='true', type=str, help='whether to use augment vectors')
    parser.add_argument('--aug_prefix', default='bert-base-uncased_avg_augment_multilevel_memory', type=str, help='prefix of augment file')
    parser.add_argument('--convert_type', default='MultilevelMemoryHEA', type=str, help='type of convert module')
    parser.add_argument('--max_hist_len', default=5, type=int, help='the max length of user history')
    parser.add_argument('--embed_dim', default=32, type=int, help='size of embedding')
    parser.add_argument('--final_mlp_arch', default='200,80', type=str2list, help='size of final layer')
    parser.add_argument('--convert_arch', default='128,32', type=str2list,
                        help='size of convert net (MLP/export net in MoE)')
    parser.add_argument('--export_num', default=2, type=int, help='number of expert')
    parser.add_argument('--top_expt_num', default=4, type=int, help='number of expert')
    parser.add_argument('--specific_export_num', default=6, type=int, help='number of specific expert in PLE')
    parser.add_argument('--auxi_loss_weight', default=0, type=float, help='loss for load balance in expert')

    parser.add_argument('--memory_mode', default='true', type=str, help='enable multilevel memory enhancement mode')
    parser.add_argument('--memory_specific_export_num', default=3, type=int, help='number of memory-specific experts')
    parser.add_argument('--memory_fusion_type', default='attention', type=str, help='hierarchical memory fusion type')
    parser.add_argument('--memory_weight_decay', default=0.01, type=float, help='regularization weight for memory consistency')
    parser.add_argument('--enable_memory_attention', default='true', type=str, help='enable multi-head attention for memory fusion')
    parser.add_argument('--memory_attn_heads', default=4, type=int, help='number of attention heads for memory fusion')
    parser.add_argument('--enable_knowledge_reduction', default='false', type=str, help='enable knowledge dimension reduction')
    parser.add_argument('--knowledge_reduction_dim', default=128, type=int, help='target dimension for knowledge reduction')
    parser.add_argument('--knowledge_reduction_dropout', default=0.3, type=float, help='dropout rate for knowledge reduction layer')

    parser.add_argument('--hidden_size', default=64, type=int, help='size of hidden size')
    parser.add_argument('--rnn_dp', default=0.0, type=float, help='dropout rate in RNN')
    parser.add_argument('--n_head', default=2, type=int, help='num of attention head')
    parser.add_argument('--deepfm_latent_dim', default=16, type=int, help='dimension of latent variable in DeepFM')
    parser.add_argument('--deepfm_deep_arch', default='200,80', type=str2list, help='size of deep net in DeepFM')
    parser.add_argument('--dien_gru', default='GRU', type=str, help='gru type in DIEN')
    parser.add_argument('--ff_dim', default=128, type=int, help='feedforward dim in PRM')
    parser.add_argument('--attn_dp', default=0.0, type=str, help='attention dropout in PRM')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature in SetRank')
    
    # DCN 参数
    parser.add_argument('--dcn_deep_arch', default='200,80', type=str2list, help='deep part architecture in DCN')
    parser.add_argument('--dcn_cross_num', default=3, type=int, help='number of cross layers in DCN')
    
    # AutoInt 参数
    parser.add_argument('--num_attn_layers', default=3, type=int, help='number of attention layers in AutoInt')
    parser.add_argument('--num_attn_heads', default=2, type=int, help='number of attention heads in AutoInt')
    parser.add_argument('--attn_size', default=32, type=int, help='attention size in AutoInt')
    parser.add_argument('--res_conn', default=True, type=bool, help='use residual connection in AutoInt/FiGNN')
    parser.add_argument('--attn_scale', default=False, type=bool, help='use attention scale in AutoInt')
    
    # FiBiNet 参数
    parser.add_argument('--reduction_ratio', default=3, type=int, help='reduction ratio in FiBiNet SENET layer')
    parser.add_argument('--bilinear_type', default='field_interaction', type=str, help='bilinear type in FiBiNet')
    
    # FiGNN 参数
    parser.add_argument('--gnn_layer_num', default=3, type=int, help='number of GNN layers in FiGNN')
    parser.add_argument('--reuse_graph_layer', default=False, type=bool, help='reuse graph layer in FiGNN')

    args, _ = parser.parse_known_args()
    args.augment = True if args.augment.lower() == 'true' else False
    args.memory_mode = True if args.memory_mode.lower() == 'true' else False
    args.enable_memory_attention = True if args.enable_memory_attention.lower() == 'true' else False
    args.enable_knowledge_reduction = True if args.enable_knowledge_reduction.lower() == 'true' else False
    
    # 解析metric_scope字符串为整数列表
    if isinstance(args.metric_scope, str):
        args.metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')]

    return args

if __name__ == '__main__':
    args = parse_args_multilevel_memory()
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)
    setup_seed(args.seed)
    
    if args.test:
        test_multilevel_memory(args)
    else:
        train_multilevel_memory(args)
