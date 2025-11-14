'''
-*- coding: utf-8 -*-
@File  : main_rerank_multilevel_memory.py
@Description: 基于多级记忆增强的Rerank模型训练脚本
'''
# 1.python
import os
import sys
import time

# 添加上级目录到路径，以便导入RS目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
import datetime
import copy
# 2.pytorch
import torch
import torch.utils.data as Data
# 3.sklearn
from sklearn.metrics import roc_auc_score, log_loss

from utils import load_parse_from_json, setup_seed, load_data, weight_init, str2list
from models import DLCM, PRM, SetRank, MIR, GSF, EGRerank, LambdaRank
from utils import evaluate_rerank
from dataset import AmzDataset
from optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup


def load_json(file_path):
    """加载JSON文件的辅助函数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {file_path}, 错误: {e}")
        return {}


class MultilevelMemoryDataset(AmzDataset):
    """多级记忆增强数据集类，继承自AmzDataset并添加多级记忆处理"""
    
    def __init__(self, data_path, set='train', task='rerank', max_hist_len=10, augment=False, aug_prefix=None, memory_mode=True):
        super().__init__(data_path, set, task, max_hist_len, augment, aug_prefix)
        self.memory_mode = memory_mode
        
        if memory_mode and augment:
            # 加载多级记忆分析数据（如果存在）
            analysis_file = data_path + f'/{aug_prefix}.analysis'
            if os.path.exists(analysis_file):
                self.memory_analysis_aug_data = load_json(analysis_file)
                print(f'多级记忆分析增强数据大小: {len(self.memory_analysis_aug_data)}')
            else:
                self.memory_analysis_aug_data = {}
                print('未找到多级记忆分析增强数据，使用零向量')
            
            # 加载多级记忆数据
            multilevel_memory_file = data_path + '/multilevel_memory.json'
            if os.path.exists(multilevel_memory_file):
                self.multilevel_memory_data = load_json(multilevel_memory_file)
                print(f'多级记忆数据大小: {len(self.multilevel_memory_data)}')
            else:
                self.multilevel_memory_data = {}
                print('未找到多级记忆数据')
            
            # 加载具体的记忆层级向量（如果存在）
            # 注意：这些独立的三层记忆向量文件目前不生成，暂时使用空字典
            # 实际的多级记忆信息从multilevel_memory_data中获取
            self.sensory_memory_vectors = {}
            self.working_memory_vectors = {}
            self.longterm_memory_vectors = {}
            
            print(f'感觉记忆向量数: {len(self.sensory_memory_vectors)}')
            print(f'工作记忆向量数: {len(self.working_memory_vectors)}')
            print(f'长期记忆向量数: {len(self.longterm_memory_vectors)}')
    
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
                
                # 添加多级记忆信息（为多头注意力机制提供先验信息）
                uid_int = int(uid) if isinstance(uid, str) else uid
                if uid_int in self.multilevel_memory_data:
                    memory_data = self.multilevel_memory_data[uid_int]
                    
                    # 计算各级记忆比例（用于多头注意力权重调整）
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
                    
                    # 添加各级记忆向量（如果存在）
                    if user_key in self.sensory_memory_vectors:
                        out_dict['sensory_memory_vec'] = torch.tensor(self.sensory_memory_vectors[user_key]).float()
                    else:
                        out_dict['sensory_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
                    
                    if user_key in self.working_memory_vectors:
                        out_dict['working_memory_vec'] = torch.tensor(self.working_memory_vectors[user_key]).float()
                    else:
                        out_dict['working_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
                    
                    if user_key in self.longterm_memory_vectors:
                        out_dict['longterm_memory_vec'] = torch.tensor(self.longterm_memory_vectors[user_key]).float()
                    else:
                        out_dict['longterm_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
                else:
                    out_dict['sensory_memory_ratio'] = torch.tensor(0.33).float()
                    out_dict['working_memory_ratio'] = torch.tensor(0.33).float()
                    out_dict['longterm_memory_ratio'] = torch.tensor(0.34).float()
                    out_dict['memory_field_num'] = torch.tensor(0).long()
                    out_dict['sensory_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
                    out_dict['working_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
                    out_dict['longterm_memory_vec'] = torch.zeros(self.aug_vec_dim).float()
        
        return out_dict


def eval_multilevel_memory(model, test_loader, metric_scope, is_rank=True):
    """多级记忆增强模型的评估函数"""
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
    res = evaluate_rerank(labels, preds, metric_scope, is_rank)
    return res, np.mean(losses), eval_time


def test_multilevel_memory(args):
    """多级记忆增强模型的测试函数"""
    model = torch.load(args.reload_path, weights_only=False)
    test_set = MultilevelMemoryDataset(args.data_dir, 'test', args.task, args.max_hist_len, 
                                      args.augment, args.aug_prefix, args.memory_mode)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')] if isinstance(args.metric_scope, str) else args.metric_scope
    print('多级记忆测试数据大小:', len(test_set))
    res, loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, True)
    print("测试损失: %.5f, 测试时间: %.5f" % (loss, eval_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
    print("MRR: %.5f" % res[3])  # MRR是全局指标，无@K


def load_model_multilevel_memory(args, dataset):
    """加载多级记忆增强模型（支持多头注意力融合）"""
    algo = args.algo
    device = args.device
    
    # 为多级记忆增强模型添加特殊参数
    if hasattr(args, 'memory_mode') and args.memory_mode:
        args.memory_specific_export_num = getattr(args, 'memory_specific_export_num', 3)
        args.memory_fusion_type = getattr(args, 'memory_fusion_type', 'attention')  # 使用多头注意力融合
    
    if algo == 'DLCM':
        model = DLCM(args, dataset).to(device)
    elif algo == 'PRM':
        model = PRM(args, dataset).to(device)
    elif algo == 'SetRank':
        model = SetRank(args, dataset).to(device)
    elif algo == 'MIR':
        model = MIR(args, dataset).to(device)
    elif algo == 'GSF':
        model = GSF(args, dataset).to(device)
    elif algo == 'EGRerank':
        model = EGRerank(args, dataset).to(device)
    elif algo == 'LambdaRank':
        model = LambdaRank(args, dataset).to(device)
    else:
        print('不支持的模型类型，请选择: DLCM, PRM, SetRank, MIR, GSF, EGRerank, LambdaRank')
        exit()
    
    model.apply(weight_init)
    return model


def get_optimizer_multilevel_memory(args, model, train_data_num):
    """获取多级记忆增强模型的优化器（简化版）"""
    no_decay = ['bias', 'LayerNorm.weight']
    named_params = [(k, v) for k, v in model.named_parameters()]
    
    # 为多级记忆相关参数设置不同的学习率（简化版，只分两类）
    memory_params = []
    regular_params = []
    
    for n, p in named_params:
        if any(memory_key in n for memory_key in ['memory_', 'multilevel_', 'sensory_', 'working_', 'longterm_']):
            memory_params.append(p)
        else:
            regular_params.append(p)
    
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
            'lr': args.lr * 0.5  # 多级记忆参数使用较小的学习率
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
    """多级记忆增强模型的训练函数"""
    print("开始多级记忆增强模型训练...")
    
    train_set = MultilevelMemoryDataset(args.data_dir, 'train', args.task, args.max_hist_len, 
                                       args.augment, args.aug_prefix, args.memory_mode)
    test_set = MultilevelMemoryDataset(args.data_dir, 'test', args.task, args.max_hist_len, 
                                      args.augment, args.aug_prefix, args.memory_mode)
    
    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print('训练数据大小:', len(train_set), '测试数据大小:', len(test_set))

    model = load_model_multilevel_memory(args, test_set)
    print("多级记忆增强模型加载成功!")

    optimizer, scheduler = get_optimizer_multilevel_memory(args, model, len(train_set))

    print("开始多级记忆增强训练...")
    print("=" * 60)
    best_map = 0
    best_model_state = None
    global_step = 0
    patience = 0
    metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')] if isinstance(args.metric_scope, str) else args.metric_scope
    print('指标范围:', metric_scope)
    
    # 初始评估
    res, eval_loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, False)
    print("初始评估 - 损失: %.5f, 时间: %.5f" % (eval_loss, eval_time))
    for i, scope in enumerate(metric_scope):
        print("  @%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
    print("  MRR: %.5f" % res[3])
    
    for epoch in range(args.epoch_num):
        t = time.time()
        train_loss = []
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            outputs = model(data)
            loss = outputs['loss']
            
            # 添加多级记忆正则化损失（如果模型支持，简化版）
            if hasattr(model, 'get_memory_regularization_loss'):
                memory_reg_loss = model.get_memory_regularization_loss()
                loss = loss + 0.01 * memory_reg_loss  # 可调整权重
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            global_step += 1
                
        train_time = time.time() - t
        res, eval_loss, eval_time = eval_multilevel_memory(model, test_loader, metric_scope, True)
        current_map = res[0][1]  # 使用 MAP@5 作为主要指标（索引1对应scope[1]=5）
        
        print("EPOCH %d | 训练损失: %.5f | 测试损失: %.5f | 时间: %.1fs | LR: %.2e" % 
              (epoch, np.mean(train_loss), eval_loss, train_time + eval_time, scheduler.get_last_lr()[0]))
        for i, scope in enumerate(metric_scope):
            print("  @%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
        print("  MRR: %.5f" % res[3])
        
        if current_map > best_map:
            improvement = current_map - best_map if best_map > 0 else current_map
            best_map = current_map
            best_model_state = copy.deepcopy(model.state_dict())
            print('✓ 新的最佳结果 (MAP@5: %.5f) | 提升: +%.5f' % (best_map, improvement))
            patience = 0
        else:
            patience += 1
            print(f'⚠ 无改善 ({patience}/{args.patience}) | 当前最佳: {best_map:.5f}')
            if patience >= args.patience:
                print(f'🛑 多级记忆训练早停: 连续{patience}轮无改善')
                break
    
    print("=" * 60)
    print(f"多级记忆训练完成! 最佳MAP@5: {best_map:.5f}")
    
    # 最终测试
    print("最终测试评估...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_res, final_loss, final_time = eval_multilevel_memory(model, test_loader, metric_scope, True)
    print("test loss: %.5f, test time: %.5f" % (final_loss, final_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, final_res[0][i], final_res[1][i], final_res[2][i]))
    print("MRR: %.5f" % final_res[3])  # MRR是全局指标，无@K


def parse_args_multilevel_memory():
    """解析多级记忆增强模型的参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/mooc/proc_data/')
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
    parser.add_argument('--metric_scope', default='1,3,5,7,10', type=str, help='metric scope (comma-separated)')

    parser.add_argument('--task', default='rerank', type=str, help='task, ctr or rerank')
    parser.add_argument('--algo', default='DLCM', type=str, help='model name')
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

    # 多级记忆增强特有参数（支持多头注意力融合）
    parser.add_argument('--memory_mode', default='true', type=str, help='enable multilevel memory enhancement mode')
    parser.add_argument('--memory_specific_export_num', default=3, type=int, help='number of memory-specific experts')
    parser.add_argument('--memory_fusion_type', default='attention', type=str, help='multilevel memory fusion type (attention/mlp)')
    parser.add_argument('--memory_weight_decay', default=0.01, type=float, help='regularization weight for memory consistency')
    
    # 多头注意力融合参数（统一的注意力头数）
    parser.add_argument('--enable_memory_attention', default='true', type=str, help='enable multi-head attention for memory fusion')
    parser.add_argument('--memory_attn_heads', default=4, type=int, help='number of attention heads for multilevel memory fusion')

    parser.add_argument('--hidden_size', default=64, type=int, help='size of hidden size')
    parser.add_argument('--rnn_dp', default=0.0, type=float, help='dropout rate in RNN')
    parser.add_argument('--n_head', default=2, type=int, help='num of attention head in PRM')
    parser.add_argument('--ff_dim', default=128, type=int, help='feedforward dim in PRM')
    parser.add_argument('--attn_dp', default=0.0, type=str, help='attention dropout in PRM')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature in SetRank')

    args, _ = parser.parse_known_args()
    args.augment = True if args.augment.lower() == 'true' else False
    args.memory_mode = True if args.memory_mode.lower() == 'true' else False
    args.enable_memory_attention = True if args.enable_memory_attention.lower() == 'true' else False

    print('最大历史长度:', args.max_hist_len)
    print('多级记忆增强模式:', args.memory_mode)
    print('多级记忆融合类型:', args.memory_fusion_type)
    if args.enable_memory_attention:
        print(f'多头注意力融合: 启用 | 注意力头数: {args.memory_attn_heads}')

    return args


if __name__ == '__main__':
    args = parse_args_multilevel_memory()
    print(args.timestamp)
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)
    setup_seed(args.seed)

    print(f'模型: {args.algo} | 任务: {args.task} | 轮数: {args.epoch_num} | 批次大小: {args.batch_size} | 学习率: {args.lr} | 嵌入维度: {args.embed_dim}')
    if args.augment:
        print(f'增强: {args.aug_prefix} | 转换: {args.convert_type} | 转换架构: {args.convert_arch}')
    if args.memory_mode:
        fusion_desc = f'{args.memory_fusion_type} (多头注意力)' if args.enable_memory_attention else args.memory_fusion_type
        print(f'多级记忆增强: 启用 | 记忆专家数: {args.memory_specific_export_num} | 融合类型: {fusion_desc}')
        if args.enable_memory_attention:
            print(f'多头注意力头数: {args.memory_attn_heads}')
    
    if args.test:
        test_multilevel_memory(args)
    else:
        train_multilevel_memory(args)

