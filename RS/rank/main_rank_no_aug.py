'''
-*- coding: utf-8 -*-
@File  : main_rank_no_aug.py
@Description: Rank training without augmentation for comparison (粗排阶段无增强基线)
'''
# 1.python
import os
import sys
import time
import warnings
import copy

# 添加上级目录到路径，以便导入RS目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
import datetime
# 2.pytorch
import torch
import torch.utils.data as Data
# 3.sklearn
from sklearn.metrics import roc_auc_score, log_loss

from utils import load_parse_from_json, setup_seed, load_data, weight_init, str2list
from models import DeepFM, xDeepFM, DeepInterestNet, DIEN, DCN, FiBiNet, FiGNN, AutoInt
from utils import evaluate_rerank, save_paper_metrics
from dataset import AmzDataset
from optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

# 屏蔽PyTorch警告
warnings.filterwarnings("ignore")


def eval(model, test_loader, metric_scope, is_rank=True, compute_auc=True):
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


def test(args):
    model = torch.load(args.reload_path, weights_only=False)
    # Rank粗排测试：加载rank数据文件，task设为rerank
    test_set = AmzDataset(args.data_dir, 'test', 'rerank', args.max_hist_len, augment=False, data_file='rank')
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    metric_scope = args.metric_scope
    res, loss, eval_time = eval(model, test_loader, metric_scope, True, compute_auc=True)
    print("test loss: %.5f, test time: %.5f" % (loss, eval_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
    print("MRR: %.5f" % res[3])  # MRR是全局指标，无@K
    if res[4] is not None:
        print("AUC: %.5f" % res[4])
    metrics_output = args.metrics_output or os.path.join(
        'results', 'paper_metrics', f'rank_{args.algo}_seed{args.seed}.json'
    )
    save_paper_metrics(
        metrics_output, 'rank', args.algo, metric_scope, res, seed=args.seed
    )


def load_model(args, dataset):
    """加载Rank模型（支持所有CTR模型）"""
    algo = args.algo
    device = args.device

    # 特征交互模型
    if algo == 'DeepFM':
        model = DeepFM(args, dataset).to(device)
    elif algo == 'xDeepFM':
        model = xDeepFM(args, dataset).to(device)
    elif algo == 'DCN':
        model = DCN(args, 'v1', dataset).to(device)  # DCN需要mode参数，使用v1
    elif algo == 'FiBiNet':
        model = FiBiNet(args, dataset).to(device)
    elif algo == 'FiGNN':
        model = FiGNN(args, dataset).to(device)
    elif algo == 'AutoInt':
        model = AutoInt(args, dataset).to(device)
    # 用户行为模型
    elif algo == 'DIN':
        model = DeepInterestNet(args, dataset).to(device)
    elif algo == 'DIEN':
        model = DIEN(args, dataset).to(device)
    else:
        print('不支持的模型类型，请选择: DeepFM, xDeepFM, DCN, FiBiNet, FiGNN, AutoInt, DIN, DIEN')
        print('所有CTR模型均可用于Rank阶段')
        exit()

    model.apply(weight_init)
    return model


def get_optimizer(args, model, train_data_num):
    no_decay = ['bias', 'LayerNorm.weight']
    named_params = [(k, v) for k, v in model.named_parameters()]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    beta1, beta2 = args.adam_betas.split(',')
    beta1, beta2 = float(beta1), float(beta2)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon,
                          betas=(beta1, beta2))

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


def train(args):
    print("Starting Rank train function (NO AUGMENTATION)...")
    # Rank粗排：加载rank数据文件，但模型task设为rerank（Rank和Rerank模型逻辑相同）
    # 注意：这里明确设置augment=False
    train_set = AmzDataset(args.data_dir, 'train', 'rerank', args.max_hist_len, augment=False, data_file='rank')
    test_set = AmzDataset(args.data_dir, 'test', 'rerank', args.max_hist_len, augment=False, data_file='rank')

    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print('Train data size:', len(train_set), 'Test data size:', len(test_set))

    model = load_model(args, test_set)
    print("Rank Model loaded successfully!")

    optimizer, scheduler = get_optimizer(args, model, len(train_set))

    print("开始Rank训练 (无增强基线模型)...")
    print("=" * 60)
    best_map = 0
    best_model_state = None
    global_step = 0
    patience = 0
    metric_scope = args.metric_scope
    print('指标范围:', metric_scope)

    # 初始评估
    res, eval_loss, eval_time = eval(model, test_loader, metric_scope, False, compute_auc=True)
    print("初始评估 - 损失: %.5f, 时间: %.5f" % (eval_loss, eval_time))
    for i, scope in enumerate(metric_scope):
        print("  @%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
    print("  MRR: %.5f" % res[3])  # MRR是全局指标，无@K
    if res[4] is not None:
        print("  AUC: %.5f" % res[4])

    for epoch in range(args.epoch_num):
        t = time.time()
        train_loss = []
        model.train()

        for batch_idx, data in enumerate(train_loader):
            outputs = model(data)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            global_step += 1

        train_time = time.time() - t
        res, eval_loss, eval_time = eval(model, test_loader, metric_scope, True)
        main_k_idx = metric_scope.index(5) if 5 in metric_scope else 0
        current_map = res[0][main_k_idx]

        print("EPOCH %d | 训练损失: %.5f | 测试损失: %.5f | 时间: %.1fs" %
              (epoch, np.mean(train_loss), eval_loss, train_time + eval_time))
        for i, scope in enumerate(metric_scope):
            print("  @%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
        print("  MRR: %.5f" % res[3])  # MRR是全局指标，无@K
        if res[4] is not None:
            print("  AUC: %.5f" % res[4])

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
                print(f'🛑 早停: 连续{patience}轮无改善')
                break

    print("=" * 60)
    print(f"Rank训练完成！最佳 MAP@5: {best_map:.5f}")

    # 最终测试
    print("最终测试评估...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_res, final_loss, final_time = eval(model, test_loader, metric_scope, True)
    print("test loss: %.5f, test time: %.5f" % (final_loss, final_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, final_res[0][i], final_res[1][i], final_res[2][i]))
    print("MRR: %.5f" % final_res[3])  # MRR是全局指标，无@K
    if final_res[4] is not None:
        print("AUC: %.5f" % final_res[4])
    metrics_output = args.metrics_output or os.path.join(
        'results', 'paper_metrics', f'rank_{args.algo}_seed{args.seed}.json'
    )
    save_paper_metrics(
        metrics_output, 'rank', args.algo, metric_scope, final_res, seed=args.seed
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/MOOCCubeX/proc_data/')
    parser.add_argument('--save_dir', type=str, default='', help='model save dir')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--output_dim', default=1, type=int, help='output_dim')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))

    parser.add_argument('--augment', default=False, type=bool, help='use augmentation (disabled for comparison)')
    parser.add_argument('--convert_dropout', default=0.0, type=float, help='convert module dropout')
    parser.add_argument('--convert_type', default='mlp', type=str, help='convert module type')
    parser.add_argument('--auxi_loss_weight', default=0.0, type=float, help='auxiliary loss weight')
    parser.add_argument('--convert_arch', default=[768, 256], type=list, help='convert architecture')

    parser.add_argument('--epoch_num', default=20, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='l2 loss scale')
    parser.add_argument('--adam_betas', default='0.9,0.999', type=str, help='beta1 and beta2 for Adam optimizer.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=str, help='Epsilon for Adam optimizer.')
    parser.add_argument('--lr_sched', default='cosine', type=str, help='Type of LR schedule method')
    parser.add_argument('--warmup_ratio', default=0.0, type=float, help='linear warmup over warmup_ratio if warmup_steps not set')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--grad_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--patience', default=3, type=int, help='The patience for early stop')
    parser.add_argument('--metric_scope', default='5,10', type=str, help='paper ranking scopes')
    parser.add_argument('--metrics_output', default=None, help='paper metric JSON output path')

    parser.add_argument('--task', default='rerank', type=str, help='task type: rerank (用于Rank和Rerank阶段，处理多候选排序)')
    parser.add_argument('--algo', default='DeepFM', type=str, help='model name')
    parser.add_argument('--max_hist_len', default=5, type=int, help='the max length of user history')
    parser.add_argument('--embed_dim', default=32, type=int, help='size of embedding')
    parser.add_argument('--final_mlp_arch', default='200,80', type=str2list, help='size of final layer')

    parser.add_argument('--hidden_size', default=64, type=int, help='size of hidden size')
    parser.add_argument('--rnn_dp', default=0.0, type=float, help='dropout rate in RNN')
    parser.add_argument('--n_head', default=2, type=int, help='num of attention head')
    parser.add_argument('--deepfm_latent_dim', default=16, type=int, help='dimension of latent variable in DeepFM')
    parser.add_argument('--deepfm_deep_arch', default='200,80', type=str2list, help='size of deep net in DeepFM')
    parser.add_argument('--cin_layer_units', default='50,50', type=str2list, help='CIN layer in xDeepFM')
    parser.add_argument('--dien_gru', default='GRU', type=str, help='gru type in DIEN')
    parser.add_argument('--ff_dim', default=128, type=int, help='feedforward dim')
    parser.add_argument('--attn_dp', default=0.0, type=str, help='attention dropout')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')

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

    print('max hist len', args.max_hist_len)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.timestamp)
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)

    # 解析metric_scope字符串为整数列表
    if isinstance(args.metric_scope, str):
        args.metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')]

    setup_seed(args.seed)

    print(f'模型: {args.algo} | 阶段: Rank粗排 | task参数: {args.task} | 轮数: {args.epoch_num} | 批次大小: {args.batch_size} | 学习率: {args.lr} | 嵌入维度: {args.embed_dim}')
    print('=' * 60)
    print('*** 无增强Rank基线模型 (NO AUGMENTATION) ***')
    print('支持的模型: DeepFM, xDeepFM, DIN, DIEN')
    print('这些模型既可用于CTR也可用于Rank阶段（2017-2019经典深度模型）')
    print('=' * 60)

    if args.test:
        test(args)
    else:
        train(args)
