'''
-*- coding: utf-8 -*-
@File  : main_rerank_no_aug.py
@Description: Rerank training without augmentation for comparison
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
from models import DLCM, PRM, SetRank, MIR, GSF, EGRerank, LambdaRank, RankFormer, PEAR, PIER
from utils import evaluate_rerank, save_paper_metrics
from dataset import AmzDataset
from optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

# 屏蔽PyTorch警告
warnings.filterwarnings("ignore")


def eval(model, test_loader, metric_scope, is_rank=True):
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


def test(args):
    model = torch.load(args.reload_path, weights_only=False)
    test_set = AmzDataset(args.data_dir, 'test', args.task, args.max_hist_len, augment=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')] if isinstance(args.metric_scope, str) else args.metric_scope
    res, loss, eval_time = eval(model, test_loader, metric_scope, True)
    print("test loss: %.5f, test time: %.5f" % (loss, eval_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, res[0][i], res[1][i], res[2][i]))
    print("MRR: %.5f" % res[3])  # MRR是全局指标，无@K
    metrics_output = args.metrics_output or os.path.join(
        'results', 'paper_metrics', f'rerank_{args.algo}_seed{args.seed}.json'
    )
    save_paper_metrics(
        metrics_output, 'rerank', args.algo, metric_scope, res, seed=args.seed
    )


def load_model(args, dataset):
    """加载模型（支持所有4个Rerank模型）"""
    algo = args.algo
    device = args.device

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
    elif algo == 'RankFormer':
        model = RankFormer(args, dataset).to(device)
    elif algo == 'PEAR':
        model = PEAR(args, dataset).to(device)
    elif algo == 'PIER':
        model = PIER(args, dataset).to(device)
    else:
        print('不支持的模型类型，请选择: DLCM, PRM, SetRank, MIR, GSF, EGRerank, LambdaRank, RankFormer, PEAR, PIER')
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
    print("Starting train function (NO AUGMENTATION)...")
    # 注意：这里明确设置augment=False
    train_set = AmzDataset(args.data_dir, 'train', args.task, args.max_hist_len, augment=False)
    test_set = AmzDataset(args.data_dir, 'test', args.task, args.max_hist_len, augment=False)

    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print('Train data size:', len(train_set), 'Test data size:', len(test_set))

    model = load_model(args, test_set)
    print("Model loaded successfully!")

    optimizer, scheduler = get_optimizer(args, model, len(train_set))

    print("开始训练 (无增强基线模型)...")
    print("=" * 60)
    best_map = 0
    best_model_state = None  # 保存最佳模型的状态字典
    global_step = 0
    patience = 0
    metric_scope = [int(x.strip()) for x in args.metric_scope.split(',')] if isinstance(args.metric_scope, str) else args.metric_scope
    print('指标范围:', metric_scope)

    # 初始评估
    res, eval_loss, eval_time = eval(model, test_loader, metric_scope, False)
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
        print("  MRR: %.5f" % res[3])

        if current_map > best_map:
            improvement = current_map - best_map if best_map > 0 else current_map
            best_map = current_map
            best_model_state = copy.deepcopy(model.state_dict())  # 深拷贝保存最佳模型状态
            print('✓ 新的最佳结果 (MAP@5: %.5f) | 提升: +%.5f' % (best_map, improvement))
            patience = 0
        else:
            patience += 1
            print(f'⚠ 无改善 ({patience}/{args.patience}) | 当前最佳: {best_map:.5f}')
            if patience >= args.patience:
                print(f'🛑 早停: 连续{patience}轮无改善')
                break

    print("=" * 60)
    print(f"训练完成！最佳 MAP@5: {best_map:.5f}")

    # 最终测试
    print("最终测试评估...")
    # 加载最佳模型状态进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_res, final_loss, final_time = eval(model, test_loader, metric_scope, True)
    print("test loss: %.5f, test time: %.5f" % (final_loss, final_time))
    for i, scope in enumerate(metric_scope):
        print("@%d, MAP: %.5f, NDCG: %.5f, HR: %.5f" % (scope, final_res[0][i], final_res[1][i], final_res[2][i]))
    print("MRR: %.5f" % final_res[3])  # MRR是全局指标，无@K
    metrics_output = args.metrics_output or os.path.join(
        'results', 'paper_metrics', f'rerank_{args.algo}_seed{args.seed}.json'
    )
    save_paper_metrics(
        metrics_output, 'rerank', args.algo, metric_scope, final_res, seed=args.seed
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/MOOCCubeX/proc_data/')
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
    parser.add_argument('--metric_scope', default='1,3,5', type=str, help='paper reranking scopes')
    parser.add_argument('--metrics_output', default=None, help='paper metric JSON output path')

    parser.add_argument('--task', default='rerank', type=str, help='task, ctr or rerank')
    parser.add_argument('--algo', default='DLCM', type=str, help='model name')
    parser.add_argument('--max_hist_len', default=5, type=int, help='the max length of user history')
    parser.add_argument('--embed_dim', default=32, type=int, help='size of embedding')
    parser.add_argument('--final_mlp_arch', default='200,80', type=str2list, help='size of final layer')

    parser.add_argument('--hidden_size', default=64, type=int, help='size of hidden size')
    parser.add_argument('--rnn_dp', default=0.0, type=float, help='dropout rate in RNN')
    parser.add_argument('--n_head', default=2, type=int, help='num of attention head in PRM')
    parser.add_argument('--ff_dim', default=128, type=int, help='feedforward dim in PRM')
    parser.add_argument('--attn_dp', default=0.0, type=str, help='attention dropout in PRM')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature in SetRank')
    parser.add_argument('--n_layers', default=2, type=int, help='num of Transformer layers for RankFormer')

    args, _ = parser.parse_known_args()

    print('max hist len', args.max_hist_len)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.timestamp)
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)
    setup_seed(args.seed)

    print(f'模型: {args.algo} | 任务: {args.task} | 轮数: {args.epoch_num} | 批次大小: {args.batch_size} | 学习率: {args.lr} | 嵌入维度: {args.embed_dim}')
    print('=' * 60)
    print('*** 无增强基线模型 (NO AUGMENTATION) ***')
    print('支持的模型: DLCM, PRM, SetRank, MIR')
    print('=' * 60)

    if args.test:
        test(args)
    else:
        train(args)
