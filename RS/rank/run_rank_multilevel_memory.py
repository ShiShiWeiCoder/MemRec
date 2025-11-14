import subprocess
import re
import json
import os
import sys
import argparse
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多级记忆增强Rank模型')
    parser.add_argument('--dataset', type=str, default='mooc', choices=['mooc', 'coursera'],
                       help='数据集选择: mooc (中文) 或 coursera (英文)')
    return parser.parse_args()

# 解析参数
args = parse_args()
dataset_name = args.dataset

# 创建日志文件夹和日志文件
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = open(f'{log_dir}/run_rank_multilevel_memory_{dataset_name}.log', 'w', encoding='utf-8')

def log_print(*args, **kwargs):
    """同时输出到控制台和日志文件"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush()

# 记录开始时间
log_print(f"多级记忆增强Rank模型训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"数据集: {dataset_name.upper()}")
log_print("=" * 60)

# ---------------------------
# 多级记忆增强训练参数
# ---------------------------
data_dir = f'data/{dataset_name}/proc_data'
# Rank粗排说明：数据文件使用rank.train/test，但模型task使用'rerank'（Rank和Rerank模型逻辑相同）
task_name = 'rerank'  # 模型task参数（Rank粗排和Rerank精排的模型处理逻辑相同）

# 根据数据集选择对应的BERT模型（统一使用bert-base-uncased）
if dataset_name == 'mooc':
    aug_prefix = 'bert-base-uncased_avg_augment_multilevel_memory'  # 英文BERT (bert-base-uncased)
else:  # coursera
    aug_prefix = 'bert-base-uncased_avg_augment_multilevel_memory'  # 英文BERT (bert-base-uncased)

# 检查增强文件是否存在
hist_file = os.path.join(data_dir, f'{aug_prefix}.hist')
item_file = os.path.join(data_dir, f'{aug_prefix}.item')

if not (os.path.exists(hist_file) and os.path.exists(item_file)):
    log_print(f"❌ 错误: {aug_prefix} 增强文件不存在")
    log_print(f"   需要的文件: {hist_file}")
    log_print(f"   需要的文件: {item_file}")
    log_print("   请先运行知识编码生成这些文件")
    sys.exit(1)

log_print(f"✅ 使用多级记忆增强: {aug_prefix}")

# 基础训练参数
augment = True
epoch = 30
lr_sched = 'cosine'

# 根据数据集设置不同的metric_scope
if dataset_name == 'mooc':
    metric_scope = '5,10,20'  # MOOC: Rank粗排(50个候选)
else:
    metric_scope = '1,2,3'  # Coursera: 使用默认指标

# 根据数据集调整正则化参数（Coursera严重过拟合，需要超强正则化）
if dataset_name == 'coursera':
    weight_decay = 5e-3  # Coursera: 超强L2正则化 (1e-3 → 5e-3)
    batch_size_list = [256]  # Coursera: 使用与基线相同的批次大小
    lr_list = ['5e-4']  # Coursera: 只用低学习率,避免1e-3过高导致过拟合
    dropout = 0.4  # Coursera: 更强dropout (0.3 → 0.4)
    convert_dropout = 0.3  # Coursera: 更强转换dropout (0.2 → 0.3)
    patience = 3  # Coursera: 降低耐心值,更早停止防过拟合 (8 → 3)
    log_print("📊 Coursera超强正则化: wd=5e-3, dropout=0.4, lr=5e-4, patience=3")
else:  # mooc
    weight_decay = 0  # MOOC: 原始参数（数据集较大，不易过拟合）
    batch_size_list = [256, 512]  # MOOC: 原始批次大小
    lr_list = ['5e-4', '1e-3']  # MOOC: 原始学习率
    dropout = 0.0  # MOOC: 原始dropout
    convert_dropout = 0.0  # MOOC: 原始转换层dropout
    patience = 3  # MOOC: 原始早停耐心
    log_print("📊 使用MOOC原始参数（适合大数据集）")

# 参数搜索网格（根据数据集已设置）
# 模型列表（所有CTR模型均可用于Rank）
model_list = ['DeepFM', 'xDeepFM', 'DCN', 'FiBiNet', 'FiGNN', 'AutoInt', 'DIN', 'DIEN']

# 模型架构参数
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'

# 多级记忆分离参数（支持多头注意力融合）
convert_type = 'MultilevelMemoryHEA'  # 混合专家适配器

# 根据数据集调整模型复杂度
if dataset_name == 'coursera':
    export_num = 1  # Coursera: 减少基础专家数量
    memory_specific_export_num = 2  # Coursera: 减少记忆专用专家数量
    memory_attn_heads = 2  # Coursera: 减少注意力头数
    enable_memory_attention = True
    # Coursera不使用降维,保持原始768维
    enable_knowledge_reduction = False
    knowledge_reduction_dim = 768
    knowledge_reduction_dropout = 0.0
    log_print("🔧 使用简化的多级记忆模块（适合小数据集）")
    log_print("🔧 保持原始768维BERT向量")
else:  # mooc
    export_num = 2  # MOOC: 原始基础专家数量
    memory_specific_export_num = 3  # MOOC: 原始记忆专用专家数量
    memory_attn_heads = 4  # MOOC: 原始注意力头数
    enable_memory_attention = True
    # MOOC数据集不使用降维
    enable_knowledge_reduction = False
    knowledge_reduction_dim = 768
    knowledge_reduction_dropout = 0.0
    log_print("🔧 使用完整的多级记忆模块（适合大数据集）")

# 训练结果记录
results = []

# 循环训练所有模型和参数组合
for model in model_list:
    log_print(f"\n🚀 开始训练模型: {model}")
    log_print("-" * 50)
    
    for batch_size in batch_size_list:
        for lr in lr_list:
            log_print(f"\n📋 参数组合: 批次={batch_size}, 学习率={lr}")
            log_print(f"   🧠 多级记忆: 感觉记忆 + 工作记忆 + 长期记忆")
            log_print(f"   📊 记忆专家数: {memory_specific_export_num}")
            log_print(f"   🎯 多头注意力: {memory_attn_heads}头融合")
            
            # 构造训练命令（支持多头注意力融合）
            cmd = ['python', '-u', 'RS/rank/main_rank_multilevel_memory.py',
                   f'--data_dir={data_dir}',
                   f'--augment={augment}',
                   f'--aug_prefix={aug_prefix}',
                   f'--task={task_name}',
                   f'--convert_arch={convert_arch}',
                   f'--convert_type={convert_type}',
                   f'--convert_dropout={convert_dropout}',
                   f'--epoch_num={epoch}',
                   f'--batch_size={batch_size}',
                   f'--lr={lr}',
                   f'--lr_sched={lr_sched}',
                   f'--weight_decay={weight_decay}',
                   f'--patience={patience}',
                   f'--algo={model}',
                   f'--embed_dim={embed_size}',
                   f'--export_num={export_num}',
                   f'--specific_export_num={memory_specific_export_num}',
                   f'--final_mlp_arch={final_mlp}',
                   f'--dropout={dropout}',
                   f'--metric_scope={metric_scope}',
                   # 🧠 多级记忆核心参数（支持多头注意力）
                   '--memory_mode=true',
                   f'--memory_fusion_type=attention',
                   f'--memory_specific_export_num={memory_specific_export_num}',
                   f'--memory_weight_decay=0.01',
                   # 🎯 多头注意力融合参数
                   '--enable_memory_attention=true',
                   f'--memory_attn_heads={memory_attn_heads}',
                   # 🔧 知识降维参数
                   f'--enable_knowledge_reduction={enable_knowledge_reduction}',
                   f'--knowledge_reduction_dim={knowledge_reduction_dim}',
                   f'--knowledge_reduction_dropout={knowledge_reduction_dropout}'
                   ]
            
            log_print("执行命令:", ' '.join(cmd))
            log_print("-" * 30)
            
            # 运行训练
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip()
                    print(line)
                    print(line, file=log_file)
                    log_file.flush()
                    output_lines.append(line + '\n')
                
            process.wait()
            output = ''.join(output_lines)
            
            # 提取训练结果 - 动态提取metric_scope中指定的K值
            metric_scope_list = [int(x.strip()) for x in metric_scope.split(',')]
            metrics_dict = {}
            for k in metric_scope_list:
                pattern = rf"@{k}, MAP: ([\d\.]+), NDCG: ([\d\.]+), HR: ([\d\.]+)"
                matches = re.findall(pattern, output)
                if matches:
                    # 获取最后一次的结果（最终测试结果）
                    last_match = matches[-1]
                    metrics_dict[k] = {
                        'map': float(last_match[0]),
                        'ndcg': float(last_match[1]),
                        'hr': float(last_match[2])
                    }
            
            # 提取MRR（全局指标，无@K）
            mrr_pattern = r"^MRR: ([\d\.]+)"
            mrr_matches = re.findall(mrr_pattern, output, re.MULTILINE)
            mrr_value = float(mrr_matches[-1]) if mrr_matches else None
            
            # 提取AUC（如果有）
            auc_pattern = r"^AUC: ([\d\.]+)"
            auc_matches = re.findall(auc_pattern, output, re.MULTILINE)
            auc_value = float(auc_matches[-1]) if auc_matches else None
            
            # 记录结果
            # 动态获取main_k作为主要指标
            main_k = metric_scope_list[len(metric_scope_list)//2] if metric_scope_list else 5
            
            result = {
                'model': model,
                'batch_size': batch_size,
                'lr': lr,
                'memory_experts': memory_specific_export_num,
                'attn_heads': memory_attn_heads,
                'metrics': metrics_dict,
                'map@5': metrics_dict.get(5, {}).get('map'),  # 向后兼容
                'ndcg@5': metrics_dict.get(5, {}).get('ndcg'),
                'hr@5': metrics_dict.get(5, {}).get('hr'),
                f'map@{main_k}': metrics_dict.get(main_k, {}).get('map'),  # 动态K值
                f'ndcg@{main_k}': metrics_dict.get(main_k, {}).get('ndcg'),
                f'hr@{main_k}': metrics_dict.get(main_k, {}).get('hr'),
                'mrr': mrr_value,  # MRR是全局指标
                'auc': auc_value,
                'save_dir': save_dir
            }
            results.append(result)
            
            if metrics_dict:
                # 动态获取metric_scope中的中间值作为主要指标显示
                metric_scope_list = [int(x.strip()) for x in metric_scope.split(',')]
                main_k = metric_scope_list[len(metric_scope_list)//2] if metric_scope_list else 5
                
                if main_k in metrics_dict:
                    map_score = metrics_dict[main_k]['map']
                    ndcg_score = metrics_dict[main_k]['ndcg']
                    hr_score = metrics_dict[main_k]['hr']
                    log_print(f"✅ 完成: MAP@{main_k}={map_score:.5f}, NDCG@{main_k}={ndcg_score:.5f}, HR@{main_k}={hr_score:.5f}")
                    if mrr_value is not None:
                        log_print(f"   MRR={mrr_value:.5f}")
                    if auc_value is not None:
                        log_print(f"   AUC={auc_value:.5f}")
                else:
                    log_print(f"   ⚠️  指标@{main_k}不可用")
            else:
                log_print(f"❌ 训练失败或结果解析错误")
            log_print("=" * 30)

# 输出最终结果摘要
log_print(f"\n🏆 多级记忆增强训练结果摘要:")
log_print("=" * 70)

# 动态获取主要K值
metric_scope_list = [int(x.strip()) for x in metric_scope.split(',')]
main_k = metric_scope_list[len(metric_scope_list)//2] if metric_scope_list else 5
map_key = f'map@{main_k}'
ndcg_key = f'ndcg@{main_k}'

# 兼容处理：优先使用@5，否则使用main_k
valid_results = [r for r in results if r.get('map@5') is not None or r.get(map_key) is not None]
if valid_results:
    # 按MAP排序（优先使用@5，否则使用main_k）
    valid_results.sort(key=lambda x: x.get('map@5') or x.get(map_key, 0), reverse=True)
    
    log_print(f"🥇 Top 10 最佳结果 (按MAP@{main_k}排序):")
    for i, result in enumerate(valid_results[:10], 1):
        map_val = result.get('map@5') or result.get(map_key, 0)
        ndcg_val = result.get('ndcg@5') or result.get(ndcg_key, 0)
        log_print(f"{i:2d}. {result['model']:8s} | MAP@{main_k}={map_val:.5f} | "
                  f"批次={result['batch_size']:3d} | 学习率={result['lr']} | "
                  f"记忆专家={result['memory_experts']} | "
                  f"注意力头={result['attn_heads']} | NDCG@{main_k}={ndcg_val:.5f}")
    
    # 每个模型的最佳结果（输出完整指标）
    log_print(f"\n📊 各模型最佳性能 (完整指标):")
    log_print("-" * 70)
    for model in model_list:
        model_results = [r for r in valid_results if r['model'] == model]
        if model_results:
            best = model_results[0]  # 已经按MAP排序
            log_print(f"\n{model}:")
            log_print(f"   参数: 批次={best['batch_size']}, 学习率={best['lr']}, "
                     f"记忆专家={best['memory_experts']}, 注意力头={best['attn_heads']}")
            
            # 输出完整的指标（根据实际的metric_scope）
            if 'metrics' in best and best['metrics']:
                for k in metric_scope_list:
                    if k in best['metrics']:
                        m = best['metrics'][k]
                        log_print(f"   @{k}: MAP={m['map']:.5f}, NDCG={m['ndcg']:.5f}, HR={m['hr']:.5f}")
            else:
                # 兼容旧数据：显示@5或main_k
                map_val = best.get('map@5') or best.get(map_key)
                ndcg_val = best.get('ndcg@5') or best.get(ndcg_key)
                hr_val = best.get('hr@5') or best.get(f'hr@{main_k}')
                if map_val is not None:
                    log_print(f"   MAP@{main_k}={map_val:.5f}, NDCG@{main_k}={ndcg_val:.5f}, HR@{main_k}={hr_val:.5f}")
            
            # MRR和AUC单独输出（全局指标）
            if best.get('mrr') is not None:
                log_print(f"   MRR={best['mrr']:.5f}")
            if best.get('auc') is not None:
                log_print(f"   AUC={best['auc']:.5f}")
    
    # 整体统计
    maps = [r.get('map@5') or r.get(map_key, 0) for r in valid_results]
    avg_map = sum(maps) / len(maps)
    max_map = max(maps)
    min_map = min(maps)
    
    log_print(f"\n📈 性能统计:")
    log_print(f"   平均MAP@{main_k}: {avg_map:.5f}")
    log_print(f"   最高MAP@{main_k}: {max_map:.5f}")
    log_print(f"   最低MAP@{main_k}: {min_map:.5f}")
    log_print(f"   成功训练: {len(valid_results)}/{len(results)} 组合")

else:
    log_print("❌ 没有成功的训练结果")

# 保存结果到JSON文件
results_file = f'multilevel_memory_rank_training_results_{dataset_name}.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log_print(f"\n💾 结果已保存到: {results_file}")
log_print(f"🏁 多级记忆增强Rank训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print("=" * 60)

# 输出认知心理学框架说明
log_print(f"\n🧠 认知心理学多级记忆框架:")
log_print("=" * 50)
log_print("🔥 感觉记忆 (Sensory Memory):")
log_print("   - 即时需求和最近浏览 (3-5次交互)")
log_print("   - 捕获用户的瞬时兴趣和直觉反应")
log_print("")
log_print("⚡ 工作记忆 (Working Memory):")
log_print("   - 当前会话行为模式 (10-15次交互)")
log_print("   - 处理正在进行的学习任务")
log_print("")
log_print("🏗️ 长期记忆 (Long-Term Memory):")
log_print("   - 职业发展方向 (基于领域分布)")
log_print("   - 存储积累的知识和技能")
log_print("=" * 50)

log_file.close()

# nohup python RS/rank/run_rank_multilevel_memory.py --dataset mooc > logs/rank_multilevel_memory.log 2>&1 &
