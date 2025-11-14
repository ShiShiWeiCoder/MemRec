import subprocess
import re
import json
import os
import sys
import argparse
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多级记忆增强Rerank模型')
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
log_file = open(f'{log_dir}/run_rerank_multilevel_memory_{dataset_name}.log', 'w', encoding='utf-8')

def log_print(*args, **kwargs):
    """同时输出到控制台和日志文件"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush()

# 记录开始时间
log_print(f"多级记忆增强Rerank模型训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"数据集: {dataset_name.upper()}")
log_print("=" * 60)

# ---------------------------
# 多级记忆增强训练参数
# ---------------------------
data_dir = f'data/{dataset_name}/proc_data'
task_name = 'rerank'

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

# 根据数据集调整正则化参数（Coursera需要平衡正则化和学习能力）
if dataset_name == 'coursera':
    weight_decay = 1e-3  # Coursera: 适中L2正则化（降低到1e-3，避免过度约束）
    batch_size_list = [256]  # Coursera: 保持较大batch size
    lr_list = ['5e-4', '1e-3']  # Coursera: 使用更大学习率帮助学习知识特征
    dropout = 0.2  # Coursera: 适中dropout（降低到0.2，避免欠拟合）
    convert_dropout = 0.2  # Coursera: 适中转换dropout（降低到0.2）
    patience = 5  # Coursera: 更多耐心（增加到5，给模型更多学习机会）
    metric_scope = '1,2,3,5,7'  # Coursera: 添加@7指标
    log_print("📊 Coursera平衡配置: wd=1e-3, dropout=0.2, convert_dropout=0.2, lr=[5e-4,1e-3], patience=5")
    log_print(f"📊 指标范围: {metric_scope}")
else:  # mooc
    weight_decay = 0  # MOOC: 原始参数（数据集较大，不易过拟合）
    batch_size_list = [256, 512]  # MOOC: 原始批次大小
    lr_list = ['5e-4', '1e-3']  # MOOC: 原始学习率
    dropout = 0.0  # MOOC: 原始dropout
    convert_dropout = 0.0  # MOOC: 原始转换层dropout
    patience = 3  # MOOC: 原始早停耐心
    metric_scope = '1,3,5,7,10'  # MOOC: 完整指标(10个候选)
    log_print("📊 使用MOOC原始参数（适合大数据集）")
    log_print(f"📊 指标范围: {metric_scope}")

# 参数搜索网格（根据数据集已设置）
# 模型列表（7个Rerank模型）
model_list = ['DLCM', 'PRM', 'SetRank', 'MIR', 'GSF', 'EGRerank', 'LambdaRank']

# 模型架构参数（统一使用标准配置）
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'  # 统一使用标准转换维度（768->128->32）

# 多级记忆分离参数（统一使用标准配置）
convert_type = 'HEA'  # 混合专家适配器
export_num = 2  # 基础专家数量
memory_specific_export_num = 3  # 记忆专家数量
memory_attn_heads = 4  # 注意力头数
enable_memory_attention = True  # 启用多头注意力融合

# 温度参数
temperature_list = [0.5, 1.0]

# 训练结果记录
results = []

# 循环训练所有模型和参数组合（简化版）
for model in model_list:
    log_print(f"\n🚀 开始训练模型: {model}")
    log_print("-" * 50)
    
    for batch_size in batch_size_list:
        for lr in lr_list:
            for temperature in temperature_list:
                log_print(f"\n📋 参数组合: 批次={batch_size}, 学习率={lr}, 温度={temperature}")
                log_print(f"   🧠 多级记忆: 感觉记忆 + 工作记忆 + 长期记忆")
                log_print(f"   📊 记忆专家数: {memory_specific_export_num}")
                log_print(f"   🎯 多头注意力: {memory_attn_heads}头融合")
                
                # 构造训练命令（支持多头注意力融合）
                cmd = ['python', '-u', 'RS/rerank/main_rerank_multilevel_memory.py',
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
                       f'--temperature={temperature}',
                       f'--metric_scope={metric_scope}',
                       # 🧠 多级记忆核心参数（支持多头注意力）
                       '--memory_mode=true',
                       f'--memory_fusion_type=attention',
                       f'--memory_specific_export_num={memory_specific_export_num}',
                       f'--memory_weight_decay=0.01',
                       # 🎯 多头注意力融合参数
                       '--enable_memory_attention=true',
                       f'--memory_attn_heads={memory_attn_heads}'
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
                
                # 提取训练结果 - 提取所有k值的最终测试结果（根据数据集动态调整）
                metrics_dict = {}
                if dataset_name == 'coursera':
                    k_values = [1, 2, 3, 5, 7]  # Coursera使用@1,2,3,5,7
                else:  # mooc
                    k_values = [1, 3, 5, 7, 10]  # MOOC使用@1,3,5,7,10
                
                for k in k_values:
                    pattern = rf"@{k}, MAP: ([\d\.]+), NDCG: ([\d\.]+), HR: ([\d\.]+)"
                    matches = re.findall(pattern, output)
                    if matches:
                        # 获取最后一次的结果（最终测试结果）
                        last_match = matches[-1]
                        metrics_dict[k] = {
                            'map': float(last_match[0]),
                            'ndcg': float(last_match[1])
                            # 不再记录HR
                        }
                
                # 提取MRR（全局指标，无@K）
                mrr_pattern = r"^MRR: ([\d\.]+)"
                mrr_matches = re.findall(mrr_pattern, output, re.MULTILINE)
                mrr_value = float(mrr_matches[-1]) if mrr_matches else None
                
                # 记录结果（根据数据集选择主要指标）
                primary_k = 3 if dataset_name == 'coursera' else 5  # Coursera用@3，MOOC用@5
                result = {
                    'model': model,
                    'batch_size': batch_size,
                    'lr': lr,
                    'temperature': temperature,
                    'memory_experts': memory_specific_export_num,
                    'attn_heads': memory_attn_heads,
                    'metrics': metrics_dict,
                    f'map@{primary_k}': metrics_dict.get(primary_k, {}).get('map'),
                    f'ndcg@{primary_k}': metrics_dict.get(primary_k, {}).get('ndcg'),
                    'mrr': mrr_value,
                    'primary_k': primary_k  # 记录主要K值
                }
                results.append(result)
                
                if metrics_dict:
                    map_score = metrics_dict.get(primary_k, {}).get('map')
                    ndcg_score = metrics_dict.get(primary_k, {}).get('ndcg')
                    if map_score is not None:
                        log_print(f"✅ 完成: MAP@{primary_k}={map_score:.5f}, NDCG@{primary_k}={ndcg_score:.5f}")
                        if mrr_value is not None:
                            log_print(f"   MRR={mrr_value:.5f}")
                    else:
                        log_print(f"❌ 训练失败或结果解析错误")
                else:
                    log_print(f"❌ 训练失败或结果解析错误")
                log_print("=" * 30)

# 输出最终结果摘要
log_print(f"\n🏆 多级记忆增强训练结果摘要:")
log_print("=" * 70)

# 根据数据集选择主要指标
primary_k = 3 if dataset_name == 'coursera' else 5
map_key = f'map@{primary_k}'
ndcg_key = f'ndcg@{primary_k}'
hr_key = f'hr@{primary_k}'

valid_results = [r for r in results if r.get(map_key) is not None]
if valid_results:
    # 按主要MAP指标排序
    valid_results.sort(key=lambda x: x[map_key], reverse=True)
    
    log_print(f"🥇 Top 10 最佳结果 (按MAP@{primary_k}排序):")
    for i, result in enumerate(valid_results[:10], 1):
        log_print(f"{i:2d}. {result['model']:8s} | MAP@{primary_k}={result[map_key]:.5f} | "
                  f"批次={result['batch_size']:3d} | 学习率={result['lr']} | "
                  f"温度={result['temperature']:.1f} | 记忆专家={result['memory_experts']} | "
                  f"注意力头={result['attn_heads']} | NDCG@{primary_k}={result[ndcg_key]:.5f}")
    
    # 每个模型的最佳结果（输出完整的@1,3,5,7指标）
    log_print(f"\n📊 各模型最佳性能 (完整指标):")
    log_print("-" * 70)
    for model in model_list:
        model_results = [r for r in valid_results if r['model'] == model]
        if model_results:
            best = model_results[0]  # 已经按MAP@5排序
            log_print(f"\n{model}:")
            log_print(f"   参数: 批次={best['batch_size']}, 学习率={best['lr']}, 温度={best['temperature']}, "
                     f"记忆专家={best['memory_experts']}, 注意力头={best['attn_heads']}")
            
            # 输出完整指标（根据数据集动态调整，不显示HR）
            if 'metrics' in best and best['metrics']:
                display_k_values = [1, 2, 3, 5, 7] if dataset_name == 'coursera' else [1, 3, 5, 7, 10]
                for k in display_k_values:
                    if k in best['metrics']:
                        m = best['metrics'][k]
                        log_print(f"   @{k}: MAP={m['map']:.5f}, NDCG={m['ndcg']:.5f}")
            else:
                pk = best.get('primary_k', primary_k)
                log_print(f"   MAP@{pk}={best.get(f'map@{pk}'):.5f}, NDCG@{pk}={best.get(f'ndcg@{pk}'):.5f}")
            
            # 输出MRR（全局指标）
            if best.get('mrr') is not None:
                log_print(f"   MRR={best['mrr']:.5f}")
    
    # 整体统计
    maps = [r[map_key] for r in valid_results]
    avg_map = sum(maps) / len(maps)
    max_map = max(maps)
    min_map = min(maps)
    
    log_print(f"\n📈 性能统计:")
    log_print(f"   平均MAP@{primary_k}: {avg_map:.5f}")
    log_print(f"   最高MAP@{primary_k}: {max_map:.5f}")
    log_print(f"   最低MAP@{primary_k}: {min_map:.5f}")
    log_print(f"   成功训练: {len(valid_results)}/{len(results)} 组合")

else:
    log_print("❌ 没有成功的训练结果")

# 保存结果到JSON文件
results_file = f'multilevel_memory_rerank_training_results_{dataset_name}.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log_print(f"\n💾 结果已保存到: {results_file}")
log_print(f"🏁 多级记忆增强Rerank训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# nohup python RS/rerank/run_rerank_multilevel_memory.py --dataset mooc > logs/rerank_multilevel_memory.log 2>&1 &

