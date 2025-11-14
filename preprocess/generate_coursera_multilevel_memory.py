#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Coursera数据集的CTR和重排序数据,以及多级记忆增强的提示词
"""

import os
import json
import pickle
import random
from collections import defaultdict

def generate_ctr_data(sequential_data, lm_hist_idx, train_users, test_users, rating_threshold=0):
    """生成CTR(点击率预测)训练和测试数据
    
    格式：[user_id, seq_idx, label]
    其中seq_idx是用户序列中的索引位置（从lm_hist_idx指定的位置开始）
    
    Args:
        sequential_data: 用户序列数据，格式：{user_id: [[items], [ratings]]}
        lm_hist_idx: 每个用户的起始索引，格式：{user_id: start_idx}
        train_users: 训练集用户ID列表
        test_users: 测试集用户ID列表
        rating_threshold: 评分阈值
    """
    print("final loading data")
    
    train_data = []
    test_data = []
    
    # 训练集
    for user in train_users:
        user_str = str(user)
        if user_str not in sequential_data:
            continue
        # sequential_data格式：[[item_seq], [rating_seq]]
        item_seq = sequential_data[user_str][0]  # item sequence
        rating_seq = sequential_data[user_str][1]   # rating/label sequence
        
        # 从lm_hist_idx指定的位置开始生成样本（确保有足够的历史记录）
        # lm_hist_idx存储的是历史项目列表，其长度决定起始索引
        hist_idx_data = lm_hist_idx.get(user_str, [])
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            train_data.append([int(user), idx, label])
    
    # 测试集
    for user in test_users:
        user_str = str(user)
        if user_str not in sequential_data:
            continue
        # sequential_data格式：[[item_seq], [rating_seq]]
        item_seq = sequential_data[user_str][0]  # item sequence
        rating_seq = sequential_data[user_str][1]   # rating/label sequence
        
        # 从lm_hist_idx指定的位置开始生成样本
        # lm_hist_idx存储的是历史项目列表，其长度决定起始索引
        hist_idx_data = lm_hist_idx.get(user_str, [])
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            test_data.append([int(user), idx, label])
    
    print(f"generating ctr train dataset user num {len(train_users)} data num {len(train_data)} " +
          f"pos ratio {sum([d[2] for d in train_data]) / len(train_data)}")
    print(train_data[:5])
    
    print(f"generating ctr test dataset user num {len(test_users)} data num {len(test_data)} " +
          f"pos ratio {sum([d[2] for d in test_data]) / len(test_data)}")
    print(test_data[:5])
    
    return train_data, test_data

def generate_rank_data(sequential_data, train_users, test_users, lm_hist_idx, rank_num=15, min_hist_len=5, min_rank_num=10):
    """
    生成Rank阶段（粗排）的训练数据
    与Rerank的区别：候选数量更多（15个 vs 10个）
    注意：Coursera数据集用户序列较短，使用15个候选更合理
    
    Args:
        sequential_data: 用户序列数据
        train_users: 训练集用户ID列表
        test_users: 测试集用户ID列表
        lm_hist_idx: 每个用户的历史索引（确保有足够的历史）
        rank_num: 每个样本的目标候选项数量（粗排用15，适合Coursera短序列）
        min_hist_len: 最小历史长度（用于确保有足够的历史）
        min_rank_num: 最小候选项数量（少于此数量则不生成样本）
    """
    train_data = []
    test_data = []
    
    # 训练集
    for user in train_users:
        user = str(user)
        if user not in sequential_data:
            continue
        courses = sequential_data[user][0]
        labels = sequential_data[user][1]
        
        if user in lm_hist_idx:
            hist_idx_data = lm_hist_idx[user]
            start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
            start_idx = max(start_idx, min_hist_len)
        else:
            start_idx = min_hist_len
        
        # 每8个课程生成一个粗排样本（比rerank稀疏）
        for i in range(start_idx, len(courses), 8):
            available_items = len(courses) - i
            if available_items < min_rank_num:
                break
            actual_rank_num = min(rank_num, available_items)
            candidate_courses = courses[i:i+actual_rank_num]
            candidate_labels = labels[i:i+actual_rank_num]
            train_data.append([int(user), i, candidate_courses, candidate_labels])
    
    # 测试集
    for user in test_users:
        user = str(user)
        if user not in sequential_data:
            continue
        courses = sequential_data[user][0]
        labels = sequential_data[user][1]
        
        if user in lm_hist_idx:
            hist_idx_data = lm_hist_idx[user]
            start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
            start_idx = max(start_idx, min_hist_len)
        else:
            start_idx = min_hist_len
        
        for i in range(start_idx, len(courses), 8):
            available_items = len(courses) - i
            if available_items < min_rank_num:
                break
            actual_rank_num = min(rank_num, available_items)
            candidate_courses = courses[i:i+actual_rank_num]
            candidate_labels = labels[i:i+actual_rank_num]
            test_data.append([int(user), i, candidate_courses, candidate_labels])
    
    print(f"generating ranking train dataset (粗排) user num {len(train_users)} data num {len(train_data)}")
    print('Rank train sample:', train_data[:2])
    
    print(f"generating ranking test dataset (粗排) user num {len(test_users)} data num {len(test_data)}")
    print('Rank test sample:', test_data[:2])
    
    return train_data, test_data


def generate_rerank_data(sequential_data, train_users, test_users, lm_hist_idx, rerank_num=10, min_hist_len=5, min_rerank_num=5):
    """生成重排序训练和测试数据
    
    注意：seq_idx 应该是在 item_seq 中的索引位置，不是课程ID！
    为了确保每个样本都有足够的历史，从 lm_hist_idx 指定的位置开始生成样本。
    
    Args:
        sequential_data: 用户序列数据
        train_users: 训练集用户ID列表
        test_users: 测试集用户ID列表
        lm_hist_idx: 每个用户的历史索引（确保有足够的历史）
        rerank_num: 每个样本的目标候选项数量
        min_hist_len: 最小历史长度（用于确保有足够的历史）
        min_rerank_num: 最小候选项数量（少于此数量则不生成样本）
    """
    train_data = []
    test_data = []
    
    # 训练集
    for user in train_users:
        user = str(user)
        if user not in sequential_data:
            continue
        # sequential_data格式：[[item_seq], [rating_seq]]
        courses = sequential_data[user][0]  # item sequence
        labels = sequential_data[user][1]   # rating/label sequence
        
        # 确定起始位置：使用 lm_hist_idx 的长度（确保有足够的历史）
        if user in lm_hist_idx:
            hist_idx_data = lm_hist_idx[user]
            start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
            # 确保至少有 min_hist_len 的历史
            start_idx = max(start_idx, min_hist_len)
        else:
            start_idx = min_hist_len
        
        # 每4个课程生成一个重排序样本，从 start_idx 开始
        for i in range(start_idx, len(courses), 4):
            # 允许候选项数量少于 rerank_num，但至少要有 min_rerank_num 个
            available_items = len(courses) - i
            if available_items < min_rerank_num:
                break
            # 取最多 rerank_num 个候选项
            actual_rerank_num = min(rerank_num, available_items)
            candidate_courses = courses[i:i+actual_rerank_num]
            candidate_labels = labels[i:i+actual_rerank_num]
            # seq_idx 应该是索引位置 i，不是 courses[i]（课程ID）
            train_data.append([int(user), i, candidate_courses, candidate_labels])
    
    # 测试集
    for user in test_users:
        user = str(user)
        if user not in sequential_data:
            continue
        # sequential_data格式：[[item_seq], [rating_seq]]
        courses = sequential_data[user][0]  # item sequence
        labels = sequential_data[user][1]   # rating/label sequence
        
        # 确定起始位置：使用 lm_hist_idx 的长度（确保有足够的历史）
        if user in lm_hist_idx:
            hist_idx_data = lm_hist_idx[user]
            start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
            # 确保至少有 min_hist_len 的历史
            start_idx = max(start_idx, min_hist_len)
        else:
            start_idx = min_hist_len
        
        # 每4个课程生成一个重排序样本，从 start_idx 开始
        for i in range(start_idx, len(courses), 4):
            # 允许候选项数量少于 rerank_num，但至少要有 min_rerank_num 个
            available_items = len(courses) - i
            if available_items < min_rerank_num:
                break
            # 取最多 rerank_num 个候选项
            actual_rerank_num = min(rerank_num, available_items)
            candidate_courses = courses[i:i+actual_rerank_num]
            candidate_labels = labels[i:i+actual_rerank_num]
            # seq_idx 应该是索引位置 i，不是 courses[i]（课程ID）
            test_data.append([int(user), i, candidate_courses, candidate_labels])
    
    print(f"generating reranking train dataset (精排) user num {len(train_users)} data num {len(train_data)}")
    print('Rerank train sample:', train_data[:2])
    
    print(f"generating reranking test dataset (精排) user num {len(test_users)} data num {len(test_data)}")
    print('Rerank test sample:', test_data[:2])
    
    return train_data, test_data

def generate_item_prompt_multilevel_memory(item2attributes, itemid2title, id2item, multilevel_memory):
    """生成基于多级记忆框架的课程认知属性分析提示词"""
    print("generating multilevel memory enhanced item prompt")
    print("=== DEBUG: generate_item_prompt_multilevel_memory ===")
    print(f"itemid2title keys (first 5): {list(itemid2title.keys())[:5]}")
    
    item_prompts = {}
    
    for item_id, attrs in item2attributes.items():
        title = itemid2title.get(item_id, f"Course {item_id}")
        
        # 获取课程领域(从attrs中提取,如果有的话)
        # 注意: Coursera数据集的attrs是技能列表,这里简化处理
        if attrs and len(attrs) > 0:
            # 假设第一个技能可以代表领域
            domain = "related domain"  # Coursera数据集可能没有明确的领域字段
            prompt = f"Introduce course {title} in the {domain} and describe "
        else:
            prompt = f"Introduce course {title} and describe "
        
        # 构建提示词 - 突出先修关系
        prompt += (
            f"its cognitive attributes from the Atkinson-Shiffrin Memory Model perspective considering "
            f"SENSORY MEMORY impact (immediate appeal and first impressions), "
            f"WORKING MEMORY demands (cognitive load and practical skill building), and "
            f"LONG-TERM MEMORY value (career development and domain expertise). "
            f"Particularly emphasize the prerequisite knowledge requirements and prerequisite course dependencies, "
            f"as these are unique characteristics of courses that determine learning progression and memory consolidation pathways. "
            f"Explain how prerequisites relate to different memory levels and learning readiness. "
            f"Your response must be in English without subtitles, bullet points, or numbered lists."
        )
        
        item_prompts[item_id] = prompt
    
    print(f"data num {len(item_prompts)}")
    
    if item_prompts:
        first_key = list(item_prompts.keys())[0]
        print(f"Sample item prompt preview: {item_prompts[first_key][:200]}...")
    
    return item_prompts

def generate_hist_prompt_multilevel_memory(sequential_data, multilevel_memory, id2user, itemid2title, item2attributes, id2item):
    """生成基于Atkinson-Shiffrin记忆模型的用户多级记忆分析提示词 - 只输入三级记忆,不包含完整历史"""
    print("generating multilevel memory enhanced history prompt")
    print("=== DEBUG: generate_hist_prompt_multilevel_memory ===")
    print(f"multilevel_memory_data keys (first 5): {list(multilevel_memory.keys())[:5]}")
    
    hist_prompts = {}
    
    # 构建原始ID到映射ID的反向映射
    item2id = {v: k for k, v in id2item.items()}
    
    for user_id in sequential_data.keys():
        if user_id not in multilevel_memory:
            continue
        
        memory = multilevel_memory[user_id]
        
        # 获取课程标题
        def get_course_title(original_cid):
            # Coursera的multilevel_memory中存储的是原始课程ID，需要映射到内部ID
            original_cid_str = str(original_cid)
            mapped_id = item2id.get(original_cid_str)
            if mapped_id:
                return itemid2title.get(mapped_id, f"Course {original_cid}")
            return f"Course {original_cid}"
        
        # 获取多级记忆的课程标题
        # Coursera数据结构: sensory_memory是[course_ids]的列表（原始ID）
        sensory_titles = [f'"{get_course_title(c)}"' for c in memory.get('sensory_memory', [])[:5]]
        working_titles = [f'"{get_course_title(c)}"' for c in memory.get('working_memory', [])[:10]]
        longterm_titles = [f'"{get_course_title(c)}"' for c in memory.get('longterm_memory', [])[:10]]
        
        # 构建提示词 - 只包含用户特征和三级记忆,不包含item信息
        prompt = "Given a user, this user's course selections are organized by the Atkinson-Shiffrin Memory Model into three levels: "
        
        if sensory_titles:
            prompt += f"SENSORY MEMORY (immediate exploration needs): {', '.join(sensory_titles)}; "
        
        if working_titles:
            prompt += f"WORKING MEMORY (current learning session and short-term skill goals): {', '.join(working_titles)}; "
        
        if longterm_titles:
            prompt += f"LONG-TERM MEMORY (strategic career planning): {', '.join(longterm_titles)}. "
        
        # 分析要求 - 强调consider各种因素
        prompt += (
            "Analyze this user's learning preferences considering factors such as subject domain, instructional approach, "
            "complexity level, pacing and duration, depth versus breadth, assessment methods, and real-world applications. "
            "Provide clear explanations based on the multilevel memory patterns. "
            "Your response must be in English without subtitles, bullet points, or numbered lists."
        )
        
        hist_prompts[user_id] = prompt
    
    print(f"data num {len(hist_prompts)}")
    
    if hist_prompts:
        first_key = list(hist_prompts.keys())[0]
        print(f"Sample prompt preview: {hist_prompts[first_key][:200]}...")
    
    return hist_prompts

def generate_multilevel_memory_analysis_prompt(multilevel_memory, itemid2title, id2item):
    """生成多级记忆层次对比分析提示词 - 用于分析不同记忆层次之间的交互作用和认知处理机制"""
    print("generating multilevel memory analysis prompt")
    print("=== DEBUG: generate_multilevel_memory_analysis_prompt ===")
    
    analysis_prompts = {}
    
    # 构建原始ID到映射ID的反向映射
    item2id = {v: k for k, v in id2item.items()}
    
    for user_id, memory in multilevel_memory.items():
        # 获取课程标题
        def get_course_title(original_cid):
            # Coursera的multilevel_memory中存储的是原始课程ID，需要映射到内部ID
            original_cid_str = str(original_cid)
            mapped_id = item2id.get(original_cid_str)
            if mapped_id:
                return itemid2title.get(mapped_id, f"Course {original_cid}")
            return f"Course {original_cid}"
        
        # Coursera数据结构: sensory_memory是[course_ids]的列表（原始ID）
        sensory_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('sensory_memory', [])[:6]])
        working_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('working_memory', [])[:8]])
        longterm_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('longterm_memory', [])[:8]])
        
        # 构建提示词 - 只包含用户特征和三级记忆,不包含item信息
        prompt = "Given a user, this user's learning behaviors are categorized by the Atkinson-Shiffrin Memory Model: "
        
        if sensory_titles:
            prompt += f"SENSORY MEMORY (immediate browsing): {sensory_titles}; "
        
        if working_titles:
            prompt += f"WORKING MEMORY (current learning session): {working_titles}; "
        
        if longterm_titles:
            prompt += f"LONG-TERM MEMORY (strategic interests): {longterm_titles}. "
        
        # 对比分析要求 - 重点在于对比三级记忆之间的关系,而非consider各种因素
        prompt += (
            "Compare and contrast the three memory levels to reveal the cognitive processing hierarchy. "
            "Explain how SENSORY MEMORY courses differ from WORKING MEMORY courses in terms of exploration versus consolidation. "
            "Analyze how WORKING MEMORY courses transition into LONG-TERM MEMORY for career planning. "
            "Identify patterns in memory consolidation and learning progression across the three levels. "
            "Describe the interactions and dependencies between different memory levels in shaping learning trajectories. "
            "Your response must be in English without subtitles, bullet points, or numbered lists."
        )
        
        analysis_prompts[user_id] = prompt
    
    print(f"multilevel memory analysis prompts num {len(analysis_prompts)}")
    
    if analysis_prompts:
        first_key = list(analysis_prompts.keys())[0]
        print(f"Sample memory analysis prompt preview: {analysis_prompts[first_key][:200]}...")
    
    return analysis_prompts

def save_data(output_dir, train_ctr, test_ctr, train_rank, test_rank, train_rerank, test_rerank, 
              item_prompts, hist_prompts, analysis_prompts):
    """保存生成的数据"""
    print("save ctr data")
    
    # 保存CTR数据为pickle格式（与mooc数据集保持一致）
    with open(os.path.join(output_dir, 'ctr.train'), 'wb') as f:
        pickle.dump(train_ctr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'ctr.test'), 'wb') as f:
        pickle.dump(test_ctr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("save ranking data (粗排)")
    
    # 保存粗排数据为pickle格式
    with open(os.path.join(output_dir, 'rank.train'), 'wb') as f:
        pickle.dump(train_rank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rank.test'), 'wb') as f:
        pickle.dump(test_rank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("save reranking data (精排)")
    
    # 保存重排序数据为pickle格式
    with open(os.path.join(output_dir, 'rerank.train'), 'wb') as f:
        pickle.dump(train_rerank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rerank.test'), 'wb') as f:
        pickle.dump(test_rerank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("save prompt data")
    
    # 保存提示词
    with open(os.path.join(output_dir, 'prompt.item.multilevel_memory'), 'w', encoding='utf-8') as f:
        json.dump(item_prompts, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'prompt.hist.multilevel_memory'), 'w', encoding='utf-8') as f:
        json.dump(hist_prompts, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'prompt.memory_analysis'), 'w', encoding='utf-8') as f:
        json.dump(analysis_prompts, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 多级记忆增强的数据生成完成！")
    print("生成的文件:")
    print("  - ctr.train/test: CTR训练和测试数据")
    print("  - rank.train/test: Rank粗排训练和测试数据（50个候选）")
    print("  - rerank.train/test: Rerank精排训练和测试数据（10个候选）")
    print("  - prompt.item.multilevel_memory: 多级记忆增强的课程提示词")
    print("  - prompt.hist.multilevel_memory: 多级记忆增强的历史提示词")
    print("  - prompt.memory_analysis: 专门的多级记忆分析提示词")

def main():
    # 数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'coursera', 'proc_data')
    
    # 加载数据
    with open(os.path.join(data_dir, 'sequential_data.json'), 'r', encoding='utf-8') as f:
        sequential_data = json.load(f)
    
    with open(os.path.join(data_dir, 'item2attributes.json'), 'r', encoding='utf-8') as f:
        item2attributes = json.load(f)
    
    with open(os.path.join(data_dir, 'datamaps.json'), 'r', encoding='utf-8') as f:
        datamaps = json.load(f)
        id2user = datamaps['id2user']
        id2item = datamaps['id2item']
        itemid2title = datamaps['itemid2title']
    
    with open(os.path.join(data_dir, 'train_test_split.json'), 'r', encoding='utf-8') as f:
        train_test_split = json.load(f)
        train_users = train_test_split['train']
        test_users = train_test_split['test']
        lm_hist_idx = train_test_split['lm_hist_idx']
    
    with open(os.path.join(data_dir, 'multilevel_memory.json'), 'r', encoding='utf-8') as f:
        multilevel_memory = json.load(f)
    
    # 生成CTR数据
    train_ctr, test_ctr = generate_ctr_data(sequential_data, lm_hist_idx, train_users, test_users)
    
    # 生成粗排数据（传递 lm_hist_idx 以确保每个样本都有足够的历史）
    train_rank, test_rank = generate_rank_data(sequential_data, train_users, test_users, lm_hist_idx)
    
    # 生成精排数据（传递 lm_hist_idx 以确保每个样本都有足够的历史）
    train_rerank, test_rerank = generate_rerank_data(sequential_data, train_users, test_users, lm_hist_idx)
    
    # 生成多级记忆增强的提示词
    item_prompts = generate_item_prompt_multilevel_memory(item2attributes, itemid2title, id2item, multilevel_memory)
    hist_prompts = generate_hist_prompt_multilevel_memory(sequential_data, multilevel_memory, id2user, itemid2title, item2attributes, id2item)
    analysis_prompts = generate_multilevel_memory_analysis_prompt(multilevel_memory, itemid2title, id2item)
    
    # 保存数据
    save_data(data_dir, train_ctr, test_ctr, train_rank, test_rank, train_rerank, test_rerank,
              item_prompts, hist_prompts, analysis_prompts)

if __name__ == '__main__':
    main()

