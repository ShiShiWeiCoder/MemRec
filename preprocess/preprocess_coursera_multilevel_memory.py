#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coursera数据集多级记忆预处理脚本
支持感觉记忆(Sensory Memory)、工作记忆(Working Memory)、长期记忆(Long-term Memory)
"""

import os
import json
import random
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Coursera dataset with multilevel memory')
    parser.add_argument('--k_core_user', type=int, default=3, help='k-core for user filtering (Coursera数据较稀疏，使用较小值)')
    parser.add_argument('--k_core_item', type=int, default=3, help='k-core for item filtering (Coursera数据较稀疏，使用较小值)')
    # Coursera序列较短（平均6.25，中位数6），调整三级记忆划分：
    # - 感觉记忆1个（~17%）: 最近的即时需求
    # - 工作记忆2个（~33%）: 当前学习路径  
    # - 长期记忆全部（100%）: 职业发展方向
    parser.add_argument('--sensory_memory_len', type=int, default=1, help='Length of sensory memory (Coursera序列短，设为1)')
    parser.add_argument('--working_memory_len', type=int, default=2, help='Length of working memory (Coursera序列短，设为2)')
    parser.add_argument('--longterm_memory_domains', type=int, default=3, help='Number of domains in long-term memory')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--neg_ratio', type=int, default=1, help='Negative sampling ratio (pos:neg = 2:1)')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    return parser.parse_args()

def load_coursera_data(data_dir):
    """加载Coursera数据集
    
    返回:
        enrolled_df: 用户学习历史（enrolled_course.csv）
        meta_df: 课程元数据（Coursera_2.csv）
    """
    print("正在加载Coursera课程数据...")
    
    # 加载用户学习历史 - 尝试多种编码
    enrolled_file = os.path.join(data_dir, 'enrolled_course.csv')
    try:
        enrolled_df = pd.read_csv(enrolled_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            enrolled_df = pd.read_csv(enrolled_file, encoding='latin1')
        except:
            enrolled_df = pd.read_csv(enrolled_file, encoding='ISO-8859-1')
    print(f"加载了 {len(enrolled_df)} 个用户的学习历史")
    
    # 加载课程元数据 - 尝试多种编码
    meta_file = os.path.join(data_dir, 'Coursera_2.csv')
    try:
        meta_df = pd.read_csv(meta_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            meta_df = pd.read_csv(meta_file, encoding='latin1')
        except:
            meta_df = pd.read_csv(meta_file, encoding='ISO-8859-1')
    print(f"加载了 {len(meta_df)} 门课程的元数据（Coursera_2.csv）")
    
    return enrolled_df, meta_df

def build_interaction_data(enrolled_df, meta_df):
    """构建用户-课程交互数据
    
    注意：enrolled_df中的History_course_id是字符串格式（如'0687'），需要转换为整数与meta_df的id列匹配
    """
    print("\n正在构建用户-课程交互数据...")
    
    user_history = {}
    # meta_df['id']是整数，需要将History_course_id转换为整数后匹配
    valid_course_ids = set(meta_df['id'].values)
    
    total_courses = 0
    invalid_courses = 0
    
    for _, row in enrolled_df.iterrows():
        user_id = row['User_id']
        course_ids_str = str(row['History_course_id']).split(', ')
        
        # 将字符串ID（如'0687'）转换为整数（687）
        valid_courses = []
        for cid_str in course_ids_str:
            try:
                cid_int = int(cid_str)  # '0687' -> 687
                total_courses += 1
                if cid_int in valid_course_ids:
                    valid_courses.append(str(cid_int))  # 存储为字符串以保持一致性
                else:
                    invalid_courses += 1
            except ValueError:
                invalid_courses += 1
                continue
        
        if valid_courses:
            user_history[user_id] = valid_courses
    
    print(f"构建完成: {len(user_history)} 个用户, 平均每用户 {np.mean([len(v) for v in user_history.values()]):.2f} 门课程")
    print(f"总课程记录: {total_courses}, 有效: {total_courses - invalid_courses}, 无效: {invalid_courses}")
    return user_history

def k_core_filter(user_history, k_core_user=20, k_core_item=10):
    """K-core过滤: 过滤掉交互次数少的用户和课程"""
    print(f"\n开始进行 {k_core_user}-core 用户和 {k_core_item}-core 课程过滤...")
    
    while True:
        # 统计课程出现次数
        item_count = defaultdict(int)
        for courses in user_history.values():
            for course in courses:
                item_count[course] += 1
        
        # 过滤低频课程
        user_history_new = {}
        for user, courses in user_history.items():
            new_courses = [c for c in courses if item_count[c] >= k_core_item]
            if len(new_courses) >= k_core_user:
                user_history_new[user] = new_courses
        
        # 检查是否收敛
        if len(user_history_new) == len(user_history):
            break
        user_history = user_history_new
    
    # 统计课程数量
    all_courses = set()
    for courses in user_history.values():
        all_courses.update(courses)
    
    print(f"K-core过滤后: {len(user_history)} 用户, {len(all_courses)} 课程")
    return user_history

def extract_multilevel_memory(user_history, meta_df, args):
    """提取多级记忆: 感觉记忆、工作记忆、长期记忆
    
    Args:
        user_history: 用户课程历史（课程ID是字符串格式的整数）
        meta_df: Coursera_2.csv，包含id和Skills列
        args: 命令行参数
    """
    print("\n开始提取多级记忆：感觉记忆、工作记忆、长期记忆...")
    print("注意：多级记忆提取基于K-core过滤后的核心数据集")
    
    multilevel_memory = {}
    course_domain_map = {}
    
    # 构建课程-领域映射(使用Skills字段)
    for _, row in meta_df.iterrows():
        course_id_int = int(row['id'])  # meta_df的id是整数
        course_id_str = str(course_id_int)  # 转换为字符串以匹配user_history中的格式
        skills = str(row['Skills']) if pd.notna(row['Skills']) else ''
        # Skills可能是逗号分隔的，提取前3个技能作为领域
        if skills and skills != 'nan':
            skill_list = [s.strip() for s in skills.split(',') if s.strip()]
            domains = skill_list[:3] if skill_list else ['Unknown']
        else:
            domains = ['Unknown']
        course_domain_map[course_id_str] = domains
    
    sensory_lens = []
    working_lens = []
    longterm_lens = []
    longterm_domain_nums = []
    
    for user_id, courses in user_history.items():
        total_len = len(courses)
        
        # 感觉记忆: 最近的即时需求 (最后N个课程)
        sensory_memory = courses[-args.sensory_memory_len:] if total_len >= args.sensory_memory_len else courses
        
        # 工作记忆: 当前会话行为模式 (中间N个课程)
        if total_len > args.sensory_memory_len + args.working_memory_len:
            working_memory = courses[-(args.sensory_memory_len + args.working_memory_len):-args.sensory_memory_len]
        elif total_len > args.sensory_memory_len:
            working_memory = courses[:-args.sensory_memory_len]
        else:
            working_memory = []
        
        # 长期记忆: 职业发展方向 (所有课程,按领域聚合)
        longterm_memory = courses
        
        # 提取长期记忆的领域信息
        domain_count = defaultdict(int)
        for course in longterm_memory:
            if course in course_domain_map:
                for domain in course_domain_map[course]:
                    domain_count[domain] += 1
        
        # 选择top-K领域作为长期记忆的领域特征
        top_domains = sorted(domain_count.items(), key=lambda x: x[1], reverse=True)
        longterm_domains = [d[0] for d in top_domains[:args.longterm_memory_domains]]
        
        multilevel_memory[user_id] = {
            'sensory_memory': sensory_memory,
            'working_memory': working_memory,
            'longterm_memory': longterm_memory,
            'longterm_domains': longterm_domains
        }
        
        sensory_lens.append(len(sensory_memory))
        working_lens.append(len(working_memory))
        longterm_lens.append(len(longterm_memory))
        longterm_domain_nums.append(len(longterm_domains))
    
    print(f"多级记忆提取完成:")
    print(f"  感觉记忆平均长度: {np.mean(sensory_lens):.2f} (最近即时需求)")
    print(f"  工作记忆平均长度: {np.mean(working_lens):.2f} (当前会话行为模式)")
    print(f"  长期记忆平均长度: {np.mean(longterm_lens):.2f} (职业发展方向)")
    print(f"  长期记忆领域平均数: {np.mean(longterm_domain_nums):.2f}")
    
    return multilevel_memory, course_domain_map

def add_negative_samples(user_history, all_courses, neg_ratio=1, seed=2023):
    """为每个用户添加负样本（与MOOC保持一致：正负样本打乱顺序）"""
    print("\n开始添加负样本...")
    random.seed(seed)
    
    sequence_data = {}
    for user_id, pos_courses in user_history.items():
        # 添加正样本
        pos_set = set(pos_courses)
        all_items = [(course, 1) for course in pos_courses]
        
        # 添加负样本
        neg_candidates = list(all_courses - pos_set)
        num_neg = len(pos_courses) // (neg_ratio + 1)  # pos:neg = 2:1
        neg_samples = random.sample(neg_candidates, min(num_neg, len(neg_candidates)))
        all_items.extend([(course, 0) for course in neg_samples])
        
        # 打乱顺序以模拟真实的交互序列（与MOOC一致）
        random.shuffle(all_items)
        
        # 分离courses和labels
        user_courses = [item[0] for item in all_items]
        user_labels = [item[1] for item in all_items]
        
        sequence_data[user_id] = {
            'courses': user_courses,
            'labels': user_labels
        }
    
    total_pos = sum([sum(v['labels']) for v in sequence_data.values()])
    total_neg = sum([len(v['labels']) - sum(v['labels']) for v in sequence_data.values()])
    print(f"添加负样本完成，正:负 = {neg_ratio + 1}:1")
    print(f"总正样本: {total_pos}, 总负样本: {total_neg}")
    
    return sequence_data

def create_id_mapping(user_history):
    """创建用户ID和课程ID的映射
    
    Args:
        user_history: 用户课程历史（课程ID已经是字符串格式的整数）
    """
    print("\n开始ID映射...")
    
    # 用户ID映射
    user_ids = sorted(list(user_history.keys()))
    id2user = {i: uid for i, uid in enumerate(user_ids)}
    user2id = {uid: i for i, uid in id2user.items()}
    
    # 课程ID映射
    all_courses = set()
    for courses in user_history.values():
        all_courses.update(courses)
    course_ids = sorted(list(all_courses))
    id2item = {i: cid for i, cid in enumerate(course_ids)}
    item2id = {cid: i for i, cid in id2item.items()}
    
    # 将sequence_data的ID转换为整数，格式与MOOC保持一致：[[item_seq], [rating_seq]]
    sequence_data_mapped = {}
    for user_id, data in user_history.items():
        new_user_id = user2id[user_id]
        # MOOC格式：[item_list, rating_list]
        sequence_data_mapped[str(new_user_id)] = [
            [item2id[c] for c in data],  # item sequence
            [1] * len(data)  # rating sequence (all positive for original data)
        ]
    
    print(f"ID映射完成: {len(user2id)} 用户, {len(item2id)} 课程")
    
    return user2id, item2id, id2user, id2item, sequence_data_mapped

def process_course_attributes(meta_df, item2id, course_domain_map):
    """处理课程属性信息
    
    Args:
        meta_df: Coursera_2.csv数据，包含id, Course Name, Skills等列
        item2id: 课程ID到内部ID的映射（键是字符串格式的整数ID，如'687'）
        course_domain_map: 课程领域映射
    """
    print("\n开始处理属性信息...")
    
    # 构建课程属性字典
    item2attributes = {}
    itemid2title = {}
    
    # 第一步：收集实际会被使用的技能（只收集item2id中课程的技能）
    used_skills = set()
    for course_id_str in item2id.keys():
        course_id_int = int(course_id_str)
        course_info = meta_df[meta_df['id'] == course_id_int]
        
        if len(course_info) > 0:
            row = course_info.iloc[0]
            skills = str(row['Skills']) if pd.notna(row['Skills']) else ''
            if skills and skills != 'nan':
                for skill in skills.split(','):
                    skill = skill.strip()
                    if skill:
                        used_skills.add(skill)
    
    # 第二步：只为实际使用的技能创建连续的ID映射（从0开始）
    skill2id = {skill: i for i, skill in enumerate(sorted(used_skills))}
    print(f"收集到 {len(used_skills)} 个实际使用的技能")
    
    # 第三步：为每个映射后的课程添加信息
    for course_id_str, internal_idx in item2id.items():
        # course_id_str是字符串格式的整数（如'687'）
        course_id_int = int(course_id_str)
        
        # 在meta_df中查找课程信息（meta_df的id列是整数）
        course_info = meta_df[meta_df['id'] == course_id_int]
        
        if len(course_info) > 0:
            row = course_info.iloc[0]
            course_name = str(row['Course Name']) if pd.notna(row['Course Name']) else f"Course {course_id_str}"
            skills = str(row['Skills']) if pd.notna(row['Skills']) else ''
            
            # 提取技能属性ID
            skill_ids = []
            if skills and skills != 'nan':
                for skill in skills.split(','):
                    skill = skill.strip()
                    if skill and skill in skill2id:
                        skill_ids.append(skill2id[skill])
            
            if not skill_ids:
                skill_ids = [0]  # 默认属性
            
            item2attributes[str(internal_idx)] = skill_ids
            itemid2title[str(internal_idx)] = course_name
        else:
            # 在Coursera_2.csv中找不到该课程
            print(f"警告: 课程ID {course_id_str} 在Coursera_2.csv中未找到")
            item2attributes[str(internal_idx)] = [0]
            itemid2title[str(internal_idx)] = f"Unknown Course {course_id_str}"
    
    # 统计属性信息
    all_attrs = set()
    for attrs in item2attributes.values():
        all_attrs.update(attrs)
    
    attr_lens = [len(attrs) for attrs in item2attributes.values()]
    
    print(f"实际使用的技能数量: {len(used_skills)}")
    print(f"属性处理后数量: {len(all_attrs)}")
    print(f"属性ID范围: 0 ~ {max(all_attrs) if all_attrs else 0}")
    
    if attr_lens:
        print(f"属性长度统计, 最小: {min(attr_lens)}, 最大: {max(attr_lens)}, 平均: {np.mean(attr_lens):.4f}")
    else:
        print("警告: 没有属性数据")
    
    print(f"itemid2title中存储了 {len(itemid2title)} 个课程的标题")
    print(f"id2item中有 {len(item2id)} 个课程ID")
    
    # 确认所有课程ID都有对应的标题
    if item2attributes:
        missing_titles = [k for k in item2attributes.keys() if k not in itemid2title]
        unknown_titles = [k for k, v in itemid2title.items() if v.startswith('Unknown Course')]
        if missing_titles:
            print(f"警告: {len(missing_titles)} 个课程缺少标题")
        if unknown_titles:
            print(f"警告: {len(unknown_titles)} 个课程使用Unknown标题")
        if not missing_titles and not unknown_titles:
            print("确认: 所有课程ID都有有效的标题")
    else:
        print("警告: 没有课程数据")
    
    return item2attributes, itemid2title, skill2id

def split_train_test(sequence_data, multilevel_memory, user2id, test_ratio=0.1, fixed_hist_len=3, seed=2023):
    """划分训练集和测试集
    
    Args:
        sequence_data: 用户序列数据，格式：{user_id: [[items], [ratings]]}
        multilevel_memory: 多级记忆数据
        user2id: 用户ID映射
        test_ratio: 测试集比例
        min_hist_len: 最小历史长度，用于lm_hist_idx（确保有足够的历史记录）
        seed: 随机种子
    """
    print("\n开始划分训练集和测试集...")
    random.seed(seed)
    
    all_users = list(sequence_data.keys())
    random.shuffle(all_users)
    
    num_test = int(len(all_users) * test_ratio)
    test_users = [int(u) for u in all_users[:num_test]]
    train_users = [int(u) for u in all_users[num_test:]]
    
    # 生成lm_hist_idx（语言模型历史索引）
    # 与MOOC格式保持一致：所有用户使用固定长度的历史列表
    # 这确保所有CTR样本的历史长度一致，避免batch collate时的张量大小不匹配
    lm_hist_idx = {}
    for user_str in sequence_data.keys():
        user_id = int(user_str)
        item_seq = sequence_data[user_str][0]  # item序列
        seq_len = len(item_seq)
        
        # 使用固定历史长度（与MOOC一致）
        # 如果序列太短，用序列本身填充；如果够长，取前fixed_hist_len个
        if seq_len <= fixed_hist_len:
            # 序列太短，使用所有项目作为历史（会导致该用户无训练样本，但保持格式一致）
            hist_list = item_seq[:]
        else:
            # 序列足够长，取前fixed_hist_len个作为历史
            hist_list = item_seq[:fixed_hist_len]
        
        # 确保所有用户的lm_hist_idx长度都是fixed_hist_len
        # 为了与MOOC完全一致，我们填充到固定长度
        if len(hist_list) < fixed_hist_len:
            # 用第一个item重复填充到fixed_hist_len（保证长度一致）
            if len(hist_list) > 0:
                hist_list = hist_list + [hist_list[-1]] * (fixed_hist_len - len(hist_list))
            else:
                hist_list = [0] * fixed_hist_len  # 空序列用0填充
        
        # 确保长度完全一致（截断过长的）
        hist_list = hist_list[:fixed_hist_len]
        
        lm_hist_idx[user_str] = hist_list
    
    # 转换multilevel_memory的用户ID
    multilevel_memory_mapped = {}
    for user_id, memory in multilevel_memory.items():
        new_user_id = str(user2id[user_id])
        multilevel_memory_mapped[new_user_id] = memory
    
    # 使用与MOOC相同的格式
    train_test_split = {
        'train': train_users,
        'test': test_users,
        'lm_hist_idx': lm_hist_idx
    }
    
    print(f"训练集用户: {len(train_users)}, 测试集用户: {len(test_users)}")
    print(f"lm_hist_idx样本: {list(lm_hist_idx.items())[:3]}")
    
    return train_test_split, multilevel_memory_mapped

def print_sample_data(sequence_data, multilevel_memory, item2attributes, itemid2title, id2item, course_domain_map, item2id):
    """打印样本数据用于检查"""
    print("\n" + "="*60)
    print("数据样本检查:")
    print("="*60)
    
    # 用户样本
    sample_user = list(sequence_data.keys())[0]
    sample_data = sequence_data[sample_user]
    print(f"\n用户样本 {sample_user}:")
    print(f"  课程序列: {sample_data[0][:10]}... (共{len(sample_data[0])}个)")
    print(f"  评分序列: {sample_data[1][:10]}... (共{len(sample_data[1])}个)")
    
    # 多级记忆样本
    if sample_user in multilevel_memory:
        memory = multilevel_memory[sample_user]
        print(f"\n多级记忆样本:")
        print(f"  感觉记忆(Sensory): {memory['sensory_memory'][:3]} (长度: {len(memory['sensory_memory'])})")
        print(f"  工作记忆(Working): {memory['working_memory'][:3]} (长度: {len(memory['working_memory'])})")
        print(f"  长期记忆(Long-term): {memory['longterm_memory'][:3]} (长度: {len(memory['longterm_memory'])})")
        print(f"  长期记忆领域: {memory['longterm_domains']}")
    
    # 课程样本
    sample_items = list(item2attributes.keys())[:3]
    print(f"\n课程样本:")
    for item_id in sample_items:
        title = itemid2title.get(item_id, "Unknown")
        attrs = item2attributes.get(item_id, [])
        
        # 获取原始course_id以查找领域
        orig_course_id = id2item.get(int(item_id), None)
        domains = course_domain_map.get(orig_course_id, ['Unknown']) if orig_course_id else ['Unknown']
        
        print(f"ID:{item_id},标题:{title[:30]}...,属性:{attrs[:5]},领域:{domains[:3]}")

def save_processed_data(output_dir, sequence_data, item2attributes, itemid2title, 
                       id2user, id2item, train_test_split, multilevel_memory):
    """保存处理后的数据"""
    print("\n正在保存处理后的数据...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存主要数据文件
    with open(os.path.join(output_dir, 'sequential_data.json'), 'w', encoding='utf-8') as f:
        json.dump(sequence_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'item2attributes.json'), 'w', encoding='utf-8') as f:
        json.dump(item2attributes, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'datamaps.json'), 'w', encoding='utf-8') as f:
        datamaps = {
            'id2user': id2user,
            'id2item': id2item,
            'itemid2title': itemid2title
        }
        json.dump(datamaps, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'train_test_split.json'), 'w', encoding='utf-8') as f:
        json.dump(train_test_split, f, ensure_ascii=False, indent=2)
    
    # 保存多级记忆数据
    with open(os.path.join(output_dir, 'multilevel_memory.json'), 'w', encoding='utf-8') as f:
        json.dump(multilevel_memory, f, ensure_ascii=False, indent=2)
    
    # 计算并保存统计信息（注意：sequential_data格式为[[items], [ratings]]）
    user_lens = [len(v[0]) for v in sequence_data.values()]  # v[0]是item序列
    item_lens = defaultdict(int)
    for data in sequence_data.values():
        for item in data[0]:  # data[0]是item序列
            item_lens[item] += 1
    
    attr_lens = [len(v) for v in item2attributes.values()]
    label_dist = defaultdict(int)
    for data in sequence_data.values():
        for label in data[1]:  # data[1]是rating/label序列
            label_dist[label] += 1
    
    # 计算属性数量：需要是最大属性ID + 1（确保嵌入层足够大）
    all_attr_ids = [a for attrs in item2attributes.values() for a in attrs]
    max_attr_id = max(all_attr_ids) if all_attr_ids else 0
    
    stats = {
        'rerank_list_len': 10,  # 重排序列表长度
        'attribute_ft_num': 1,  # 属性特征数量
        'rating_threshold': 0,  # 评分阈值（coursera使用0/1标签）
        'item_num': len(item2attributes),  # 课程总数
        'attribute_num': max_attr_id + 1,  # 属性总数（最大ID + 1）
        'rating_num': 2,  # 评分类别数（0和1）
        'dense_dim': 0,  # 密集向量维度（无增强时为0）
        'num_users': len(sequence_data),
        'num_items': len(item2attributes),
        'num_attributes': max_attr_id + 1,  # 与attribute_num保持一致
        'num_interactions': sum(user_lens),
        'avg_user_interactions': np.mean(user_lens),
        'min_user_interactions': min(user_lens),
        'max_user_interactions': max(user_lens),
        'avg_item_interactions': np.mean(list(item_lens.values())),
        'min_item_interactions': min(item_lens.values()),
        'max_item_interactions': max(item_lens.values()),
        'avg_attributes': np.mean(attr_lens),
        'label_distribution': dict(label_dist),
        'pos_ratio': label_dist[1] / sum(label_dist.values())
    }
    
    with open(os.path.join(output_dir, 'stat.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n总用户数: {stats['num_users']}, 平均用户交互: {stats['avg_user_interactions']:.4f}, " +
          f"最小长度: {stats['min_user_interactions']}, 最大长度: {stats['max_user_interactions']}")
    print(f"总课程数: {stats['num_items']}, 平均课程交互: {stats['avg_item_interactions']:.4f}, " +
          f"最小交互: {stats['min_item_interactions']}, 最大交互: {stats['max_item_interactions']}")
    print(f"总交互数: {stats['num_interactions']}, " +
          f"稀疏度: {100 * (1 - stats['num_interactions'] / (stats['num_users'] * stats['num_items'])):.2f}%")
    print(f"总属性数: {stats['num_attributes']}, 平均属性数: {stats['avg_attributes']:.4f}")
    print(f"评分分布: {stats['label_distribution']}")
    print(f"正样本比例: {stats['pos_ratio']:.4f}")
    
    print(f"\n训练集用户: {len(train_test_split['train'])}, " +
          f"测试集用户: {len(train_test_split['test'])}")
    
    print("\n数据预处理完成！")
    print(f"输出文件保存在: {output_dir}")
    print("新增文件: multilevel_memory.json (包含多级记忆数据)")

def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'coursera')
    output_dir = os.path.join(data_dir, 'proc_data')
    
    # 1. 加载数据
    enrolled_df, meta_df = load_coursera_data(data_dir)
    
    # 2. 构建交互数据（使用meta_df即Coursera_2.csv来匹配课程ID）
    user_history = build_interaction_data(enrolled_df, meta_df)
    
    # 3. K-core过滤
    user_history = k_core_filter(user_history, args.k_core_user, args.k_core_item)
    
    # 4. 提取多级记忆（使用meta_df即Coursera_2.csv）
    multilevel_memory, course_domain_map = extract_multilevel_memory(user_history, meta_df, args)
    
    # 5. 添加负样本
    all_courses = set()
    for courses in user_history.values():
        all_courses.update(courses)
    sequence_data = add_negative_samples(user_history, all_courses, args.neg_ratio, args.seed)
    
    # 6. ID映射
    user2id, item2id, id2user, id2item, sequence_data_mapped = create_id_mapping(user_history)
    
    # 更新sequence_data为带负样本的版本，使用MOOC格式：[[item_seq], [rating_seq]]
    for user_id, data in sequence_data.items():
        new_user_id = str(user2id[user_id])
        sequence_data_mapped[new_user_id] = [
            [item2id[c] for c in data['courses']],  # item sequence
            data['labels']  # rating/label sequence
        ]
    
    # 7. 处理课程属性（使用meta_df即Coursera_2.csv）
    item2attributes, itemid2title, skill2id = process_course_attributes(meta_df, item2id, course_domain_map)
    
    # 8. 划分训练集和测试集
    train_test_split, multilevel_memory_mapped = split_train_test(
        sequence_data_mapped, multilevel_memory, user2id, 
        test_ratio=args.test_ratio, fixed_hist_len=5, seed=args.seed
    )
    
    # 9. 打印样本数据
    print_sample_data(sequence_data_mapped, multilevel_memory_mapped, item2attributes, 
                     itemid2title, id2item, course_domain_map, item2id)
    
    # 10. 保存数据
    save_processed_data(output_dir, sequence_data_mapped, item2attributes, itemid2title,
                       id2user, id2item, train_test_split, multilevel_memory_mapped)

if __name__ == '__main__':
    main()

