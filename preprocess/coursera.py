'''
Coursera dataset preprocessing with hierarchical memory model
'''

import os
import json
import random
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Coursera dataset')
    parser.add_argument('--k_core_user', type=int, default=3, help='k-core threshold for users')
    parser.add_argument('--k_core_item', type=int, default=3, help='k-core threshold for items')
    parser.add_argument('--sensory_memory_len', type=int, default=1, help='Length of sensory memory window')
    parser.add_argument('--working_memory_len', type=int, default=2, help='Length of working memory window')
    parser.add_argument('--longterm_memory_domains', type=int, default=3, help='Number of domains tracked for long-term memory')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of users placed in the test split')
    parser.add_argument('--neg_ratio', type=int, default=1, help='Negative sampling ratio (pos:neg = (neg_ratio+1):1)')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility')
    return parser.parse_args()

def load_coursera_data(data_dir):
    enrolled_file = os.path.join(data_dir, 'enrolled_course.csv')
    try:
        enrolled_df = pd.read_csv(enrolled_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            enrolled_df = pd.read_csv(enrolled_file, encoding='latin1')
        except:
            enrolled_df = pd.read_csv(enrolled_file, encoding='ISO-8859-1')
    
    meta_file = os.path.join(data_dir, 'Coursera_2.csv')
    try:
        meta_df = pd.read_csv(meta_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            meta_df = pd.read_csv(meta_file, encoding='latin1')
        except:
            meta_df = pd.read_csv(meta_file, encoding='ISO-8859-1')
    
    return enrolled_df, meta_df

def build_interaction_data(enrolled_df, meta_df):
    user_history = {}
    valid_course_ids = set(meta_df['id'].values)
    
    for _, row in enrolled_df.iterrows():
        user_id = row['User_id']
        course_ids_str = str(row['History_course_id']).split(', ')
        
        valid_courses = []
        for cid_str in course_ids_str:
            try:
                cid_int = int(cid_str)
                if cid_int in valid_course_ids:
                    valid_courses.append(str(cid_int))
            except ValueError:
                continue
        
        if valid_courses:
            user_history[user_id] = valid_courses
    
    return user_history

def k_core_filter(user_history, k_core_user=20, k_core_item=10):
    while True:
        item_count = defaultdict(int)
        for courses in user_history.values():
            for course in courses:
                item_count[course] += 1
        
        user_history_new = {}
        for user, courses in user_history.items():
            new_courses = [c for c in courses if item_count[c] >= k_core_item]
            if len(new_courses) >= k_core_user:
                user_history_new[user] = new_courses
        
        if len(user_history_new) == len(user_history):
            break
        user_history = user_history_new
    
    return user_history

def extract_multilevel_memory(user_history, meta_df, args):
    multilevel_memory = {}
    course_domain_map = {}
    
    for _, row in meta_df.iterrows():
        course_id_int = int(row['id'])
        course_id_str = str(course_id_int)
        skills = str(row['Skills']) if pd.notna(row['Skills']) else ''
        if skills and skills != 'nan':
            skill_list = [s.strip() for s in skills.split(',') if s.strip()]
            domains = skill_list[:3] if skill_list else ['Unknown']
        else:
            domains = ['Unknown']
        course_domain_map[course_id_str] = domains
    
    for user_id, courses in user_history.items():
        total_len = len(courses)
        
        sensory_memory = courses[-args.sensory_memory_len:] if total_len >= args.sensory_memory_len else courses
        
        if total_len > args.sensory_memory_len + args.working_memory_len:
            working_memory = courses[-(args.sensory_memory_len + args.working_memory_len):-args.sensory_memory_len]
        elif total_len > args.sensory_memory_len:
            working_memory = courses[:-args.sensory_memory_len]
        else:
            working_memory = []
        
        longterm_memory = courses
        
        domain_count = defaultdict(int)
        for course in longterm_memory:
            if course in course_domain_map:
                for domain in course_domain_map[course]:
                    domain_count[domain] += 1
        
        top_domains = sorted(domain_count.items(), key=lambda x: x[1], reverse=True)
        longterm_domains = [d[0] for d in top_domains[:args.longterm_memory_domains]]
        
        multilevel_memory[user_id] = {
            'sensory_memory': sensory_memory,
            'working_memory': working_memory,
            'longterm_memory': longterm_memory,
            'longterm_domains': longterm_domains
        }
    
    return multilevel_memory, course_domain_map

def add_negative_samples(user_history, all_courses, neg_ratio=1, seed=2023):
    random.seed(seed)
    
    sequence_data = {}
    for user_id, pos_courses in user_history.items():
        pos_set = set(pos_courses)
        all_items = [(course, 1) for course in pos_courses]
        
        # 使用排序后的列表消除 set 带来的随机遍历顺序，保证在相同 seed 下完全可复现
        neg_candidates = sorted(all_courses - pos_set)
        num_neg = len(pos_courses) // (neg_ratio + 1)
        neg_samples = random.sample(neg_candidates, min(num_neg, len(neg_candidates)))
        all_items.extend([(course, 0) for course in neg_samples])
        
        random.shuffle(all_items)
        
        user_courses = [item[0] for item in all_items]
        user_labels = [item[1] for item in all_items]
        
        sequence_data[user_id] = {
            'courses': user_courses,
            'labels': user_labels
        }
    
    return sequence_data

def create_id_mapping(user_history):
    user_ids = sorted(list(user_history.keys()))
    id2user = {i: uid for i, uid in enumerate(user_ids)}
    user2id = {uid: i for i, uid in id2user.items()}
    
    all_courses = set()
    for courses in user_history.values():
        all_courses.update(courses)
    course_ids = sorted(list(all_courses))
    id2item = {i: cid for i, cid in enumerate(course_ids)}
    item2id = {cid: i for i, cid in id2item.items()}
    
    sequence_data_mapped = {}
    for user_id, data in user_history.items():
        new_user_id = user2id[user_id]
        sequence_data_mapped[str(new_user_id)] = [
            [item2id[c] for c in data],
            [1] * len(data)
        ]
    
    return user2id, item2id, id2user, id2item, sequence_data_mapped

def process_course_attributes(meta_df, item2id, course_domain_map):
    item2attributes = {}
    itemid2title = {}
    
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
    
    skill2id = {skill: i for i, skill in enumerate(sorted(used_skills))}
    
    for course_id_str, internal_idx in item2id.items():
        course_id_int = int(course_id_str)
        course_info = meta_df[meta_df['id'] == course_id_int]
        
        if len(course_info) > 0:
            row = course_info.iloc[0]
            course_name = str(row['Course Name']) if pd.notna(row['Course Name']) else f"Course {course_id_str}"
            skills = str(row['Skills']) if pd.notna(row['Skills']) else ''
            
            skill_ids = []
            if skills and skills != 'nan':
                for skill in skills.split(','):
                    skill = skill.strip()
                    if skill and skill in skill2id:
                        skill_ids.append(skill2id[skill])
            
            if not skill_ids:
                skill_ids = [0]
            
            item2attributes[str(internal_idx)] = skill_ids
            itemid2title[str(internal_idx)] = course_name
        else:
            item2attributes[str(internal_idx)] = [0]
            itemid2title[str(internal_idx)] = f"Unknown Course {course_id_str}"
    
    return item2attributes, itemid2title, skill2id

def split_train_test(sequence_data, multilevel_memory, user2id, test_ratio=0.1, fixed_hist_len=3, seed=2023):
    random.seed(seed)
    
    all_users = list(sequence_data.keys())
    random.shuffle(all_users)
    
    num_test = int(len(all_users) * test_ratio)
    test_users = [int(u) for u in all_users[:num_test]]
    train_users = [int(u) for u in all_users[num_test:]]
    
    lm_hist_idx = {}
    for user_str in sequence_data.keys():
        user_id = int(user_str)
        item_seq = sequence_data[user_str][0]
        seq_len = len(item_seq)
        
        if seq_len <= fixed_hist_len:
            hist_list = item_seq[:]
        else:
            hist_list = item_seq[:fixed_hist_len]
        
        if len(hist_list) < fixed_hist_len:
            if len(hist_list) > 0:
                hist_list = hist_list + [hist_list[-1]] * (fixed_hist_len - len(hist_list))
            else:
                hist_list = [0] * fixed_hist_len
        
        hist_list = hist_list[:fixed_hist_len]
        lm_hist_idx[user_str] = hist_list
    
    multilevel_memory_mapped = {}
    for user_id, memory in multilevel_memory.items():
        new_user_id = str(user2id[user_id])
        multilevel_memory_mapped[new_user_id] = memory
    
    train_test_split = {
        'train': train_users,
        'test': test_users,
        'lm_hist_idx': lm_hist_idx
    }
    
    return train_test_split, multilevel_memory_mapped

def save_processed_data(output_dir, sequence_data, item2attributes, itemid2title, 
                       id2user, id2item, train_test_split, multilevel_memory):
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    with open(os.path.join(output_dir, 'multilevel_memory.json'), 'w', encoding='utf-8') as f:
        json.dump(multilevel_memory, f, ensure_ascii=False, indent=2)
    
    user_lens = [len(v[0]) for v in sequence_data.values()]
    item_lens = defaultdict(int)
    for data in sequence_data.values():
        for item in data[0]:
            item_lens[item] += 1
    
    attr_lens = [len(v) for v in item2attributes.values()]
    label_dist = defaultdict(int)
    for data in sequence_data.values():
        for label in data[1]:
            label_dist[label] += 1
    
    all_attr_ids = [a for attrs in item2attributes.values() for a in attrs]
    max_attr_id = max(all_attr_ids) if all_attr_ids else 0
    
    stats = {
        'rerank_list_len': 10,
        'attribute_ft_num': 1,
        'rating_threshold': 0,
        'item_num': len(item2attributes),
        'attribute_num': max_attr_id + 1,
        'rating_num': 2,
        'dense_dim': 0,
        'num_users': len(sequence_data),
        'num_items': len(item2attributes),
        'num_attributes': max_attr_id + 1,
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
    
    print(f"Users: {stats['num_users']}, Items: {stats['num_items']}, Interactions: {stats['num_interactions']}, Sparsity: {100 * (1 - stats['num_interactions'] / (stats['num_users'] * stats['num_items'])):.2f}%")

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'coursera')
    output_dir = os.path.join(data_dir, 'proc_data')
    
    enrolled_df, meta_df = load_coursera_data(data_dir)
    user_history = build_interaction_data(enrolled_df, meta_df)
    user_history = k_core_filter(user_history, args.k_core_user, args.k_core_item)
    
    multilevel_memory, course_domain_map = extract_multilevel_memory(user_history, meta_df, args)
    
    all_courses = set()
    for courses in user_history.values():
        all_courses.update(courses)
    sequence_data = add_negative_samples(user_history, all_courses, args.neg_ratio, args.seed)
    
    user2id, item2id, id2user, id2item, sequence_data_mapped = create_id_mapping(user_history)
    
    for user_id, data in sequence_data.items():
        new_user_id = str(user2id[user_id])
        sequence_data_mapped[new_user_id] = [
            [item2id[c] for c in data['courses']],
            data['labels']
        ]
    
    item2attributes, itemid2title, skill2id = process_course_attributes(meta_df, item2id, course_domain_map)
    
    train_test_split, multilevel_memory_mapped = split_train_test(
        sequence_data_mapped, multilevel_memory, user2id, 
        test_ratio=args.test_ratio, fixed_hist_len=5, seed=args.seed
    )
    
    save_processed_data(output_dir, sequence_data_mapped, item2attributes, itemid2title,
                       id2user, id2item, train_test_split, multilevel_memory_mapped)

if __name__ == '__main__':
    main()

