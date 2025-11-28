'''
Generate CTR, Rank, and Rerank data with multilevel memory prompts
Supports both Coursera and MOOC datasets
'''

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import random
from collections import defaultdict
from preprocess.utils import load_json, save_json, save_pickle


def generate_ctr_data(sequential_data, lm_hist_idx, uid_set, rating_threshold=0):
    """Generate CTR data for a set of users"""
    full_data = []
    for uid in uid_set:
        uid_str = str(uid)
        if uid_str not in sequential_data:
            continue
        hist_idx_data = lm_hist_idx.get(uid_str, [])
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        
        item_seq, rating_seq = sequential_data[uid_str]
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([int(uid) if isinstance(uid, str) else uid, idx, label])
    return full_data


def generate_rank_data(sequential_data, lm_hist_idx, uid_set, item_set=None, rank_list_len=50, rank_item_from_hist=5, rating_threshold=0, dataset_name='general'):
    """Generate Rank data for a set of users"""
    dataset = dataset_name.lower()
    
    if dataset == 'coursera':
        rank_num = 15
        min_hist_len = 5
        min_rank_num = 10
        step = 8
        full_data = []
        
        for uid in uid_set:
            uid_str = str(uid)
            if uid_str not in sequential_data:
                continue
            
            courses = sequential_data[uid_str][0]
            labels = sequential_data[uid_str][1]
            
            if uid_str in lm_hist_idx:
                hist_idx_data = lm_hist_idx[uid_str]
                start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
                start_idx = max(start_idx, min_hist_len)
            else:
                start_idx = min_hist_len
            
            for i in range(start_idx, len(courses), step):
                available_items = len(courses) - i
                if available_items < min_rank_num:
                    break
                actual_rank_num = min(rank_num, available_items)
                candidate_courses = courses[i:i+actual_rank_num]
                candidate_labels = labels[i:i+actual_rank_num]
                try:
                    user_idx = int(uid)
                except (TypeError, ValueError):
                    user_idx = uid
                full_data.append([user_idx, i, candidate_courses, candidate_labels])
        
        return full_data
    
    full_data = []
    
    for uid in uid_set:
        uid_str = str(uid)
        if uid_str not in sequential_data:
            continue
        
        hist_idx_data = lm_hist_idx.get(uid_str, [])
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        
        item_seq, rating_seq = sequential_data[uid_str]
        idx = start_idx
        seq_len = len(item_seq)
        
        while idx < seq_len:
            end_idx = min(idx + rank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rank_list_len - len(chosen_iid)
            
            if item_set:
                neg_sample = random.sample(item_set, neg_sample_num)
            else:
                available_items = [i for i in item_seq if i not in chosen_iid]
                neg_sample = random.sample(available_items, min(neg_sample_num, len(available_items)))
                if len(neg_sample) < neg_sample_num:
                    neg_sample += [0] * (neg_sample_num - len(neg_sample))
            
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in chosen_rating] + [0] * neg_sample_num
            
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([int(uid) if isinstance(uid, str) else uid, idx, candidates, candidate_lbs])
            idx = end_idx
    
    return full_data


def generate_rerank_data(sequential_data, lm_hist_idx, uid_set, item_set=None, rerank_list_len=10, rerank_item_from_hist=4, rating_threshold=0, dataset_name='general'):
    """Generate Rerank data for a set of users"""
    dataset = dataset_name.lower()
    
    if dataset == 'coursera':
        rerank_num = 10
        min_hist_len = 5
        min_rerank_num = 5
        step = 4
        full_data = []
        
        for uid in uid_set:
            uid_str = str(uid)
            if uid_str not in sequential_data:
                continue
            
            courses = sequential_data[uid_str][0]
            labels = sequential_data[uid_str][1]
            
            if uid_str in lm_hist_idx:
                hist_idx_data = lm_hist_idx[uid_str]
                start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
                start_idx = max(start_idx, min_hist_len)
            else:
                start_idx = min_hist_len
            
            for i in range(start_idx, len(courses), step):
                available_items = len(courses) - i
                if available_items < min_rerank_num:
                    break
                actual_rerank_num = min(rerank_num, available_items)
                candidate_courses = courses[i:i+actual_rerank_num]
                candidate_labels = labels[i:i+actual_rerank_num]
                try:
                    user_idx = int(uid)
                except (TypeError, ValueError):
                    user_idx = uid
                full_data.append([user_idx, i, candidate_courses, candidate_labels])
        
        return full_data
    
    full_data = []
    
    for uid in uid_set:
        uid_str = str(uid)
        if uid_str not in sequential_data:
            continue
        
        hist_idx_data = lm_hist_idx.get(uid_str, [])
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        
        item_seq, rating_seq = sequential_data[uid_str]
        idx = start_idx
        seq_len = len(item_seq)
        
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            
            if item_set:
                neg_sample = random.sample(item_set, neg_sample_num)
            else:
                available_items = [i for i in item_seq if i not in chosen_iid]
                neg_sample = random.sample(available_items, min(neg_sample_num, len(available_items)))
                if len(neg_sample) < neg_sample_num:
                    neg_sample += [0] * (neg_sample_num - len(neg_sample))
            
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in chosen_rating] + [0] * neg_sample_num
            
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([int(uid) if isinstance(uid, str) else uid, idx, candidates, candidate_lbs])
            idx = end_idx
    
    return full_data


def generate_item_prompt_multilevel_memory(item2attributes, itemid2title, id2item, multilevel_memory, dataset_name='coursera'):
    """Generate item prompts with multilevel memory context"""
    item_prompts = {}
    
    for item_id, attrs in item2attributes.items():
        title = itemid2title.get(item_id, f"Course {item_id}")
        
        if attrs and len(attrs) > 0:
            if dataset_name == 'mooc':
                from preprocess.utils import load_json
                datamap_path = os.path.join('data', dataset_name, 'proc_data', 'datamaps.json')
                datamap = load_json(datamap_path)
                attrid2name = datamap.get('id2attribute', {})
                main_field = attrid2name.get(str(attrs[0]), 'Unknown Domain')
                prompt = f"Introduce course {title} in the {main_field} domain and describe "
            else:
                prompt = f"Introduce course {title} in the related domain and describe "
        else:
            prompt = f"Introduce course {title} and describe "
        
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
    
    return item_prompts


def generate_hist_prompt_multilevel_memory(sequential_data, multilevel_memory, id2user, itemid2title, item2attributes, id2item, dataset_name='coursera'):
    """Generate user history prompts with multilevel memory context"""
    hist_prompts = {}
    item2id = {v: k for k, v in id2item.items()}
    
    for user_id in sequential_data.keys():
        if user_id not in multilevel_memory:
            continue
        
        memory = multilevel_memory[user_id]
        
        def get_course_title(original_cid):
            original_cid_str = str(original_cid)
            mapped_id = item2id.get(original_cid_str)
            if mapped_id:
                return itemid2title.get(mapped_id, f"Course {original_cid}")
            return f"Course {original_cid}"
        
        sensory_titles = [f'"{get_course_title(c)}"' for c in memory.get('sensory_memory', [])[:5]]
        working_titles = [f'"{get_course_title(c)}"' for c in memory.get('working_memory', [])[:10]]
        longterm_titles = [f'"{get_course_title(c)}"' for c in memory.get('longterm_memory', [])[:10]]
        
        if dataset_name == 'mooc':
            datamap_path = os.path.join('data', dataset_name, 'proc_data', 'datamaps.json')
            datamap = load_json(datamap_path)
            user2attribute = datamap.get('user2attribute', {})
            user_attrs = user2attribute.get(user_id, {})
            user_info_parts = []
            if user_attrs.get('gender'):
                gender_value = user_attrs['gender']
                gender_text = 'male' if gender_value == 1 else 'female' if gender_value == 2 else str(gender_value)
                user_info_parts.append(gender_text)
            if user_attrs.get('school'):
                user_info_parts.append(f'from {user_attrs["school"]}')
            
            if user_info_parts:
                user_description = ', '.join(user_info_parts)
                prompt = f"Given a user who is {user_description}, "
            else:
                prompt = "Given a user, "
        else:
            prompt = "Given a user, "
        
        prompt += "this user's course selections are organized by the Atkinson-Shiffrin Memory Model into three levels: "
        
        if sensory_titles:
            prompt += f"SENSORY MEMORY (immediate exploration needs): {', '.join(sensory_titles)}; "
        
        if working_titles:
            prompt += f"WORKING MEMORY (current learning session and short-term skill goals): {', '.join(working_titles)}; "
        
        if longterm_titles:
            prompt += f"LONG-TERM MEMORY (strategic career planning): {', '.join(longterm_titles)}. "
        
        prompt += (
            "Analyze this user's learning preferences considering factors such as subject domain, instructional approach, "
            "complexity level, pacing and duration, depth versus breadth, assessment methods, and real-world applications. "
            "Provide clear explanations based on the multilevel memory patterns. "
            "Your response must be in English without subtitles, bullet points, or numbered lists."
        )
        
        if dataset_name == 'mooc':
            prompt += " Translate any Chinese course names to English in your analysis."
        
        hist_prompts[user_id] = prompt
    
    return hist_prompts


def generate_multilevel_memory_analysis_prompt(multilevel_memory, itemid2title, id2item, dataset_name='coursera'):
    """Generate multilevel memory analysis prompts"""
    analysis_prompts = {}
    item2id = {v: k for k, v in id2item.items()}
    
    for user_id, memory in multilevel_memory.items():
        def get_course_title(original_cid):
            original_cid_str = str(original_cid)
            mapped_id = item2id.get(original_cid_str)
            if mapped_id:
                return itemid2title.get(mapped_id, f"Course {original_cid}")
            return f"Course {original_cid}"
        
        sensory_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('sensory_memory', [])[:6]])
        working_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('working_memory', [])[:8]])
        longterm_titles = ', '.join([f'"{get_course_title(c)}"' for c in memory.get('longterm_memory', [])[:8]])
        
        if dataset_name == 'mooc':
            datamap_path = os.path.join('data', dataset_name, 'proc_data', 'datamaps.json')
            datamap = load_json(datamap_path)
            user2attribute = datamap.get('user2attribute', {})
            user_attrs = user2attribute.get(user_id, {})
            user_info_parts = []
            if user_attrs.get('gender'):
                gender_value = user_attrs['gender']
                gender_text = 'male' if gender_value == 1 else 'female' if gender_value == 2 else str(gender_value)
                user_info_parts.append(gender_text)
            if user_attrs.get('school'):
                user_info_parts.append(f'from {user_attrs["school"]}')
            
            if user_info_parts:
                user_description = ', '.join(user_info_parts)
                prompt = f"Given a user who is {user_description}, "
            else:
                prompt = "Given a user, "
        else:
            prompt = "Given a user, "
        
        prompt += "this user's learning behaviors are categorized by the Atkinson-Shiffrin Memory Model: "
        
        if sensory_titles:
            prompt += f"SENSORY MEMORY (immediate browsing): {sensory_titles}; "
        
        if working_titles:
            prompt += f"WORKING MEMORY (current learning session): {working_titles}; "
        
        if longterm_titles:
            prompt += f"LONG-TERM MEMORY (strategic interests): {longterm_titles}. "
        
        prompt += (
            "Compare and contrast the three memory levels to reveal the cognitive processing hierarchy. "
            "Explain how SENSORY MEMORY courses differ from WORKING MEMORY courses in terms of exploration versus consolidation. "
            "Analyze how WORKING MEMORY courses transition into LONG-TERM MEMORY for career planning. "
            "Identify patterns in memory consolidation and learning progression across the three levels. "
            "Describe the interactions and dependencies between different memory levels in shaping learning trajectories. "
            "Your response must be in English without subtitles, bullet points, or numbered lists."
        )
        
        if dataset_name == 'mooc':
            prompt += " Translate any Chinese course names to English in your analysis."
        
        analysis_prompts[user_id] = prompt
    
    return analysis_prompts


def save_data(output_dir, train_ctr, test_ctr, train_rank, test_rank, train_rerank, test_rerank, 
              item_prompts, hist_prompts, analysis_prompts):
    """Save generated data and prompts"""
    with open(os.path.join(output_dir, 'ctr.train'), 'wb') as f:
        pickle.dump(train_ctr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'ctr.test'), 'wb') as f:
        pickle.dump(test_ctr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rank.train'), 'wb') as f:
        pickle.dump(train_rank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rank.test'), 'wb') as f:
        pickle.dump(test_rank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rerank.train'), 'wb') as f:
        pickle.dump(train_rerank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'rerank.test'), 'wb') as f:
        pickle.dump(test_rerank, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'prompt.item.multilevel_memory'), 'w', encoding='utf-8') as f:
        json.dump(item_prompts, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'prompt.hist.multilevel_memory'), 'w', encoding='utf-8') as f:
        json.dump(hist_prompts, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'prompt.memory_analysis'), 'w', encoding='utf-8') as f:
        json.dump(analysis_prompts, f, ensure_ascii=False, indent=2)


def main_coursera():
    """Main function for Coursera dataset"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'coursera', 'proc_data')
    
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
    
    train_ctr = generate_ctr_data(sequential_data, lm_hist_idx, train_users)
    test_ctr = generate_ctr_data(sequential_data, lm_hist_idx, test_users)
    train_rank = generate_rank_data(sequential_data, lm_hist_idx, train_users, dataset_name='coursera')
    test_rank = generate_rank_data(sequential_data, lm_hist_idx, test_users, dataset_name='coursera')
    train_rerank = generate_rerank_data(sequential_data, lm_hist_idx, train_users, dataset_name='coursera')
    test_rerank = generate_rerank_data(sequential_data, lm_hist_idx, test_users, dataset_name='coursera')
    
    item_prompts = generate_item_prompt_multilevel_memory(item2attributes, itemid2title, id2item, multilevel_memory, 'coursera')
    hist_prompts = generate_hist_prompt_multilevel_memory(sequential_data, multilevel_memory, id2user, itemid2title, item2attributes, id2item, 'coursera')
    analysis_prompts = generate_multilevel_memory_analysis_prompt(multilevel_memory, itemid2title, id2item, 'coursera')
    
    save_data(data_dir, train_ctr, test_ctr, train_rank, test_rank, train_rerank, test_rerank,
              item_prompts, hist_prompts, analysis_prompts)


def main_mooc():
    """Main function for MOOC dataset"""
    random.seed(12345)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'mooc', 'proc_data')
    
    sequence_data = load_json(os.path.join(data_dir, 'sequential_data.json'))
    train_test_split = load_json(os.path.join(data_dir, 'train_test_split.json'))
    item2attribute = load_json(os.path.join(data_dir, 'item2attributes.json'))
    multilevel_memory_data = load_json(os.path.join(data_dir, 'multilevel_memory.json'))
    datamap = load_json(os.path.join(data_dir, 'datamaps.json'))
    
    item_set = list(map(int, item2attribute.keys()))
    itemid2title = datamap['itemid2title']
    id2item = datamap['id2item']
    id2user = datamap['id2user']
    
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['train'])
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['test'])
    save_pickle(train_ctr, os.path.join(data_dir, 'ctr.train'))
    save_pickle(test_ctr, os.path.join(data_dir, 'ctr.test'))
    
    train_rank = generate_rank_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['train'], item_set, dataset_name='mooc')
    test_rank = generate_rank_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['test'], item_set, dataset_name='mooc')
    save_pickle(train_rank, os.path.join(data_dir, 'rank.train'))
    save_pickle(test_rank, os.path.join(data_dir, 'rank.test'))
    
    train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['train'], item_set, dataset_name='mooc')
    test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'], train_test_split['test'], item_set, dataset_name='mooc')
    save_pickle(train_rerank, os.path.join(data_dir, 'rerank.train'))
    save_pickle(test_rerank, os.path.join(data_dir, 'rerank.test'))
    
    statis = {
        'rerank_list_len': 10,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': 0,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 2,
        'dense_dim': 0,
    }
    save_json(statis, os.path.join(data_dir, 'stat.json'))
    
    item_prompt = generate_item_prompt_multilevel_memory(item2attribute, itemid2title, id2item, multilevel_memory_data, 'mooc')
    hist_prompt = generate_hist_prompt_multilevel_memory(sequence_data, multilevel_memory_data, id2user, itemid2title, item2attribute, id2item, 'mooc')
    memory_analysis_prompt = generate_multilevel_memory_analysis_prompt(multilevel_memory_data, itemid2title, id2item, 'mooc')
    
    save_json(item_prompt, os.path.join(data_dir, 'prompt.item.multilevel_memory'), ensure_ascii=False)
    save_json(hist_prompt, os.path.join(data_dir, 'prompt.hist.multilevel_memory'), ensure_ascii=False)
    save_json(memory_analysis_prompt, os.path.join(data_dir, 'prompt.memory_analysis'), ensure_ascii=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate multilevel memory prompts')
    parser.add_argument('--dataset', type=str, default='coursera', choices=['coursera', 'mooc'],
                       help='Dataset name: coursera or mooc')
    args = parser.parse_args()
    
    if args.dataset == 'mooc':
        main_mooc()
    else:
        main_coursera()

