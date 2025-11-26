'''
MOOC dataset preprocessing with hierarchical memory model
Train/test split by user IDs, ratio 9:1
Rating: positive=1, negative=0, negative sampling ratio 2:1
'''

import os
import json
import random
import numpy as np
from collections import defaultdict, Counter

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)

def parse_json_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

def correct_title(title):
    return title.strip() if title else "Unknown"

lm_hist_max = 30
sensory_memory_len = 5
working_memory_len = 15
long_term_min_interactions = 5
train_ratio = 0.9
user_core = 65
item_core = 40

def load_courses_new(course_file):
    meta = {}
    for c in parse_json_lines(course_file):
        cid = c['id']
        fields = c.get('field', [])
        name = c.get('name', 'Unknown')
        about = c.get('about', '')
        
        meta[cid] = {
            'fields': fields,
            'title': correct_title(name),
            'about': about
        }
    return meta

def load_users(user_file):
    interactions = {}
    user_attrs = {}
    
    for u in parse_json_lines(user_file):
        uid = u['id']
        user_attrs[uid] = {
            'gender': u.get('gender'),
            'school': u.get('school'),
            'year_of_birth': u.get('year_of_birth')
        }
        
        raw_courses = u.get('course_order', [])
        enroll_times = u.get('enroll_time', [])
        
        course_time_pairs = []
        for i, cid in enumerate(raw_courses):
            course_id = f"C_{cid}"
            timestamp = enroll_times[i] if i < len(enroll_times) else i
            course_time_pairs.append((course_id, timestamp, 1))
        
        course_time_pairs.sort(key=lambda x: x[1])
        interactions[uid] = [(cid, rating) for cid, _, rating in course_time_pairs]
    
    return interactions, user_attrs

def extract_multilevel_memory(user_items, meta_infos):
    user_memory = {}
    
    for user, items in user_items.items():
        total_interactions = len(items)
        
        if total_interactions < sensory_memory_len:
            user_memory[user] = {
                'sensory_memory': items,
                'working_memory': [],
                'long_term_memory': [],
                'long_term_fields': []
            }
            continue
        
        sensory_memory = items[-sensory_memory_len:]
        working_memory_start = max(0, total_interactions - working_memory_len)
        working_memory_end = total_interactions - sensory_memory_len
        working_memory = items[working_memory_start:working_memory_end] if working_memory_end > working_memory_start else []
        
        field_counter = Counter()
        field_courses = defaultdict(list)
        
        for course_id, rating in items:
            if rating > 0 and course_id in meta_infos:
                course_fields = meta_infos[course_id]['fields']
                for field in course_fields:
                    field_counter[field] += 1
                    field_courses[field].append((course_id, rating))
        
        long_term_fields = []
        long_term_courses = []
        
        for field, count in field_counter.items():
            if count >= long_term_min_interactions:
                long_term_fields.append(field)
                field_course_list = field_courses[field]
                field_indices = []
                for course_id, rating in field_course_list:
                    for idx, (item_id, item_rating) in enumerate(items):
                        if item_id == course_id and item_rating == rating:
                            field_indices.append(idx)
                            break
                
                field_indices.sort()
                if len(field_indices) >= 3:
                    selected_indices = [
                        field_indices[0],
                        field_indices[len(field_indices)//2],
                        field_indices[-1]
                    ]
                else:
                    selected_indices = field_indices
                
                for idx in selected_indices:
                    if items[idx] not in long_term_courses:
                        long_term_courses.append(items[idx])
        
        if not long_term_courses and len(items) >= lm_hist_max:
            step = len(items) // 10
            for i in range(0, len(items) - working_memory_len, step):
                if len(long_term_courses) < 15:
                    long_term_courses.append(items[i])
        
        user_memory[user] = {
            'sensory_memory': sensory_memory,
            'working_memory': working_memory,
            'long_term_memory': long_term_courses[:20],
            'long_term_fields': long_term_fields
        }
    
    return user_memory

def check_Kcore(user_items, u_core, i_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:
                user_count[user] += 1
                item_count[item_id] += 1
    
    user_ok = all(count >= u_core for count in user_count.values())
    item_ok = all(count >= i_core for count in item_count.values())
    
    return user_count, item_count, user_ok and item_ok

def filter_Kcore(user_items, u_core, i_core):
    user_count, item_count, is_kcore = check_Kcore(user_items, u_core, i_core)
    
    while not is_kcore:
        users_to_remove = []
        for user in user_items:
            pos_count = sum(1 for _, rating in user_items[user] if rating > 0)
            if pos_count < u_core:
                users_to_remove.append(user)
        
        for user in users_to_remove:
            user_items.pop(user)
        
        for user in user_items:
            filtered_items = []
            for item_id, rating in user_items[user]:
                if rating > 0 and item_count[item_id] >= i_core:
                    filtered_items.append((item_id, rating))
                elif rating <= 0:
                    continue
            user_items[user] = filtered_items
        
        user_count, item_count, is_kcore = check_Kcore(user_items, u_core, i_core)
    
    return user_items

def add_negative_samples(user_items, core_items_set):
    result = {}
    
    for user, items in user_items.items():
        positive_items = [item_id for item_id, rating in items if rating > 0]
        positive_set = set(positive_items)
        
        num_negatives = len(positive_items) // 2
        available_negatives = list(core_items_set - positive_set)
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)
        
        negative_items = random.sample(available_negatives, num_negatives)
        all_items = [(item_id, 1) for item_id in positive_items] + \
                   [(item_id, 0) for item_id in negative_items]
        
        random.shuffle(all_items)
        result[user] = all_items
    
    return result

def get_interaction_stats(user_items):
    user_counts = []
    item_counts = defaultdict(int)
    rating_counts = defaultdict(int)
    
    for user, items in user_items.items():
        user_counts.append(len(items))
        for item_id, rating in items:
            item_counts[item_id] += 1
            rating_counts[rating] += 1
    
    return user_counts, item_counts, rating_counts

def id_map(user_items, user_attrs, user_memory):
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user2attribute = {}
    
    final_data = {}
    lm_hist_idx = {}
    memory_data = {}
    
    user_id = 1
    item_id = 1
    
    user_list = list(user_items.keys())
    random.shuffle(user_list)
    
    for user in user_list:
        items = user_items[user]
        
        user2id[user] = user_id
        id2user[user_id] = user
        user2attribute[user_id] = user_attrs.get(user, {})
        
        user_item_ids = []
        user_ratings = []
        
        for item, rating in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            
            user_item_ids.append(item2id[item])
            user_ratings.append(rating)
        
        final_data[user_id] = [user_item_ids, user_ratings]
        
        if len(user_item_ids) > lm_hist_max:
            lm_hist_idx[user_id] = user_item_ids[-lm_hist_max:]
        else:
            lm_hist_idx[user_id] = user_item_ids
        
        if user in user_memory:
            memory_info = user_memory[user]
            
            sensory_memory_ids = []
            sensory_memory_ratings = []
            for item, rating in memory_info['sensory_memory']:
                if item in item2id:
                    sensory_memory_ids.append(item2id[item])
                    sensory_memory_ratings.append(rating)
            
            working_memory_ids = []
            working_memory_ratings = []
            for item, rating in memory_info['working_memory']:
                if item in item2id:
                    working_memory_ids.append(item2id[item])
                    working_memory_ratings.append(rating)
            
            long_term_memory_ids = []
            long_term_memory_ratings = []
            for item, rating in memory_info['long_term_memory']:
                if item in item2id:
                    long_term_memory_ids.append(item2id[item])
                    long_term_memory_ratings.append(rating)
            
            memory_data[user_id] = {
                'sensory_memory': [sensory_memory_ids, sensory_memory_ratings],
                'working_memory': [working_memory_ids, working_memory_ratings],
                'long_term_memory': [long_term_memory_ids, long_term_memory_ratings],
                'long_term_fields': memory_info['long_term_fields']
            }
        
        user_id += 1
    
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'user2attribute': user2attribute,
        'lm_hist_idx': lm_hist_idx
    }
    
    return final_data, len(user2id), len(item2id), data_maps, memory_data

def get_attribute_mooc(meta_infos, data_maps):
    attributes = defaultdict(int)
    
    for course_id, info in meta_infos.items():
        for field in info['fields']:
            attributes[field] += 1
    
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    
    items2attributes = {}
    attribute_lens = []
    itemid2title = {}
    
    for item_id, original_course_id in data_maps['id2item'].items():
        if original_course_id in meta_infos:
            itemid2title[item_id] = meta_infos[original_course_id]['title']
        else:
            itemid2title[item_id] = "Unknown"
    
    for course_id, info in meta_infos.items():
        if course_id in data_maps['item2id']:
            item_id = data_maps['item2id'][course_id]
            items2attributes[item_id] = []
            
            for field in info['fields']:
                if field not in attribute2id:
                    attribute2id[field] = attribute_id
                    id2attribute[attribute_id] = field
                    attribute_id += 1
                
                attributeid2num[attribute2id[field]] += 1
                items2attributes[item_id].append(attribute2id[field])
            
            if not items2attributes[item_id]:
                if 'Unknown' not in attribute2id:
                    attribute2id['Unknown'] = attribute_id
                    id2attribute[attribute_id] = 'Unknown'
                    attribute_id += 1
                items2attributes[item_id].append(attribute2id['Unknown'])
            
            attribute_lens.append(len(items2attributes[item_id]))
    
    for item_id in data_maps['id2item'].keys():
        if item_id not in items2attributes:
            items2attributes[item_id] = []
            if 'Unknown' not in attribute2id:
                attribute2id['Unknown'] = attribute_id
                id2attribute[attribute_id] = 'Unknown'
                attribute_id += 1
            items2attributes[item_id].append(attribute2id['Unknown'])
            attribute_lens.append(1)
    
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute
    data_maps['attributeid2num'] = attributeid2num
    data_maps['itemid2title'] = itemid2title
    data_maps['attribute_ft_num'] = 1
    
    return len(attribute2id), np.mean(attribute_lens), data_maps, items2attributes

def preprocess(course_file, user_file, processed_dir):
    set_seed(1234)
    
    meta_infos = load_courses_new(course_file)
    user_items, user_attrs = load_users(user_file)
    
    if user_core > 0 or item_core > 0:
        user_items = filter_Kcore(user_items, user_core, item_core)
    
    user_memory = extract_multilevel_memory(user_items, meta_infos)
    
    core_items_set = set()
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:
                core_items_set.add(item_id)
    
    user_items = add_negative_samples(user_items, core_items_set)
    final_data, user_num, item_num, data_maps, memory_data = id_map(user_items, user_attrs, user_memory)
    
    user_counts, item_counts, rating_counts = get_interaction_stats(user_items)
    
    user_avg = np.mean(user_counts)
    item_count_list = list(item_counts.values())
    item_avg = np.mean(item_count_list)
    interact_num = sum(user_counts)
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_mooc(meta_infos, data_maps)
    
    print(f"Users: {user_num}, Items: {item_num}, Interactions: {interact_num}, Sparsity: {sparsity:.2f}%")
    
    user_set = list(final_data.keys())
    random.shuffle(user_set)
    train_size = int(len(user_set) * train_ratio)
    
    train_test_split = {
        'train': user_set[:train_size],
        'test': user_set[train_size:],
        'lm_hist_idx': data_maps['lm_hist_idx']
    }
    
    os.makedirs(processed_dir, exist_ok=True)
    
    save_json(final_data, os.path.join(processed_dir, 'sequential_data.json'))
    save_json(item2attributes, os.path.join(processed_dir, 'item2attributes.json'))
    save_json(data_maps, os.path.join(processed_dir, 'datamaps.json'))
    save_json(train_test_split, os.path.join(processed_dir, 'train_test_split.json'))
    save_json(memory_data, os.path.join(processed_dir, 'multilevel_memory.json'))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(base_dir, 'data')
    DATA_SET_NAME = 'mooc'
    
    COURSE_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'course.json')
    USER_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'user.json')
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    
    preprocess(COURSE_FILE, USER_FILE, PROCESSED_DIR)
