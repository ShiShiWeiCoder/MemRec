'''
MOOCCube 预处理（基于 MOOCCube 实体/关系文件）
split train/test by user IDs, train: test= 9: 1
RS history: recent interactions 10 items (pos & neg), ID & attributes & rating
LM history: one lm history for each user (max_len=30, item ID, attributes, rating)
attribute: course field (由 course-concept + concept-field 推导)
rating: positive=1, negative=0, negative sampling ratio 2:1 (pos:neg)
'''

import os
import json
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

# 工具函数
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)

def parse_json_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def parse_relation_pairs(path, delimiter='\t'):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) >= 2:
                yield parts[0], parts[1]

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

def add_comma(num):
    return f"{num:,}"

def correct_title(title):
    return title.strip() if title else "未知课程"

def parse_timestamp(value, fallback):
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if abs(timestamp) >= 1e15:
            return timestamp / 1e6
        if abs(timestamp) >= 1e12:
            return timestamp / 1e3
        return timestamp
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return float(fallback)
        try:
            timestamp = float(value)
            if abs(timestamp) >= 1e15:
                return timestamp / 1e6
            if abs(timestamp) >= 1e12:
                return timestamp / 1e3
            return timestamp
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
    return float(fallback)

# 配置参数
lm_hist_max = 30
train_ratio = 0.9
# 调整K-core阈值：控制用户数在1w左右，课程数600左右
user_core = 9
item_core = 10

def load_concept_name_map(concept_file):
    """加载概念名称映射（含领域概念）"""
    name_map = {}
    for c in parse_json_lines(concept_file):
        cid = c.get('id')
        if not cid:
            continue
        name = c.get('name') or c.get('en') or cid
        name_map[cid] = name
    print(f"加载了 {len(name_map)} 个概念名称")
    return name_map

def load_concept_field_map(concept_field_file):
    """加载 concept -> field 概念映射"""
    concept_to_fields = defaultdict(set)
    for concept_id, field_id in parse_relation_pairs(concept_field_file):
        if concept_id and field_id:
            concept_to_fields[concept_id].add(field_id)
    print(f"加载了 {len(concept_to_fields)} 个概念的领域映射")
    return concept_to_fields

def build_course_fields(course_concept_file, concept_to_fields):
    """构建 course -> field 概念ID映射"""
    course_to_fields = defaultdict(set)
    missing_field_concepts = 0
    for course_id, concept_id in parse_relation_pairs(course_concept_file):
        field_ids = concept_to_fields.get(concept_id)
        if not field_ids:
            missing_field_concepts += 1
            continue
        for field_id in field_ids:
            course_to_fields[course_id].add(field_id)
    print(f"课程-领域映射构建完成: {len(course_to_fields)} 门课程, 未匹配领域概念 {missing_field_concepts} 条")
    return course_to_fields

def build_course_fields_map(concept_file, concept_field_file, course_concept_file):
    """构建 course -> field name 列表"""
    concept_name_map = load_concept_name_map(concept_file)
    concept_to_fields = load_concept_field_map(concept_field_file)
    course_to_field_ids = build_course_fields(course_concept_file, concept_to_fields)
    course_to_fields = {}
    for course_id, field_ids in course_to_field_ids.items():
        fields = [concept_name_map.get(fid, fid) for fid in field_ids if fid]
        # 去重并稳定排序，便于复现
        course_to_fields[course_id] = sorted(set(fields))
    return course_to_fields

def load_courses(course_file, course_fields=None):
    """加载课程元数据（MOOCCube 无字段信息，使用关系推导）"""
    meta = {}
    course_fields = course_fields or {}
    for c in parse_json_lines(course_file):
        cid = c['id']
        raw_fields = c.get('field', [])
        if not isinstance(raw_fields, list):
            raw_fields = [raw_fields] if raw_fields else []
        derived_fields = course_fields.get(cid, [])
        fields = list(dict.fromkeys(raw_fields + derived_fields))
        name = c.get('name', '未知课程')
        about = c.get('about', '')
        meta[cid] = {
            'fields': fields,
            'title': correct_title(name),
            'about': about
        }
    print(f"加载了 {len(meta)} 门课程")
    return meta

def load_users(user_file):
    """加载用户交互数据和属性（MOOCCube）"""
    interactions = {}
    user_attrs = {}
    for u in parse_json_lines(user_file):
        uid = u['id']
        user_attrs[uid] = {
            'name': u.get('name')
        }
        raw_courses = u.get('course_order', [])
        enroll_times = u.get('enroll_time', [])
        course_time_pairs = []
        for i, cid in enumerate(raw_courses):
            if not cid:
                continue
            timestamp = parse_timestamp(enroll_times[i] if i < len(enroll_times) else None, i)
            course_time_pairs.append((cid, timestamp, i, 1))  # 所有交互都是正样本
        # 按时间排序，时间相同用原序号稳定排序
        course_time_pairs.sort(key=lambda x: (x[1], x[2]))
        interactions[uid] = [(cid, rating) for cid, _, _, rating in course_time_pairs]
    lens = [len(v) for v in interactions.values()] if interactions else [0]
    print(f"加载了 {len(interactions)} 个用户，平均每用户 {np.mean(lens):.2f} 次学习记录")
    return interactions, user_attrs

def check_Kcore(user_items, u_core, i_core):
    """检查K-core条件"""
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:
                user_count[user] += 1
                item_count[item_id] += 1
    user_ok = all(count >= u_core for count in user_count.values()) if user_count else True
    item_ok = all(count >= i_core for count in item_count.values()) if item_count else True
    return user_count, item_count, user_ok and item_ok

def filter_Kcore(user_items, u_core, i_core):
    """K-core过滤（只基于正样本）"""
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
    print(f"K-core过滤后: {len(user_items)} 用户, {len(item_count)} 课程")
    return user_items

def add_negative_samples(user_items, core_items_set):
    """添加负样本，比例为正2:负1，只从K-core过滤后的核心物品中采样"""
    result = {}
    for user, items in user_items.items():
        positive_items = [item_id for item_id, rating in items if rating > 0]
        positive_set = set(positive_items)
        num_negatives = len(positive_items) // 2
        available_negatives = list(core_items_set - positive_set)
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)
        negative_items = random.sample(available_negatives, num_negatives) if num_negatives > 0 else []
        all_items = [(item_id, 1) for item_id in positive_items] + \
                   [(item_id, 0) for item_id in negative_items]
        random.shuffle(all_items)
        result[user] = all_items
    print(f"添加负样本完成，正:负 = 2:1")
    return result

def get_interaction_stats(user_items):
    """获取交互统计信息"""
    user_counts = []
    item_counts = defaultdict(int)
    rating_counts = defaultdict(int)
    for user, items in user_items.items():
        user_counts.append(len(items))
        for item_id, rating in items:
            item_counts[item_id] += 1
            rating_counts[rating] += 1
    return user_counts, item_counts, rating_counts

def id_map(user_items, user_attrs):
    """ID映射"""
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user2attribute = {}
    final_data = {}
    lm_hist_idx = {}
    user_id = 1
    item_id = 1
    user_list = list(user_items.keys())
    random.shuffle(user_list)
    for user in user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user2attribute[user_id] = user_attrs.get(user, {})
            user_id += 1
        item_ids = []
        ratings = []
        for item, rating in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            item_ids.append(item2id[item])
            ratings.append(rating)
        uid = user2id[user]
        lm_hist_idx[uid] = min(len(item_ids), lm_hist_max)
        final_data[uid] = [item_ids, ratings]
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'user2attribute': user2attribute,
        'lm_hist_idx': lm_hist_idx
    }
    return final_data, user_id - 1, item_id - 1, data_maps

def get_attribute_mooc(meta_infos, data_maps):
    """处理课程属性信息"""
    attributes = defaultdict(int)
    for course_id, info in meta_infos.items():
        for field in info['fields']:
            attributes[field] += 1
    print(f'属性预处理前数量: {len(attributes)}')
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
            itemid2title[item_id] = "未知课程"
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
                if '未知领域' not in attribute2id:
                    attribute2id['未知领域'] = attribute_id
                    id2attribute[attribute_id] = '未知领域'
                    attribute_id += 1
                items2attributes[item_id].append(attribute2id['未知领域'])
            attribute_lens.append(len(items2attributes[item_id]))
    for item_id in data_maps['id2item'].keys():
        if item_id not in items2attributes:
            items2attributes[item_id] = []
            if '未知领域' not in attribute2id:
                attribute2id['未知领域'] = attribute_id
                id2attribute[attribute_id] = '未知领域'
                attribute_id += 1
            items2attributes[item_id].append(attribute2id['未知领域'])
            attribute_lens.append(1)
    print(f'属性处理后数量: {len(attribute2id)}')
    print(f'属性长度统计, 最小: {np.min(attribute_lens)}, 最大: {np.max(attribute_lens)}, 平均: {np.mean(attribute_lens):.4f}')
    print(f'itemid2title中存储了 {len(itemid2title)} 个物品的标题')
    print(f'id2item中有 {len(data_maps["id2item"])} 个物品ID')
    missing_titles = []
    for item_id in data_maps['id2item'].keys():
        if item_id not in itemid2title:
            missing_titles.append(item_id)
    if missing_titles:
        print(f'警告: 有 {len(missing_titles)} 个物品ID没有对应的标题: {missing_titles[:10]}...')
    else:
        print('确认: 所有物品ID都有对应的标题')
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute
    data_maps['attributeid2num'] = attributeid2num
    data_maps['itemid2title'] = itemid2title
    data_maps['attribute_ft_num'] = 1
    return len(attribute2id), np.mean(attribute_lens), data_maps, items2attributes

def preprocess(course_file, user_file, concept_file, concept_field_file, course_concept_file, processed_dir):
    """主预处理函数"""
    set_seed(1234)
    print("正在加载课程领域映射...")
    course_fields = build_course_fields_map(concept_file, concept_field_file, course_concept_file)
    print("正在加载课程元数据...")
    meta_infos = load_courses(course_file, course_fields)
    print("正在加载用户数据...")
    user_items, user_attrs = load_users(user_file)
    print('原始数据加载完成！')
    if user_core > 0 or item_core > 0:
        print(f"开始进行 {user_core}-core 用户和 {item_core}-core 课程过滤...")
        user_items = filter_Kcore(user_items, user_core, item_core)
    core_items_set = set()
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:
                core_items_set.add(item_id)
    print("开始添加负样本...")
    user_items = add_negative_samples(user_items, core_items_set)
    print("开始ID映射...")
    final_data, user_num, item_num, data_maps = id_map(user_items, user_attrs)
    user_counts, item_counts, rating_counts = get_interaction_stats(user_items)
    user_avg = np.mean(user_counts)
    user_min, user_max = np.min(user_counts), np.max(user_counts)
    item_count_list = list(item_counts.values())
    item_avg = np.mean(item_count_list) if item_count_list else 0
    item_min = np.min(item_count_list) if item_count_list else 0
    item_max = np.max(item_count_list) if item_count_list else 0
    interact_num = sum(user_counts) if user_counts else 0
    total_pairs = user_num * item_num
    sparsity = (1 - interact_num / total_pairs) * 100 if total_pairs > 0 else 0.0
    print("开始处理属性信息...")
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_mooc(meta_infos, data_maps)
    show_info = f'总用户数: {user_num}, 平均用户交互: {user_avg:.4f}, 最小长度: {user_min}, 最大长度: {user_max} ' + \
                f'总课程数: {item_num}, 平均课程交互: {item_avg:.4f}, 最小交互: {item_min}, 最大交互: {item_max} ' + \
                f'总交互数: {interact_num}, 稀疏度: {sparsity:.2f}% ' + \
                f'总属性数: {attribute_num}, 平均属性数: {avg_attribute:.4f}'
    print(show_info)
    if rating_counts:
        print(f'评分分布: {dict(rating_counts)} 正样本比例: {rating_counts[1] / sum(rating_counts.values()):.4f}')
    user_set = list(final_data.keys())
    random.shuffle(user_set)
    train_size = int(len(user_set) * train_ratio)
    train_test_split = {
        'train': user_set[:train_size],
        'test': user_set[train_size:],
        'lm_hist_idx': data_maps['lm_hist_idx']
    }
    print(f'训练集用户: {len(train_test_split["train"])}, 测试集用户: {len(train_test_split["test"])}')
    if user_set:
        sample_user = user_set[0]
        sample_data = final_data[sample_user]
        print(f'用户样本: {{"1": [{sample_data[0][:3]}, {sample_data[1][:3]}]}}')
    sample_items = list(data_maps['itemid2title'].items())[:3]
    print('课程样本:', end=' ')
    for i, (item_id, title) in enumerate(sample_items):
        if item_id in item2attributes:
            attrs = item2attributes[item_id]
            attr_names = [data_maps['id2attribute'][aid] for aid in attrs]
            print(f'ID:{item_id},标题:{title[:10]}...,领域:{attr_names}', end=' ' if i < 2 else '\n')
    print("正在保存处理后的数据...")
    os.makedirs(processed_dir, exist_ok=True)
    save_data_file = os.path.join(processed_dir, 'sequential_data.json')
    item2attributes_file = os.path.join(processed_dir, 'item2attributes.json')
    datamaps_file = os.path.join(processed_dir, 'datamaps.json')
    split_file = os.path.join(processed_dir, 'train_test_split.json')
    save_json(final_data, save_data_file)
    save_json(item2attributes, item2attributes_file)
    save_json(data_maps, datamaps_file)
    save_json(train_test_split, split_file)
    print("数据预处理完成！")
    print(f"输出文件保存在: {processed_dir}")

if __name__ == '__main__':
    DATA_DIR = 'data/'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'MOOCCube')
    DATA_SET_NAME = 'mooccube'
    COURSE_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'course.json')
    USER_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'user.json')
    CONCEPT_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'concept.json')
    CONCEPT_FIELD_FILE = os.path.join(RAW_DATA_DIR, 'relations', 'concept-field.json')
    COURSE_CONCEPT_FILE = os.path.join(RAW_DATA_DIR, 'relations', 'course-concept.json')
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    preprocess(COURSE_FILE, USER_FILE, CONCEPT_FILE, CONCEPT_FIELD_FILE, COURSE_CONCEPT_FILE, PROCESSED_DIR)
