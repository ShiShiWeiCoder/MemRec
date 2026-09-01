'''
多级记忆增强的MOOC数据预处理
split every user's interactions chronologically, train: test = 9:1
感觉记忆(Sensory Memory): 最近3-5次交互记录 (用户即时需求或最近浏览的课程)
工作记忆(Working Memory): 最近10-15次交互记录 (用户当前会话中的行为模式、正在进行的课程、短期学习目标)
长期记忆(Long-Term Memory): 基于field领域信息的职业发展方向 (长时间积累的兴趣偏好)
attribute: course field
rating: positive=1, negative=0, negative sampling ratio 2:1 (pos:neg)
'''

import os
import json
import random
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

try:
    from .memory_partition import (
        MemoryPartitionConfig,
        align_filtered_timestamps,
        build_temporal_train_test_split,
        partition_memory_at_cutoff,
    )
except ImportError:
    from memory_partition import (
        MemoryPartitionConfig,
        align_filtered_timestamps,
        build_temporal_train_test_split,
        partition_memory_at_cutoff,
    )

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

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

def add_comma(num):
    return f"{num:,}"

def correct_title(title):
    return title.strip() if title else "未知课程"


def normalize_timestamp(value, fallback_index):
    """Convert raw timestamp to numeric seconds for span calculation."""
    if value is None:
        return float(fallback_index)

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if abs(timestamp) >= 1e15:
            return timestamp / 1e6
        if abs(timestamp) >= 1e12:
            return timestamp / 1e3
        return timestamp

    if isinstance(value, str):
        ts = value.strip()
        if not ts:
            return float(fallback_index)

        # Case 1: numeric timestamp string.
        try:
            timestamp = float(ts)
            if abs(timestamp) >= 1e15:
                return timestamp / 1e6
            if abs(timestamp) >= 1e12:
                return timestamp / 1e3
            return timestamp
        except ValueError:
            pass

        # Case 2: ISO-like datetime string.
        iso_ts = ts.replace('Z', '+00:00')
        try:
            return datetime.fromisoformat(iso_ts).timestamp()
        except ValueError:
            pass

        # Case 3: common datetime formats in raw logs.
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue

    return float(fallback_index)

# 配置参数 - 多级记忆参数（默认值，可通过命令行覆盖）
lm_hist_max = 30
sensory_memory_len = 5  # 感觉记忆长度：最近的即时需求
working_memory_len = 15  # 工作记忆长度：当前会话行为模式
long_term_min_interactions = 7  # 长期记忆最少交互数：职业发展方向
long_term_min_timespan = 30  # 长期记忆最小时间跨度（天）
time_window_days = 30
sensory_tightening = 4
train_ratio = 0.9
user_core = 65
item_core = 40

def load_courses_new(course_file):
    """加载course_new.json课程元数据"""
    meta = {}
    for c in parse_json_lines(course_file):
        cid = c['id']  # course_new.json中的id格式
        fields = c.get('field', [])
        name = c.get('name', '未知课程')
        about = c.get('about', '')

        meta[cid] = {
            'fields': fields,
            'title': correct_title(name),
            'about': about
        }
    print(f"加载了 {len(meta)} 门课程 (来自course_new.json)")
    return meta

def load_users(user_file):
    """加载用户交互数据和属性，同时返回时间戳映射"""
    interactions = {}
    user_attrs = {}
    user_timestamps = {}

    for u in parse_json_lines(user_file):
        uid = u['id']
        user_attrs[uid] = {
            'gender': u.get('gender'),
            'school': u.get('school'),
            'year_of_birth': u.get('year_of_birth')
        }

        # 处理课程顺序，添加时间戳信息
        raw_courses = u.get('course_order', [])
        enroll_times = u.get('enroll_time', [])

        # 创建课程-时间对，并按时间排序
        course_time_pairs = []
        for i, cid in enumerate(raw_courses):
            course_id = f"C_{cid}"  # course_order中是682129格式，需要添加C_前缀
            # 如果有时间信息就使用，否则使用索引作为时间
            raw_timestamp = enroll_times[i] if i < len(enroll_times) else i
            timestamp = normalize_timestamp(raw_timestamp, i)
            course_time_pairs.append((course_id, timestamp, 1))  # 所有交互都是正样本

        # 按时间排序
        course_time_pairs.sort(key=lambda x: x[1])

        # 转换为最终格式：[(item_id, rating)]
        interactions[uid] = [(cid, rating) for cid, _, rating in course_time_pairs]
        # Preserve one timestamp per interaction, including repeated courses.
        user_timestamps[uid] = [ts for _, ts, _ in course_time_pairs]

    lens = [len(v) for v in interactions.values()]
    print(f"加载了 {len(interactions)} 个用户，平均每用户 {np.mean(lens):.2f} 次学习记录")
    return interactions, user_attrs, user_timestamps

def extract_multilevel_memory(user_items, meta_infos, user_timestamps=None):
    """Partition full histories with the same equations used at each cutoff."""
    if user_timestamps is None:
        raise ValueError("paper-aligned memory partitioning requires timestamps")

    config = MemoryPartitionConfig(
        time_window_days=time_window_days,
        sensory_ratio=_adaptive_sensory_ratio,
        working_ratio=_adaptive_working_ratio,
        sensory_tightening=sensory_tightening,
        long_term_threshold=long_term_min_interactions,
        long_term_min_timespan_days=long_term_min_timespan,
    )
    item_to_fields = {
        str(course_id): info.get('fields', [])
        for course_id, info in meta_infos.items()
    }
    user_multilevel_memory = {}
    for user, items in user_items.items():
        timestamps = user_timestamps.get(user)
        if timestamps is None:
            raise ValueError(f"missing timestamps for user {user}")
        memory = partition_memory_at_cutoff(
            [item_id for item_id, _ in items],
            [rating for _, rating in items],
            timestamps,
            item_to_fields,
            config=config,
            rating_threshold=0,
        )
        user_multilevel_memory[user] = {
            'sensory_memory': list(zip(*memory['sensory_memory'])),
            'working_memory': list(zip(*memory['working_memory'])),
            'long_term_memory': list(zip(*memory['long_term_memory'])),
            'long_term_fields': memory['long_term_fields'],
        }

    # 统计信息
    sensory_lens = [len(data['sensory_memory']) for data in user_multilevel_memory.values()]
    working_lens = [len(data['working_memory']) for data in user_multilevel_memory.values()]
    long_term_lens = [len(data['long_term_memory']) for data in user_multilevel_memory.values()]
    field_nums = [len(data['long_term_fields']) for data in user_multilevel_memory.values()]

    print(f"多级记忆提取完成:")
    print(f"  感觉记忆平均长度: {np.mean(sensory_lens):.2f} (最近即时需求)")
    print(f"  工作记忆平均长度: {np.mean(working_lens):.2f} (当前会话行为模式)")
    print(f"  长期记忆平均长度: {np.mean(long_term_lens):.2f} (职业发展方向)")
    print(f"  长期记忆领域平均数: {np.mean(field_nums):.2f}")
    print(f"  时间窗口: Δ={time_window_days} 天, k_s={sensory_tightening}")
    print(f"  长期记忆阈值: τ={long_term_min_interactions}, T_min={long_term_min_timespan} 天")

    return user_multilevel_memory

def check_Kcore(user_items, u_core, i_core):
    """检查K-core条件"""
    user_count = defaultdict(int)
    item_count = defaultdict(int)

    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:  # 只统计正样本
                user_count[user] += 1
                item_count[item_id] += 1

    user_ok = all(count >= u_core for count in user_count.values())
    item_ok = all(count >= i_core for count in item_count.values())

    return user_count, item_count, user_ok and item_ok

def filter_Kcore(user_items, u_core, i_core):
    """K-core过滤（只基于正样本）"""
    user_count, item_count, is_kcore = check_Kcore(user_items, u_core, i_core)

    while not is_kcore:
        # 删除用户交互数不足的用户
        users_to_remove = []
        for user in user_items:
            pos_count = sum(1 for _, rating in user_items[user] if rating > 0)
            if pos_count < u_core:
                users_to_remove.append(user)

        for user in users_to_remove:
            user_items.pop(user)

        # 删除交互数不足的物品
        for user in user_items:
            filtered_items = []
            for item_id, rating in user_items[user]:
                if rating > 0 and item_count[item_id] >= i_core:
                    filtered_items.append((item_id, rating))
                elif rating <= 0:  # 保留所有负样本，稍后重新采样
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

        # 计算需要的负样本数量 (正样本:负样本 = 2:1)
        num_negatives = len(positive_items) // 2

        # 从未交互的核心课程中随机选择负样本
        available_negatives = list(core_items_set - positive_set)
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)

        negative_items = random.sample(available_negatives, num_negatives)

        # 合并正负样本
        all_items = [(item_id, 1) for item_id in positive_items] + \
                   [(item_id, 0) for item_id in negative_items]

        # 打乱顺序以模拟真实的交互序列
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

def id_map(user_items, user_attrs, user_multilevel_memory, user_timestamps):
    """ID映射，包含多级记忆信息"""
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user2attribute = {}

    final_data = {}
    lm_hist_idx = {}
    multilevel_memory_data = {}  # 多级记忆数据
    sequential_timestamps = {}

    user_id = 1
    item_id = 1

    # 随机打乱用户顺序
    user_list = list(user_items.keys())
    random.shuffle(user_list)

    for user in user_list:
        items = user_items[user]

        # 用户ID映射
        user2id[user] = user_id
        id2user[user_id] = user

        # 用户属性
        user2attribute[user_id] = user_attrs.get(user, {})

        # 物品ID映射
        user_item_ids = []
        user_ratings = []

        for item, rating in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1

            user_item_ids.append(item2id[item])
            user_ratings.append(rating)

        # 构建序列数据
        final_data[user_id] = [user_item_ids, user_ratings]
        timestamp_seq = user_timestamps.get(user, [])
        if len(timestamp_seq) != len(user_item_ids):
            raise ValueError(
                f"timestamp alignment failed for user {user}: "
                f"items={len(user_item_ids)}, timestamps={len(timestamp_seq)}"
            )
        sequential_timestamps[user_id] = timestamp_seq

        # 构建LM历史索引
        if len(user_item_ids) > lm_hist_max:
            lm_hist_idx[user_id] = user_item_ids[-lm_hist_max:]
        else:
            lm_hist_idx[user_id] = user_item_ids

        # 构建多级记忆数据
        if user in user_multilevel_memory:
            memory_data = user_multilevel_memory[user]

            # 感觉记忆ID映射
            sensory_memory_ids = []
            sensory_memory_ratings = []
            for item, rating in memory_data['sensory_memory']:
                if item in item2id:
                    sensory_memory_ids.append(item2id[item])
                    sensory_memory_ratings.append(rating)

            # 工作记忆ID映射
            working_memory_ids = []
            working_memory_ratings = []
            for item, rating in memory_data['working_memory']:
                if item in item2id:
                    working_memory_ids.append(item2id[item])
                    working_memory_ratings.append(rating)

            # 长期记忆ID映射
            long_term_memory_ids = []
            long_term_memory_ratings = []
            for item, rating in memory_data['long_term_memory']:
                if item in item2id:
                    long_term_memory_ids.append(item2id[item])
                    long_term_memory_ratings.append(rating)

            multilevel_memory_data[user_id] = {
                'sensory_memory': [sensory_memory_ids, sensory_memory_ratings],
                'working_memory': [working_memory_ids, working_memory_ratings],
                'long_term_memory': [long_term_memory_ids, long_term_memory_ratings],
                'long_term_fields': memory_data['long_term_fields']
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

    print(f'ID映射完成: {len(user2id)} 用户, {len(item2id)} 课程')
    return (
        final_data,
        len(user2id),
        len(item2id),
        data_maps,
        multilevel_memory_data,
        sequential_timestamps,
    )

def get_attribute_mooc(meta_infos, data_maps):
    """处理课程属性信息"""
    attributes = defaultdict(int)

    # 统计属性频次
    for course_id, info in meta_infos.items():
        for field in info['fields']:
            attributes[field] += 1

    print(f'属性预处理前数量: {len(attributes)}')

    # 构建属性映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1

    items2attributes = {}
    attribute_lens = []
    itemid2title = {}

    # 首先为所有id2item中的物品初始化标题
    for item_id, original_course_id in data_maps['id2item'].items():
        if original_course_id in meta_infos:
            itemid2title[item_id] = meta_infos[original_course_id]['title']
        else:
            itemid2title[item_id] = "未知课程"

    for course_id, info in meta_infos.items():
        if course_id in data_maps['item2id']:
            item_id = data_maps['item2id'][course_id]
            items2attributes[item_id] = []

            # 处理领域属性
            for field in info['fields']:
                if field not in attribute2id:
                    attribute2id[field] = attribute_id
                    id2attribute[attribute_id] = field
                    attribute_id += 1

                attributeid2num[attribute2id[field]] += 1
                items2attributes[item_id].append(attribute2id[field])

            # 如果没有领域信息，添加默认属性
            if not items2attributes[item_id]:
                if '未知领域' not in attribute2id:
                    attribute2id['未知领域'] = attribute_id
                    id2attribute[attribute_id] = '未知领域'
                    attribute_id += 1
                items2attributes[item_id].append(attribute2id['未知领域'])

            attribute_lens.append(len(items2attributes[item_id]))

    # 为没有在meta_infos中找到的物品添加默认属性
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

    # 检查是否每个id2item中的id都有对应的标题
    missing_titles = []
    for item_id in data_maps['id2item'].keys():
        if item_id not in itemid2title:
            missing_titles.append(item_id)

    if missing_titles:
        print(f'警告: 有 {len(missing_titles)} 个物品ID没有对应的标题: {missing_titles[:10]}...')
    else:
        print('确认: 所有物品ID都有对应的标题')

    # 更新数据映射
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute
    data_maps['attributeid2num'] = attributeid2num
    data_maps['itemid2title'] = itemid2title
    data_maps['attribute_ft_num'] = 1

    return len(attribute2id), np.mean(attribute_lens), data_maps, items2attributes

def preprocess(course_file, user_file, processed_dir):
    """主预处理函数"""
    set_seed(1234)

    # 加载数据
    print("正在加载课程元数据 (course_new.json)...")
    meta_infos = load_courses_new(course_file)

    print("正在加载用户数据...")
    user_items, user_attrs, user_timestamps = load_users(user_file)
    original_user_items = {user: list(items) for user, items in user_items.items()}
    original_timestamps = {user: list(values) for user, values in user_timestamps.items()}

    print(f'原始数据加载完成！')

    # K-core过滤 - 先进行过滤
    if user_core > 0 or item_core > 0:
        print(f"开始进行 {user_core}-core 用户和 {item_core}-core 课程过滤...")
        user_items = filter_Kcore(user_items, user_core, item_core)
        user_timestamps = {
            user: align_filtered_timestamps(
                original_user_items[user], original_timestamps[user], items
            )
            for user, items in user_items.items()
        }

    # 多级记忆提取 - 基于K-core过滤后的数据
    print("开始提取多级记忆：感觉记忆、工作记忆、长期记忆...")
    print("注意：多级记忆提取基于K-core过滤后的核心数据集")
    if long_term_min_timespan > 0:
        print(f"启用长期记忆时间跨度约束: timespan >= {long_term_min_timespan} 天")
    user_multilevel_memory = extract_multilevel_memory(user_items, meta_infos, user_timestamps)

    # 获取K-core过滤后的核心物品集合
    core_items_set = set()
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:  # 只包含正样本的物品
                core_items_set.add(item_id)

    # 不在基础序列中混入负样本，保持 user_items 为按时间排序的正交互序列。
    # Rank/Rerank 的负例在候选列表构造阶段固定采样，避免破坏 seq_idx 的时间含义。
    print("跳过基础序列负采样：保持用户正交互序列的时间顺序")

    # ID映射
    print("开始ID映射...")
    (
        final_data,
        user_num,
        item_num,
        data_maps,
        multilevel_memory_data,
        sequential_timestamps,
    ) = id_map(user_items, user_attrs, user_multilevel_memory, user_timestamps)

    # 获取统计信息
    user_counts, item_counts, rating_counts = get_interaction_stats(user_items)

    user_avg = np.mean(user_counts)
    user_min, user_max = np.min(user_counts), np.max(user_counts)

    item_count_list = list(item_counts.values())
    item_avg = np.mean(item_count_list)
    item_min, item_max = np.min(item_count_list), np.max(item_count_list)

    interact_num = sum(user_counts)
    sparsity = (1 - interact_num / (user_num * item_num)) * 100

    # 处理属性信息
    print("开始处理属性信息...")
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_mooc(meta_infos, data_maps)

    # 显示统计信息
    show_info = f'总用户数: {user_num}, 平均用户交互: {user_avg:.4f}, 最小长度: {user_min}, 最大长度: {user_max} ' + \
                f'总课程数: {item_num}, 平均课程交互: {item_avg:.4f}, 最小交互: {item_min}, 最大交互: {item_max} ' + \
                f'总交互数: {interact_num}, 稀疏度: {sparsity:.2f}% ' + \
                f'总属性数: {attribute_num}, 平均属性数: {avg_attribute:.4f}'

    print(show_info)
    print(f'评分分布: {dict(rating_counts)} 正样本比例: {rating_counts[1] / sum(rating_counts.values()):.4f}')

    # 每个用户内部按时间顺序划分，训练和测试共享同一批合格用户。
    train_test_split = build_temporal_train_test_split(final_data, train_ratio)
    train_test_split['lm_hist_idx'] = data_maps['lm_hist_idx']
    user_set = train_test_split['train']
    print(
        f'按用户时间顺序 9:1 划分: {len(user_set)} 个用户同时进入训练集和测试集'
    )

    # 显示样本
    if user_set:
        sample_user = user_set[0]
        sample_data = final_data[sample_user]
        print(f'用户样本: {{"1": [{sample_data[0][:3]}, {sample_data[1][:3]}]}}')

    # 显示多级记忆样本
    if user_set and sample_user in multilevel_memory_data:
        memory_sample = multilevel_memory_data[sample_user]
        print(f'多级记忆样本:')
        print(f'  感觉记忆(Sensory): {memory_sample["sensory_memory"][0][:3]} (长度: {len(memory_sample["sensory_memory"][0])})')
        print(f'  工作记忆(Working): {memory_sample["working_memory"][0][:3]} (长度: {len(memory_sample["working_memory"][0])})')
        print(f'  长期记忆(Long-term): {memory_sample["long_term_memory"][0][:3]} (长度: {len(memory_sample["long_term_memory"][0])})')
        print(f'  长期记忆领域: {memory_sample["long_term_fields"][:3]}')

    sample_items = list(data_maps['itemid2title'].items())[:3]
    print('课程样本:', end=' ')
    for i, (item_id, title) in enumerate(sample_items):
        if item_id in item2attributes:
            attrs = item2attributes[item_id]
            attr_names = [data_maps['id2attribute'][aid] for aid in attrs]
            print(f'ID:{item_id},标题:{title[:10]}...,领域:{attr_names}', end=' ' if i < 2 else '\n')

    # 保存数据文件
    print("正在保存处理后的数据...")
    os.makedirs(processed_dir, exist_ok=True)

    save_data_file = os.path.join(processed_dir, 'sequential_data.json')
    item2attributes_file = os.path.join(processed_dir, 'item2attributes.json')
    datamaps_file = os.path.join(processed_dir, 'datamaps.json')
    split_file = os.path.join(processed_dir, 'train_test_split.json')
    multilevel_memory_file = os.path.join(processed_dir, 'multilevel_memory.json')  # 多级记忆文件
    timestamp_file = os.path.join(processed_dir, 'sequential_timestamps.json')
    partition_config_file = os.path.join(processed_dir, 'memory_partition_config.json')

    save_json(final_data, save_data_file)
    save_json(item2attributes, item2attributes_file)
    save_json(data_maps, datamaps_file)
    save_json(train_test_split, split_file)
    save_json(multilevel_memory_data, multilevel_memory_file)  # 保存多级记忆数据
    save_json(sequential_timestamps, timestamp_file)
    partition_config = MemoryPartitionConfig(
        time_window_days=time_window_days,
        sensory_ratio=_adaptive_sensory_ratio,
        working_ratio=_adaptive_working_ratio,
        sensory_tightening=sensory_tightening,
        long_term_threshold=long_term_min_interactions,
        long_term_min_timespan_days=long_term_min_timespan,
    )
    save_json(partition_config.to_dict(), partition_config_file)

    print("数据预处理完成！")
    print(f"输出文件保存在: {processed_dir}")
    print("新增文件: multilevel_memory.json (包含多级记忆数据)")

adaptive_window_mode = False

_adaptive_sensory_ratio = 0.08
_adaptive_working_ratio = 0.22

def compute_adaptive_window(total_interactions):
    """根据用户总交互数动态计算感觉记忆和工作记忆窗口大小"""
    s_len = max(3, min(10, round(total_interactions * _adaptive_sensory_ratio)))
    w_len = max(5, min(25, round(total_interactions * _adaptive_working_ratio)))
    return s_len, w_len

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MOOCCubeX多级记忆预处理')
    parser.add_argument('--sensory_len', type=int, default=5, help='感觉记忆长度')
    parser.add_argument('--working_len', type=int, default=15, help='工作记忆长度')
    parser.add_argument('--long_term_threshold', type=int, default=7, help='长期记忆最少交互数 (τ)')
    parser.add_argument('--long_term_min_timespan', type=float, default=30, help='长期记忆最小时间跨度(天)')
    parser.add_argument('--time_window_days', type=float, default=30, help='感觉/工作记忆时间窗口 Δ（天）')
    parser.add_argument('--sensory_tightening', type=float, default=4, help='感觉记忆时间收紧系数 k_s，范围[3,6]')
    parser.add_argument('--adaptive_window', action='store_true', help='启用自适应记忆窗口')
    parser.add_argument('--sensory_ratio', type=float, default=0.08, help='自适应窗口感觉记忆比例 (默认8%%)')
    parser.add_argument('--working_ratio', type=float, default=0.22, help='自适应窗口工作记忆比例 (默认22%%)')
    parser.add_argument('--user_core', type=int, default=65, help='用户k-core阈值')
    parser.add_argument('--item_core', type=int, default=40, help='物品k-core阈值')
    parser.add_argument('--output_dir', type=str, default=None, help='自定义输出目录')
    cli_args = parser.parse_args()

    sensory_memory_len = cli_args.sensory_len
    working_memory_len = cli_args.working_len
    long_term_min_interactions = cli_args.long_term_threshold
    long_term_min_timespan = cli_args.long_term_min_timespan
    time_window_days = cli_args.time_window_days
    sensory_tightening = cli_args.sensory_tightening
    user_core = cli_args.user_core
    item_core = cli_args.item_core
    adaptive_window_mode = cli_args.adaptive_window
    _adaptive_sensory_ratio = cli_args.sensory_ratio
    _adaptive_working_ratio = cli_args.working_ratio

    if adaptive_window_mode:
        print(f"自适应记忆窗口: 启用 (sensory={_adaptive_sensory_ratio*100:.0f}%/3-10, working={_adaptive_working_ratio*100:.0f}%/5-25)")
    else:
        print(f"记忆划分参数: sensory={sensory_memory_len}, working={working_memory_len}")
    print(f"k-core参数: user_core={user_core}, item_core={item_core}, "
          f"τ={long_term_min_interactions}, T_min={long_term_min_timespan}天")

    DATA_DIR = 'data/'
    DATA_SET_NAME = 'MOOCCubeX'
    COURSE_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'raw_data', 'entities', 'course_new.json')
    USER_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'raw_data', 'entities', 'user.json')
    PROCESSED_DIR = cli_args.output_dir if cli_args.output_dir else os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')

    preprocess(COURSE_FILE, USER_FILE, PROCESSED_DIR)
