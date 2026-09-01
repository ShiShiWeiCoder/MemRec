'''
多级记忆增强的 MOOCCube 数据预处理
split every user's interactions chronologically, train: test = 9:1
感觉记忆(Sensory Memory): 最近3-5次交互记录
工作记忆(Working Memory): 最近10-15次交互记录
长期记忆(Long-Term Memory): 基于 field 领域信息的长期兴趣
attribute: course field (由 course-concept + concept-field 推导)
rating: positive=1, negative=0, negative sampling ratio 2:1 (pos:neg)
'''

import os
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

from preprocess_mooccube import (
    set_seed,
    load_courses,
    build_course_fields_map,
    filter_Kcore,
    add_negative_samples,
    get_interaction_stats,
    get_attribute_mooc,
    save_json,
    parse_json_lines,
    parse_timestamp,
)

# 配置参数（可由 CLI 覆盖）
lm_hist_max = 30
long_term_min_interactions = 7
long_term_min_timespan = 30
time_window_days = 30
sensory_tightening = 4
train_ratio = 0.9
user_core = 9
item_core = 10

# 自适应窗口参数（CLI 可覆盖）
adaptive_window_mode = False    # CLI: --adaptive_window
_adaptive_sensory_ratio = 0.08
_adaptive_working_ratio = 0.22
max_sensory_memory_len = 5
max_working_memory_len = 15


def load_users_with_timestamps(user_file):
    """加载用户数据，保留时间戳供 timespan 计算。"""
    interactions = {}
    user_timestamps = {}
    user_attrs = {}
    for u in parse_json_lines(user_file):
        uid = u['id']
        user_attrs[uid] = {'name': u.get('name')}
        raw_courses = u.get('course_order', [])
        enroll_times = u.get('enroll_time', [])
        course_time_pairs = []
        timestamp_seq = []
        for i, cid in enumerate(raw_courses):
            if not cid:
                continue
            ts = parse_timestamp(enroll_times[i] if i < len(enroll_times) else None, i)
            course_time_pairs.append((cid, ts, i, 1))
            timestamp_seq.append(ts)
        course_time_pairs.sort(key=lambda x: (x[1], x[2]))
        interactions[uid] = [(cid, rating) for cid, _, _, rating in course_time_pairs]
        user_timestamps[uid] = [record[1] for record in course_time_pairs]
    lens = [len(v) for v in interactions.values()] if interactions else [0]
    print(f"加载了 {len(interactions)} 个用户，平均每用户 {np.mean(lens):.2f} 次学习记录")
    return interactions, user_attrs, user_timestamps


def compute_adaptive_window(total_len):
    s = max(1, round(total_len * _adaptive_sensory_ratio))
    w = max(0, round(total_len * _adaptive_working_ratio))
    return s, w

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
    sensory_lens = [len(data['sensory_memory']) for data in user_multilevel_memory.values()]
    working_lens = [len(data['working_memory']) for data in user_multilevel_memory.values()]
    long_term_lens = [len(data['long_term_memory']) for data in user_multilevel_memory.values()]
    field_nums = [len(data['long_term_fields']) for data in user_multilevel_memory.values()]
    print(f"多级记忆提取完成:")
    print("  模式: 论文公式的比例与时间双约束窗口")
    print(f"  感觉记忆平均长度: {np.mean(sensory_lens):.2f}")
    print(f"  工作记忆平均长度: {np.mean(working_lens):.2f}")
    print(f"  长期记忆平均长度: {np.mean(long_term_lens):.2f}")
    print(f"  长期记忆领域平均数: {np.mean(field_nums):.2f}")
    if long_term_min_timespan > 0:
        print(f"  时间跨度约束: >= {long_term_min_timespan} 天")
    return user_multilevel_memory

def id_map(user_items, user_attrs, user_multilevel_memory, user_timestamps):
    """ID映射，包含多级记忆信息"""
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user2attribute = {}
    final_data = {}
    lm_hist_idx = {}
    multilevel_memory_data = {}
    sequential_timestamps = {}
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
        timestamp_seq = user_timestamps.get(user, [])
        if len(timestamp_seq) != len(user_item_ids):
            raise ValueError(
                f"timestamp alignment failed for user {user}: "
                f"items={len(user_item_ids)}, timestamps={len(timestamp_seq)}"
            )
        sequential_timestamps[user_id] = timestamp_seq
        if len(user_item_ids) > lm_hist_max:
            lm_hist_idx[user_id] = user_item_ids[-lm_hist_max:]
        else:
            lm_hist_idx[user_id] = user_item_ids
        if user in user_multilevel_memory:
            memory_data = user_multilevel_memory[user]
            sensory_memory_ids = []
            sensory_memory_ratings = []
            for item, rating in memory_data['sensory_memory']:
                if item in item2id:
                    sensory_memory_ids.append(item2id[item])
                    sensory_memory_ratings.append(rating)
            working_memory_ids = []
            working_memory_ratings = []
            for item, rating in memory_data['working_memory']:
                if item in item2id:
                    working_memory_ids.append(item2id[item])
                    working_memory_ratings.append(rating)
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

def preprocess(course_file, user_file, concept_file, concept_field_file, course_concept_file, processed_dir):
    """主预处理函数"""
    set_seed(1234)
    print("正在加载课程领域映射...")
    course_fields = build_course_fields_map(concept_file, concept_field_file, course_concept_file)
    print("正在加载课程元数据...")
    meta_infos = load_courses(course_file, course_fields)
    print("正在加载用户数据...")
    user_items, user_attrs, user_timestamps = load_users_with_timestamps(user_file)
    original_user_items = {user: list(items) for user, items in user_items.items()}
    original_timestamps = {user: list(values) for user, values in user_timestamps.items()}
    print('原始数据加载完成！')
    if user_core > 0 or item_core > 0:
        print(f"开始进行 {user_core}-core 用户和 {item_core}-core 课程过滤...")
        user_items = filter_Kcore(user_items, user_core, item_core)
        user_timestamps = {
            user: align_filtered_timestamps(
                original_user_items[user], original_timestamps[user], items
            )
            for user, items in user_items.items()
        }
    print("开始提取多级记忆：感觉记忆、工作记忆、长期记忆...")
    user_multilevel_memory = extract_multilevel_memory(user_items, meta_infos, user_timestamps)
    core_items_set = set()
    for user, items in user_items.items():
        for item_id, rating in items:
            if rating > 0:
                core_items_set.add(item_id)
    print("跳过基础序列负采样：保持用户正交互序列的时间顺序")
    print("开始ID映射...")
    final_data, user_num, item_num, data_maps, multilevel_memory_data, sequential_timestamps = id_map(
        user_items, user_attrs, user_multilevel_memory, user_timestamps
    )
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
    train_test_split = build_temporal_train_test_split(final_data, train_ratio)
    train_test_split['lm_hist_idx'] = data_maps['lm_hist_idx']
    user_set = train_test_split['train']
    print(
        f'按用户时间顺序 9:1 划分: {len(user_set)} 个用户同时进入训练集和测试集'
    )
    if user_set:
        sample_user = user_set[0]
        sample_data = final_data[sample_user]
        print(f'用户样本: {{"1": [{sample_data[0][:3]}, {sample_data[1][:3]}]}}')
        if sample_user in multilevel_memory_data:
            memory_sample = multilevel_memory_data[sample_user]
            print('多级记忆样本:')
            print(f'  感觉记忆: {memory_sample["sensory_memory"][0][:3]} (长度: {len(memory_sample["sensory_memory"][0])})')
            print(f'  工作记忆: {memory_sample["working_memory"][0][:3]} (长度: {len(memory_sample["working_memory"][0])})')
            print(f'  长期记忆: {memory_sample["long_term_memory"][0][:3]} (长度: {len(memory_sample["long_term_memory"][0])})')
            print(f'  长期记忆领域: {memory_sample["long_term_fields"][:3]}')
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
    multilevel_memory_file = os.path.join(processed_dir, 'multilevel_memory.json')
    timestamp_file = os.path.join(processed_dir, 'sequential_timestamps.json')
    partition_config_file = os.path.join(processed_dir, 'memory_partition_config.json')
    save_json(final_data, save_data_file)
    save_json(item2attributes, item2attributes_file)
    save_json(data_maps, datamaps_file)
    save_json(train_test_split, split_file)
    save_json(multilevel_memory_data, multilevel_memory_file)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MOOCCube多级记忆预处理')
    parser.add_argument('--long_term_min_timespan', type=float, default=30,
                        help='长期记忆最小时间跨度(天)')
    parser.add_argument('--time_window_days', type=float, default=30,
                        help='感觉/工作记忆时间窗口 Δ（天）')
    parser.add_argument('--sensory_tightening', type=float, default=4,
                        help='感觉记忆时间收紧系数 k_s，范围[3,6]')
    parser.add_argument('--adaptive_window', action='store_true',
                        help='启用自适应记忆窗口（按比例计算）')
    parser.add_argument('--sensory_ratio', type=float, default=0.08,
                        help='自适应窗口感觉记忆比例 (默认8%%)')
    parser.add_argument('--working_ratio', type=float, default=0.22,
                        help='自适应窗口工作记忆比例 (默认22%%)')
    parser.add_argument('--user_core', type=int, default=9, help='用户k-core阈值')
    parser.add_argument('--item_core', type=int, default=10, help='课程k-core阈值')
    parser.add_argument('--long_term_threshold', type=int, default=7,
                        help='长期记忆最少交互数 (τ)')
    parser.add_argument('--output_dir', type=str, default=None, help='自定义输出目录')
    cli_args = parser.parse_args()

    # 更新全局参数
    long_term_min_timespan = cli_args.long_term_min_timespan
    time_window_days = cli_args.time_window_days
    sensory_tightening = cli_args.sensory_tightening
    adaptive_window_mode = cli_args.adaptive_window
    _adaptive_sensory_ratio = cli_args.sensory_ratio
    _adaptive_working_ratio = cli_args.working_ratio
    user_core = cli_args.user_core
    item_core = cli_args.item_core
    long_term_min_interactions = cli_args.long_term_threshold

    DATA_DIR = 'data/'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'MOOCCube')
    DATA_SET_NAME = 'MOOCCube'
    COURSE_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'course.json')
    USER_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'user.json')
    CONCEPT_FILE = os.path.join(RAW_DATA_DIR, 'entities', 'concept.json')
    CONCEPT_FIELD_FILE = os.path.join(RAW_DATA_DIR, 'relations', 'concept-field.json')
    COURSE_CONCEPT_FILE = os.path.join(RAW_DATA_DIR, 'relations', 'course-concept.json')
    PROCESSED_DIR = cli_args.output_dir or os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')

    print(f"配置: adaptive={'是' if adaptive_window_mode else '否'} "
          f"sensory_ratio={_adaptive_sensory_ratio} working_ratio={_adaptive_working_ratio} "
          f"timespan>={long_term_min_timespan}天 τ={long_term_min_interactions}")
    print(f"输出目录: {PROCESSED_DIR}")

    preprocess(COURSE_FILE, USER_FILE, CONCEPT_FILE, CONCEPT_FIELD_FILE, COURSE_CONCEPT_FILE, PROCESSED_DIR)
