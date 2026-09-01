import os
import json
import random
import pickle
from collections import defaultdict
from pre_utils import load_json, save_json, save_pickle
from memory_partition import (
    MemoryPartitionConfig,
    partition_memory_at_cutoff,
    temporal_sample_bounds,
)
from memory_features import compute_memory_features

# Parameters
rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10

# Threshold for implicit positive label
rating_threshold = 0


def sample_negative_items(item_set, positive_set, exclude_set, sample_num):
    candidates = list(set(item_set) - set(positive_set) - set(exclude_set))
    if len(candidates) < sample_num:
        raise ValueError(
            f"Not enough negative candidates: need={sample_num}, available={len(candidates)}"
        )
    return random.sample(candidates, sample_num)

def generate_ctr_data(sequence_data, uid_set, temporal_cutoffs, split):
    full_data = []
    total_label = []
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        start_idx, end_idx = temporal_sample_bounds(
            len(item_seq), temporal_cutoffs[str(uid)], split
        )
        positive_set = {iid for iid, rating in zip(item_seq, rating_seq) if rating > rating_threshold}
        for idx in range(start_idx, end_idx):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label) if total_label else 0)
    print(full_data[:5])
    return full_data


def generate_rank_data(sequence_data, uid_set, item_set, temporal_cutoffs, split):
    """
    生成Rank阶段（粗排）的训练数据
    与Rerank的区别：候选数量更多（50个 vs 10个）
    """
    full_data = []
    rank_list_len = 50  # Rank阶段候选数量更多
    rank_item_from_hist = 5  # 从历史中选择的正样本数量

    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        start_idx, stop_idx = temporal_sample_bounds(
            len(item_seq), temporal_cutoffs[str(uid)], split
        )
        positive_set = {
            iid
            for iid, rating in zip(item_seq, rating_seq)
            if rating > rating_threshold
        }
        idx = start_idx
        while idx < stop_idx:
            end_idx = min(idx + rank_item_from_hist, stop_idx)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rank_list_len - len(chosen_iid)
            neg_sample = sample_negative_items(item_set, positive_set, chosen_iid, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('Rank data - user num', len(uid_set), 'data num', len(full_data))
    print('Rank data sample:', full_data[:2])
    return full_data


def generate_rerank_data(sequence_data, uid_set, item_set, temporal_cutoffs, split):
    """
    生成Rerank阶段（精排）的训练数据
    候选数量较少（10个），更注重精细排序
    """
    full_data = []
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        start_idx, stop_idx = temporal_sample_bounds(
            len(item_seq), temporal_cutoffs[str(uid)], split
        )
        positive_set = {iid for iid, rating in zip(item_seq, rating_seq) if rating > rating_threshold}
        idx = start_idx
        while idx < stop_idx:
            end_idx = min(idx + rerank_item_from_hist, stop_idx)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = sample_negative_items(item_set, positive_set, chosen_iid, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('Rerank data - user num', len(uid_set), 'data num', len(full_data))
    print('Rerank data sample:', full_data[:2])
    return full_data


def collect_cutoff_indices(*datasets):
    cutoffs = set()
    for data in datasets:
        for uid, seq_idx, _, _ in data:
            cutoffs.add((str(uid), int(seq_idx)))
    return sorted(cutoffs, key=lambda value: (int(value[0]), value[1]))


def build_causal_multilevel_memory(
    sequence_data,
    timestamp_data,
    item2attribute,
    datamap,
    cutoffs,
    partition_config,
):
    causal_memory = {}
    for uid, seq_idx in cutoffs:
        item_seq, rating_seq = sequence_data[uid]
        timestamp_seq = timestamp_data[uid]
        causal_memory[f"{uid}:{seq_idx}"] = partition_memory_at_cutoff(
            item_seq[:seq_idx],
            rating_seq[:seq_idx],
            timestamp_seq[:seq_idx],
            item2attribute,
            field_names=datamap.get('id2attribute', {}),
            config=partition_config,
            rating_threshold=rating_threshold,
        )
    print(f'causal multilevel memory num {len(causal_memory)}')
    return causal_memory


def generate_hist_prompt_multilevel_memory(sequence_data, item2attribute, datamap, lm_hist_idx, multilevel_memory_data, dataset_name):
    """
    生成基于Atkinson-Shiffrin记忆模型的用户多级记忆分析提示词
    只输入三个记忆层次的课程,不包含完整历史
    """
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    user2attribute = datamap['user2attribute']
    hist_prompts = {}

    print('item2attribute', list(item2attribute.items())[:10])
    print("=== DEBUG: generate_hist_prompt_multilevel_memory ===")
    print("multilevel_memory_data keys (first 5):", list(multilevel_memory_data.keys())[:5])

    for memory_key, memory_data in multilevel_memory_data.items():
        uid = str(memory_key).split(':', 1)[0]
        # 获取多级记忆数据

        # 构建感觉记忆文本(Sensory Memory)
        sensory_memory_texts = []
        if 'sensory_memory' in memory_data and memory_data['sensory_memory']:
            if isinstance(memory_data['sensory_memory'], list) and len(memory_data['sensory_memory']) > 0:
                sensory_memory_ids = memory_data['sensory_memory'][0] if isinstance(memory_data['sensory_memory'][0], list) else memory_data['sensory_memory']
                for iid in sensory_memory_ids[:5]:  # 只取前5个
                    if str(iid) in itemid2title:
                        sensory_memory_texts.append('"{}"'.format(itemid2title[str(iid)]))
                    else:
                        sensory_memory_texts.append('"Course {}"'.format(iid))

        # 构建工作记忆文本(Working Memory)
        working_memory_texts = []
        if 'working_memory' in memory_data and memory_data['working_memory']:
            if isinstance(memory_data['working_memory'], list) and len(memory_data['working_memory']) > 0:
                working_memory_ids = memory_data['working_memory'][0] if isinstance(memory_data['working_memory'][0], list) else memory_data['working_memory']
                for iid in working_memory_ids[:10]:  # 只取前10个
                    if str(iid) in itemid2title:
                        working_memory_texts.append('"{}"'.format(itemid2title[str(iid)]))
                    else:
                        working_memory_texts.append('"Course {}"'.format(iid))

        # 构建长期记忆文本(Long-Term Memory)
        long_term_memory_texts = []
        long_term_fields = []
        if 'long_term_memory' in memory_data and memory_data['long_term_memory']:
            if isinstance(memory_data['long_term_memory'], list) and len(memory_data['long_term_memory']) > 0:
                long_term_memory_ids = memory_data['long_term_memory'][0] if isinstance(memory_data['long_term_memory'][0], list) else memory_data['long_term_memory']
                for iid in long_term_memory_ids[:10]:  # 只取前10个
                    if str(iid) in itemid2title:
                        long_term_memory_texts.append('"{}"'.format(itemid2title[str(iid)]))
                    else:
                        long_term_memory_texts.append('"Course {}"'.format(iid))

        if 'long_term_fields' in memory_data:
            long_term_fields = memory_data['long_term_fields']

        if dataset_name == 'mooccube':
            # MOOCCube数据集：用户无额外属性，直接构建提示词
            prompt = "Given a user, "

            prompt += "this user's course selections are organized by the Atkinson-Shiffrin Memory Model into three levels: "

            # 添加三级记忆课程
            if sensory_memory_texts:
                prompt += "SENSORY MEMORY (immediate exploration needs): {}; ".format(', '.join(sensory_memory_texts[:5]))

            if working_memory_texts:
                prompt += "WORKING MEMORY (current learning session and short-term skill goals): {}; ".format(', '.join(working_memory_texts[:10]))

            if long_term_memory_texts:
                prompt += "LONG-TERM MEMORY (strategic career planning): {}. ".format(', '.join(long_term_memory_texts[:10]))

            # 分析要求
            prompt += (
                "Analyze this user's learning preferences considering factors such as subject domain, instructional approach, "
                "complexity level, pacing and duration, depth versus breadth, assessment methods, and real-world applications. "
                "Provide clear explanations based on the multilevel memory patterns. "
                "Your response must be in English without subtitles, bullet points, or Chinese text. "
                "Translate any Chinese course names to English in your analysis."
            )

            hist_prompts[str(memory_key)] = prompt
        else:
            raise NotImplementedError

    print('data num', len(hist_prompts))
    print("Sample prompt preview:", list(hist_prompts.values())[0][:200] + "...")
    return hist_prompts


def generate_item_prompt_multilevel_memory(item2attribute, datamap, dataset_name):
    """
    生成基于多级记忆框架的课程认知属性分析提示词
    """
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = {}

    print("=== DEBUG: generate_item_prompt_multilevel_memory ===")
    print("itemid2title keys (first 5):", list(itemid2title.keys())[:5])

    for iid, title in itemid2title.items():
        if dataset_name == 'mooccube':
            # 获取课程领域(第一个属性作为主领域)
            attrs = item2attribute.get(str(iid), [])
            if attrs:
                main_field = attrid2name.get(str(attrs[0]), 'Unknown Domain')
                prompt = "Introduce course {} in the {} domain and describe ".format(title, main_field)
            else:
                prompt = "Introduce course {} and describe ".format(title)

            prompt += (
                "its cognitive attributes from the Atkinson-Shiffrin Memory Model perspective considering "
                "SENSORY MEMORY impact (immediate appeal and first impressions), "
                "WORKING MEMORY demands (cognitive load and practical skill building), and "
                "LONG-TERM MEMORY value (career development and domain expertise). "
                "Particularly emphasize the prerequisite knowledge requirements and prerequisite course dependencies, "
                "as these are unique characteristics of courses that determine learning progression and memory consolidation pathways. "
                "Explain how prerequisites relate to different memory levels and learning readiness. "
                "Your response must be in English without subtitles, bullet points, or numbered lists."
            )

            item_prompts[iid] = prompt
        else:
            raise NotImplementedError

    print('data num', len(item_prompts))
    print("Sample item prompt preview:", list(item_prompts.values())[0][:200] + "...")
    return item_prompts


def generate_multilevel_memory_analysis_prompt(multilevel_memory_data, datamap, dataset_name):
    """
    生成多级记忆层次对比分析提示词
    用于分析不同记忆层次之间的交互作用和认知处理机制
    """
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    user2attribute = datamap['user2attribute']
    memory_analysis_prompts = {}

    print("=== DEBUG: generate_multilevel_memory_analysis_prompt ===")

    for uid, memory_data in multilevel_memory_data.items():
        if dataset_name == 'mooccube':
            # 构建感觉记忆课程列表
            sensory_memory_courses = []
            if 'sensory_memory' in memory_data and memory_data['sensory_memory']:
                if isinstance(memory_data['sensory_memory'], list) and len(memory_data['sensory_memory']) > 0:
                    sensory_ids = memory_data['sensory_memory'][0] if isinstance(memory_data['sensory_memory'][0], list) else memory_data['sensory_memory']
                    for iid in sensory_ids[:6]:  # 只取前6个
                        if str(iid) in itemid2title:
                            sensory_memory_courses.append('"{}"'.format(itemid2title[str(iid)]))
                        else:
                            sensory_memory_courses.append('"Course {}"'.format(iid))

            # 构建工作记忆课程列表
            working_memory_courses = []
            if 'working_memory' in memory_data and memory_data['working_memory']:
                if isinstance(memory_data['working_memory'], list) and len(memory_data['working_memory']) > 0:
                    working_ids = memory_data['working_memory'][0] if isinstance(memory_data['working_memory'][0], list) else memory_data['working_memory']
                    for iid in working_ids[:8]:  # 只取前8个
                        if str(iid) in itemid2title:
                            working_memory_courses.append('"{}"'.format(itemid2title[str(iid)]))
                        else:
                            working_memory_courses.append('"Course {}"'.format(iid))

            # 构建长期记忆课程列表和领域
            long_term_memory_courses = []
            if 'long_term_memory' in memory_data and memory_data['long_term_memory']:
                if isinstance(memory_data['long_term_memory'], list) and len(memory_data['long_term_memory']) > 0:
                    longterm_ids = memory_data['long_term_memory'][0] if isinstance(memory_data['long_term_memory'][0], list) else memory_data['long_term_memory']
                    for iid in longterm_ids[:8]:  # 只取前8个
                        if str(iid) in itemid2title:
                            long_term_memory_courses.append('"{}"'.format(itemid2title[str(iid)]))
                        else:
                            long_term_memory_courses.append('"Course {}"'.format(iid))

            # 构建用户描述
            prompt = "Given a user, "

            prompt += "this user's learning behaviors are categorized by the Atkinson-Shiffrin Memory Model: "

            # 添加三级记忆课程
            if sensory_memory_courses:
                prompt += "SENSORY MEMORY (immediate browsing): {}; ".format(', '.join(sensory_memory_courses[:6]))

            if working_memory_courses:
                prompt += "WORKING MEMORY (current learning session): {}; ".format(', '.join(working_memory_courses[:8]))

            if long_term_memory_courses:
                prompt += "LONG-TERM MEMORY (strategic interests): {}. ".format(', '.join(long_term_memory_courses[:8]))

            # 对比分析要求
            prompt += (
                "Compare and contrast the three memory levels to reveal the cognitive processing hierarchy. "
                "Explain how SENSORY MEMORY courses differ from WORKING MEMORY courses in terms of exploration versus consolidation. "
                "Analyze how WORKING MEMORY courses transition into LONG-TERM MEMORY for career planning. "
                "Identify patterns in memory consolidation and learning progression across the three levels. "
                "Describe the interactions and dependencies between different memory levels in shaping learning trajectories. "
                "Your response must be in English without subtitles, bullet points, or Chinese text. "
                "Translate any Chinese course names to English in your analysis."
            )

            memory_analysis_prompts[uid] = prompt
        else:
            raise NotImplementedError

    print('multilevel memory analysis prompts num', len(memory_analysis_prompts))
    print("Sample memory analysis prompt preview:", list(memory_analysis_prompts.values())[0][:250] + "...")
    return memory_analysis_prompts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_dir', default=None)
    cli_args = parser.parse_args()
    random.seed(12345)
    DATA_DIR = 'data/'
    DATA_SET_NAME = 'MOOCCube'
    PROCESSED_DIR = cli_args.proc_dir or os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')
    MULTILEVEL_MEMORY_PATH = os.path.join(PROCESSED_DIR, 'multilevel_memory.json')
    TIMESTAMP_PATH = os.path.join(PROCESSED_DIR, 'sequential_timestamps.json')
    PARTITION_CONFIG_PATH = os.path.join(PROCESSED_DIR, 'memory_partition_config.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    multilevel_memory_data = load_json(MULTILEVEL_MEMORY_PATH)
    timestamp_data = load_json(TIMESTAMP_PATH)
    partition_config = MemoryPartitionConfig.from_dict(load_json(PARTITION_CONFIG_PATH))
    datamap = load_json(DATAMAP_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(
        sequence_data, train_test_split['train'],
        train_test_split['temporal_cutoffs'], 'train'
    )
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(
        sequence_data, train_test_split['test'],
        train_test_split['temporal_cutoffs'], 'test'
    )
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    print('generating ranking train dataset (粗排)')
    train_rank = generate_rank_data(
        sequence_data, train_test_split['train'], item_set,
        train_test_split['temporal_cutoffs'], 'train'
    )
    print('generating ranking test dataset (粗排)')
    test_rank = generate_rank_data(
        sequence_data, train_test_split['test'], item_set,
        train_test_split['temporal_cutoffs'], 'test'
    )
    print('save ranking data')
    save_pickle(train_rank, PROCESSED_DIR + '/rank.train')
    save_pickle(test_rank, PROCESSED_DIR + '/rank.test')

    print('generating reranking train dataset (精排)')
    train_rerank = generate_rerank_data(
        sequence_data, train_test_split['train'], item_set,
        train_test_split['temporal_cutoffs'], 'train'
    )
    print('generating reranking test dataset (精排)')
    test_rerank = generate_rerank_data(
        sequence_data, train_test_split['test'], item_set,
        train_test_split['temporal_cutoffs'], 'test'
    )
    print('save reranking data')
    save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    cutoff_indices = collect_cutoff_indices(
        train_rank or [], test_rank or [], train_rerank or [], test_rerank or []
    )
    causal_multilevel_memory_data = build_causal_multilevel_memory(
        sequence_data,
        timestamp_data,
        item2attribute,
        datamap,
        cutoff_indices,
        partition_config,
    )
    save_json(causal_multilevel_memory_data, PROCESSED_DIR + '/causal_multilevel_memory.json')
    distribution_features, transition_features = compute_memory_features(
        causal_multilevel_memory_data, item2attribute
    )
    save_json(distribution_features, PROCESSED_DIR + '/enhanced_gating_features.json')
    save_json(transition_features, PROCESSED_DIR + '/transition_features.json')
    train_rank, test_rank = None, None
    train_rerank, test_rerank = None, None
    memory_for_prompts = causal_multilevel_memory_data

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 2,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')

    print('generating multilevel memory enhanced item prompt')
    item_prompt = generate_item_prompt_multilevel_memory(item2attribute, datamap, DATA_SET_NAME)
    print('generating multilevel memory enhanced history prompt')
    hist_prompt = generate_hist_prompt_multilevel_memory(sequence_data, item2attribute, datamap,
                                          train_test_split['lm_hist_idx'], memory_for_prompts, DATA_SET_NAME)
    print('generating multilevel memory analysis prompt')
    memory_analysis_prompt = generate_multilevel_memory_analysis_prompt(memory_for_prompts, datamap, DATA_SET_NAME)

    print('save prompt data')
    save_json(item_prompt, PROCESSED_DIR + '/prompt.item.multilevel_memory', ensure_ascii=False)
    save_json(hist_prompt, PROCESSED_DIR + '/prompt.hist.multilevel_memory', ensure_ascii=False)
    save_json(memory_analysis_prompt, PROCESSED_DIR + '/prompt.memory_analysis', ensure_ascii=False)

    item_prompt, hist_prompt, memory_analysis_prompt = None, None, None

    print("✅ MOOCCube 多级记忆增强的数据生成完成！")
    print("生成的文件:")
    print("  - prompt.item.multilevel_memory: 多级记忆增强的课程提示词")
    print("  - prompt.hist.multilevel_memory: 多级记忆增强的历史提示词")
    print("  - prompt.memory_analysis: 专门的多级记忆分析提示词")
