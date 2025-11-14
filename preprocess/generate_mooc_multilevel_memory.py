import os
import json
import random
import pickle
from collections import defaultdict
from pre_utils import load_json, save_json, save_pickle

# Parameters
rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10

# Threshold for implicit positive label
rating_threshold = 0

def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    full_data = []
    total_label = []
    for uid in uid_set:
        hist_idx_data = lm_hist_idx[str(uid)]
        # 如果lm_hist_idx存储的是列表，使用列表长度；如果是整数，直接使用
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label) if total_label else 0)
    print(full_data[:5])
    return full_data


def generate_rank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    """
    生成Rank阶段（粗排）的训练数据
    与Rerank的区别：候选数量更多（50个 vs 10个）
    """
    full_data = []
    rank_list_len = 50  # Rank阶段候选数量更多
    rank_item_from_hist = 5  # 从历史中选择的正样本数量
    
    for uid in uid_set:
        hist_idx_data = lm_hist_idx[str(uid)]
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
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


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    """
    生成Rerank阶段（精排）的训练数据
    候选数量较少（10个），更注重精细排序
    """
    full_data = []
    for uid in uid_set:
        hist_idx_data = lm_hist_idx[str(uid)]
        # 如果lm_hist_idx存储的是列表，使用列表长度；如果是整数，直接使用
        start_idx = len(hist_idx_data) if isinstance(hist_idx_data, list) else hist_idx_data
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
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
    
    for uid, item_rating in sequence_data.items():
        # 获取多级记忆数据
        # MOOC的multilevel_memory key是字符串
        memory_data = multilevel_memory_data.get(uid, {})
        
        # 构建感觉记忆文本(Sensory Memory)
        # MOOC数据结构: sensory_memory是[[course_ids], [ratings]]
        sensory_memory_texts = []
        if 'sensory_memory' in memory_data and memory_data['sensory_memory']:
            # memory_data['sensory_memory']是[[ids], [ratings]]格式
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
        
        if dataset_name == 'mooc':
            # 获取用户属性信息
            user_attrs = user2attribute.get(uid, {})
            user_info_parts = []
            if user_attrs.get('gender'):
                gender_value = user_attrs['gender']
                if gender_value == 1:
                    gender_text = 'male'
                elif gender_value == 2:
                    gender_text = 'female'
                else:
                    gender_text = str(gender_value)
                user_info_parts.append('{}'.format(gender_text))
            if user_attrs.get('school'):
                user_info_parts.append('from {}'.format(user_attrs['school']))
            
            # 构建提示词 - 只包含用户特征和三级记忆,不包含item信息
            if user_info_parts:
                user_description = ', '.join(user_info_parts)
                prompt = "Given a user who is {}, ".format(user_description)
            else:
                prompt = "Given a user, "
            
            prompt += "this user's course selections are organized by the Atkinson-Shiffrin Memory Model into three levels: "
            
            # 添加三级记忆课程(不包含career domains,因为那是从item中提取的)
            if sensory_memory_texts:
                prompt += "SENSORY MEMORY (immediate exploration needs): {}; ".format(', '.join(sensory_memory_texts[:5]))
            
            if working_memory_texts:
                prompt += "WORKING MEMORY (current learning session and short-term skill goals): {}; ".format(', '.join(working_memory_texts[:10]))
            
            if long_term_memory_texts:
                prompt += "LONG-TERM MEMORY (strategic career planning): {}. ".format(', '.join(long_term_memory_texts[:10]))
            
            # 分析要求 - 强调consider各种因素
            prompt += (
                "Analyze this user's learning preferences considering factors such as subject domain, instructional approach, "
                "complexity level, pacing and duration, depth versus breadth, assessment methods, and real-world applications. "
                "Provide clear explanations based on the multilevel memory patterns. "
                "Your response must be in English without subtitles, bullet points, or Chinese text. "
                "Translate any Chinese course names to English in your analysis."
            )
            
            hist_prompts[uid] = prompt
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
        if dataset_name == 'mooc':
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
        if dataset_name == 'mooc':
            # 获取用户属性
            user_attrs = user2attribute.get(uid, {})
            user_info_parts = []
            if user_attrs.get('gender'):
                gender_value = user_attrs['gender']
                gender_text = 'male' if gender_value == 1 else 'female' if gender_value == 2 else str(gender_value)
                user_info_parts.append('{}'.format(gender_text))
            if user_attrs.get('school'):
                user_info_parts.append('from {}'.format(user_attrs['school']))
            
            # 构建感觉记忆课程列表
            # MOOC数据结构: sensory_memory是[[course_ids], [ratings]]
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
            
            # 构建用户描述 - 只包含用户特征和三级记忆,不包含item信息
            if user_info_parts:
                user_description = ', '.join(user_info_parts)
                prompt = "Given a user who is {}, ".format(user_description)
            else:
                prompt = "Given a user, "
            
            prompt += "this user's learning behaviors are categorized by the Atkinson-Shiffrin Memory Model: "
            
            # 添加三级记忆课程(不包含career domains,因为那是从item中提取的)
            if sensory_memory_courses:
                prompt += "SENSORY MEMORY (immediate browsing): {}; ".format(', '.join(sensory_memory_courses[:6]))
            
            if working_memory_courses:
                prompt += "WORKING MEMORY (current learning session): {}; ".format(', '.join(working_memory_courses[:8]))
            
            if long_term_memory_courses:
                prompt += "LONG-TERM MEMORY (strategic interests): {}. ".format(', '.join(long_term_memory_courses[:8]))
            
            # 对比分析要求 - 重点在于对比三级记忆之间的关系,而非consider各种因素
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
    random.seed(12345)
    DATA_DIR = 'data/'
    DATA_SET_NAME = 'mooc'
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')
    MULTILEVEL_MEMORY_PATH = os.path.join(PROCESSED_DIR, 'multilevel_memory.json')  # 修改文件名

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    multilevel_memory_data = load_json(MULTILEVEL_MEMORY_PATH)  # 加载多级记忆数据
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'])
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'])
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    print('generating ranking train dataset (粗排)')
    train_rank = generate_rank_data(sequence_data, train_test_split['lm_hist_idx'],
                                    train_test_split['train'], item_set)
    print('generating ranking test dataset (粗排)')
    test_rank = generate_rank_data(sequence_data, train_test_split['lm_hist_idx'],
                                   train_test_split['test'], item_set)
    print('save ranking data')
    save_pickle(train_rank, PROCESSED_DIR + '/rank.train')
    save_pickle(test_rank, PROCESSED_DIR + '/rank.test')
    train_rank, test_rank = None, None
    
    print('generating reranking train dataset (精排)')
    train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                        train_test_split['train'], item_set)
    print('generating reranking test dataset (精排)')
    test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                       train_test_split['test'], item_set)
    print('save reranking data')
    save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    train_rerank, test_rerank = None, None

    datamap = load_json(DATAMAP_PATH)

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
                                          train_test_split['lm_hist_idx'], multilevel_memory_data, DATA_SET_NAME)
    print('generating multilevel memory analysis prompt')
    memory_analysis_prompt = generate_multilevel_memory_analysis_prompt(multilevel_memory_data, datamap, DATA_SET_NAME)
    
    print('save prompt data')
    save_json(item_prompt, PROCESSED_DIR + '/prompt.item.multilevel_memory', ensure_ascii=False)
    save_json(hist_prompt, PROCESSED_DIR + '/prompt.hist.multilevel_memory', ensure_ascii=False)
    save_json(memory_analysis_prompt, PROCESSED_DIR + '/prompt.memory_analysis', ensure_ascii=False)  # 新增
    
    item_prompt, hist_prompt, memory_analysis_prompt = None, None, None
    
    print("✅ 多级记忆增强的数据生成完成！")
    print("生成的文件:")
    print("  - prompt.item.multilevel_memory: 多级记忆增强的课程提示词")
    print("  - prompt.hist.multilevel_memory: 多级记忆增强的历史提示词")
    print("  - prompt.memory_analysis: 专门的多级记忆分析提示词") 