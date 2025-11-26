'''
Encode hierarchical memory enhanced knowledge using BERT
'''

import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from encoding.utils import save_json, get_paragraph_representation

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def load_data(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for id, value in data.items():
            res.append([id, value['prompt'], value['ans']])
    return res

def get_history_text(data_path):
    raw_data = load_data(data_path)
    idx_list, hist_text = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        pure_hist = prompt[::-1].split(';', 1)[-1][::-1]
        hist_text.append(pure_hist + '. ' + answer)
        idx_list.append(idx)
    return idx_list, hist_text

def get_item_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list

def get_memory_analysis_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list

def get_sensory_memory_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list

def get_working_memory_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list

def get_long_term_memory_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list

def get_text_data_loader_multilevel_memory(data_path, batch_size):
    user_memory_path = os.path.join(data_path, 'user.klg')
    hist_idxes, history = get_history_text(user_memory_path)
    
    item_memory_path = os.path.join(data_path, 'item.klg')
    item_idxes, items = get_item_text(item_memory_path)
    
    memory_analysis_path = os.path.join(data_path, 'memory_analysis.klg')
    analysis_idxes, analysis_texts = get_memory_analysis_text(memory_analysis_path)

    sensory_memory_path = os.path.join(data_path, 'sensory_memory.klg')
    working_memory_path = os.path.join(data_path, 'working_memory.klg')
    long_term_memory_path = os.path.join(data_path, 'long_term_memory.klg')
    
    sensory_idxes, sensory_texts = [], []
    working_idxes, working_texts = [], []
    longterm_idxes, longterm_texts = [], []
    
    if os.path.exists(sensory_memory_path):
        sensory_idxes, sensory_texts = get_sensory_memory_text(sensory_memory_path)
    
    if os.path.exists(working_memory_path):
        working_idxes, working_texts = get_working_memory_text(working_memory_path)
    
    if os.path.exists(long_term_memory_path):
        longterm_idxes, longterm_texts = get_long_term_memory_text(long_term_memory_path)

    history_loader = DataLoader(history, batch_size, shuffle=False)
    item_loader = DataLoader(items, batch_size, shuffle=False)
    analysis_loader = DataLoader(analysis_texts, batch_size, shuffle=False)
    
    sensory_loader = DataLoader(sensory_texts, batch_size, shuffle=False) if sensory_texts else None
    working_loader = DataLoader(working_texts, batch_size, shuffle=False) if working_texts else None
    longterm_loader = DataLoader(longterm_texts, batch_size, shuffle=False) if longterm_texts else None
    
    return (history_loader, hist_idxes, 
            item_loader, item_idxes, 
            analysis_loader, analysis_idxes,
            sensory_loader, sensory_idxes,
            working_loader, working_idxes,
            longterm_loader, longterm_idxes)

def remap_item(item_idxes, item_vec):
    item_vec_map = {}
    for idx, vec in zip(item_idxes, item_vec):
        item_vec_map[idx] = vec
    return item_vec_map

def inference(model, tokenizer, dataloader, model_name, aggregate_type):
    pred_list = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader, desc='Encoding'):
            if device == 'cuda':
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()
            
            x = tokenizer(x, padding=True, truncation=True, max_length=512, 
                         return_tensors="pt", return_attention_mask=True).to(device)
            mask = x['attention_mask']
            outputs = model(**x, output_hidden_states=True, return_dict=True)
            
            pred = get_paragraph_representation(outputs, mask, aggregate_type)
            pred_list.extend(pred.tolist())
    return pred_list

def main(knowledge_path, data_path, model_name, batch_size, aggregate_type):
    (hist_loader, hist_idxes, 
     item_loader, item_idxes, 
     analysis_loader, analysis_idxes,
     sensory_loader, sensory_idxes,
     working_loader, working_idxes,
     longterm_loader, longterm_idxes) = get_text_data_loader_multilevel_memory(knowledge_path, batch_size)

    if model_name == 'bert-base-uncased':
        checkpoint = 'bert-base-uncased'
        hidden_size = 768
    elif model_name == 'bert-base-cased':
        checkpoint = 'bert-base-cased'
        hidden_size = 768
    elif model_name == 'bert-large-uncased':
        checkpoint = 'bert-large-uncased'
        hidden_size = 1024
    elif model_name == 'bert-chinese':
        checkpoint = 'bert-base-chinese'
        hidden_size = 768
    elif model_name == 'bert-multilingual':
        checkpoint = 'bert-base-multilingual-cased'
        hidden_size = 768
    elif model_name == 'roberta-chinese':
        checkpoint = 'hfl/chinese-roberta-wwm-ext'
        hidden_size = 768
    elif model_name == 'bert-large-chinese':
        checkpoint = 'hfl/chinese-bert-wwm-ext'
        hidden_size = 768
    elif model_name == 'macbert':
        checkpoint = 'hfl/chinese-macbert-base'
        hidden_size = 768
    else:
        checkpoint = 'bert-base-uncased'
        hidden_size = 768
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if device == 'cpu':
        model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    elif device == 'mps':
        model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModel.from_pretrained(checkpoint).half().to(device)
    
    item_vec = inference(model, tokenizer, item_loader, model_name, aggregate_type)
    item_vec_dict = remap_item(item_idxes, item_vec)
    
    hist_vec = inference(model, tokenizer, hist_loader, model_name, aggregate_type)
    hist_vec_dict = remap_item(hist_idxes, hist_vec)
    
    analysis_vec = inference(model, tokenizer, analysis_loader, model_name, aggregate_type)
    analysis_vec_dict = remap_item(analysis_idxes, analysis_vec)

    sensory_vec_dict, working_vec_dict, longterm_vec_dict = {}, {}, {}
    
    if sensory_loader is not None:
        sensory_vec = inference(model, tokenizer, sensory_loader, model_name, aggregate_type)
        sensory_vec_dict = remap_item(sensory_idxes, sensory_vec)
    
    if working_loader is not None:
        working_vec = inference(model, tokenizer, working_loader, model_name, aggregate_type)
        working_vec_dict = remap_item(working_idxes, working_vec)
        
    if longterm_loader is not None:
        longterm_vec = inference(model, tokenizer, longterm_loader, model_name, aggregate_type)
        longterm_vec_dict = remap_item(longterm_idxes, longterm_vec)

    save_json(item_vec_dict, os.path.join(data_path, '{}_{}_augment_multilevel_memory.item'.format(model_name, aggregate_type)))
    save_json(hist_vec_dict, os.path.join(data_path, '{}_{}_augment_multilevel_memory.hist'.format(model_name, aggregate_type)))
    save_json(analysis_vec_dict, os.path.join(data_path, '{}_{}_augment_multilevel_memory.analysis'.format(model_name, aggregate_type)))
    
    if sensory_vec_dict:
        save_json(sensory_vec_dict, os.path.join(data_path, '{}_{}_augment_sensory_memory.vector'.format(model_name, aggregate_type)))
    if working_vec_dict:
        save_json(working_vec_dict, os.path.join(data_path, '{}_{}_augment_working_memory.vector'.format(model_name, aggregate_type)))
    if longterm_vec_dict:
        save_json(longterm_vec_dict, os.path.join(data_path, '{}_{}_augment_longterm_memory.vector'.format(model_name, aggregate_type)))

    stat_path = os.path.join(data_path, 'stat.json')
    with open(stat_path, 'r') as f:
        stat = json.load(f)

    stat['dense_dim'] = hidden_size
    stat['multilevel_memory_knowledge'] = {
        'user_memory_count': len(hist_vec_dict),
        'item_memory_count': len(item_vec_dict),
        'memory_analysis_count': len(analysis_vec_dict),
        'sensory_memory_count': len(sensory_vec_dict),
        'working_memory_count': len(working_vec_dict),
        'longterm_memory_count': len(longterm_vec_dict),
        'model_used': model_name,
        'checkpoint_used': checkpoint,
        'device_used': device,
        'hidden_size': hidden_size
    }
    
    with open(stat_path, 'w') as f:
        stat = json.dumps(stat, indent=2)
        f.write(stat)

    print(f"Encoded: Users={len(hist_vec_dict)}, Items={len(item_vec_dict)}, Analysis={len(analysis_vec_dict)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode multilevel memory knowledge')
    parser.add_argument('--dataset', type=str, default='mooc', choices=['mooc', 'coursera'],
                       help='Dataset name: mooc or coursera')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(base_dir, 'data')
    DATA_SET_NAME = args.dataset
    KLG_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'knowledge_multilevel_memory')
    DATA_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    
    MODEL_NAME = 'bert-base-uncased'
    AGGREGATE_TYPE = 'avg'
    BATCH_SIZE = 1 if device == 'cpu' else 8
    
    main(KLG_PATH, DATA_PATH, MODEL_NAME, BATCH_SIZE, AGGREGATE_TYPE)
