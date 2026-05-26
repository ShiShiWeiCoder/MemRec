import argparse
import json
import os
import pickle
import random
from collections import defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_pickle(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def sample_negative_items(item_set, positive_set, exclude_set, sample_num):
    candidates = list(item_set - positive_set - exclude_set - {0})
    if not candidates:
        return []
    if len(candidates) >= sample_num:
        return random.sample(candidates, sample_num)
    return random.choices(candidates, k=sample_num)


def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    records = []
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        for seq_idx in lm_hist_idx.get(str(uid), []):
            if 0 <= seq_idx < len(item_seq):
                records.append([uid, seq_idx, rating_seq[seq_idx]])
    return records


def generate_rank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    records = []
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        positives = {item for item, rating in zip(item_seq, rating_seq) if rating > 0}
        for seq_idx in lm_hist_idx.get(str(uid), []):
            history = set(item_seq[:seq_idx])
            positive_window = [
                item for item, rating in zip(item_seq[seq_idx : seq_idx + 5], rating_seq[seq_idx : seq_idx + 5])
                if rating > 0 and item not in history
            ][:5]
            negatives = sample_negative_items(item_set, positives, history, 50 - len(positive_window))
            candidates = positive_window + negatives
            labels = [1] * len(positive_window) + [0] * len(negatives)
            records.append([uid, seq_idx, candidates, labels])
    return records


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    records = []
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        positives = {item for item, rating in zip(item_seq, rating_seq) if rating > 0}
        for seq_idx in lm_hist_idx.get(str(uid), []):
            history = set(item_seq[:seq_idx])
            positive_window = [
                item for item, rating in zip(item_seq[seq_idx : seq_idx + 4], rating_seq[seq_idx : seq_idx + 4])
                if rating > 0 and item not in history
            ][:4]
            negatives = sample_negative_items(item_set, positives, history, 10 - len(positive_window))
            candidates = positive_window + negatives
            labels = [1] * len(positive_window) + [0] * len(negatives)
            records.append([uid, seq_idx, candidates, labels])
    return records


def collect_cutoff_indices(*datasets):
    cutoffs = defaultdict(set)
    for dataset in datasets:
        for uid, seq_idx, *_ in dataset:
            cutoffs[str(uid)].add(seq_idx)
    return {uid: sorted(values) for uid, values in cutoffs.items()}


def extract_causal_memory(prefix_items, prefix_ratings, item2attribute, datamap):
    positive_items = [item for item, rating in zip(prefix_items, prefix_ratings) if rating > 0]
    sensory = positive_items[-5:]
    working = positive_items[-15:]

    field_items = defaultdict(list)
    id2attribute = datamap.get("id2attribute", {})
    for item in positive_items:
        attrs = item2attribute.get(str(item), [])
        field = id2attribute.get(str(attrs[0]), "Unknown Field") if attrs else "Unknown Field"
        field_items[field].append(item)

    long_term = []
    for _, items in sorted(field_items.items(), key=lambda x: len(x[1]), reverse=True):
        long_term.extend(items[-10:])
        if len(long_term) >= 20:
            break

    return {
        "sensory_memory": sensory,
        "working_memory": working,
        "long_term_memory": long_term[:20],
    }


def build_causal_multilevel_memory(sequence_data, item2attribute, datamap, cutoffs):
    memory = {}
    for uid, indices in cutoffs.items():
        item_seq, rating_seq = sequence_data[str(uid)]
        for seq_idx in indices:
            key = f"{uid}:{seq_idx}"
            memory[key] = extract_causal_memory(
                item_seq[:seq_idx], rating_seq[:seq_idx], item2attribute, datamap
            )
    return memory


def generate_hist_prompt_multilevel_memory(sequence_data, item2attribute, datamap, lm_hist_idx, multilevel_memory_data, dataset_name):
    prompts = {}
    id2item = datamap.get("id2item", {})
    id2title = datamap.get("itemid2title", {})
    for uid, indices in lm_hist_idx.items():
        item_seq, _ = sequence_data[str(uid)]
        for seq_idx in indices:
            key = f"{uid}:{seq_idx}"
            memory = multilevel_memory_data.get(key, {})
            recent_titles = [id2title.get(str(item), id2item.get(str(item), "Unknown Course")) for item in item_seq[:seq_idx][-10:]]
            prompts[key] = (
                "You are an expert learning behavior analyst for course recommendation.\n"
                f"Dataset: {dataset_name}\n"
                f"Recent interaction courses: {recent_titles}\n"
                f"Sensory memory courses: {memory.get('sensory_memory', [])}\n"
                f"Working memory courses: {memory.get('working_memory', [])}\n"
                f"Long-term memory courses: {memory.get('long_term_memory', [])}\n\n"
                "Analyze this user's learning preferences considering factors such as subject domain, instructional approach, "
                "complexity level, pacing and duration, depth versus breadth, assessment methods, and real-world applications. "
                "Provide clear explanations based on the multilevel memory patterns. "
                "Focus on how immediate interests (sensory memory), current learning agenda (working memory), and stable accumulation "
                "(long-term memory) jointly indicate recommendation needs.\n\n"
                "Your response must be in English without subtitles, bullet points, or Chinese text. "
                "Translate any Chinese course names to English in your analysis."
            )
    return prompts


def generate_item_prompt_multilevel_memory(item2attribute, datamap, dataset_name):
    prompts = {}
    id2title = datamap.get("itemid2title", {})
    id2attribute = datamap.get("id2attribute", {})
    for item_id, attrs in item2attribute.items():
        fields = [id2attribute.get(str(attr), "Unknown Field") for attr in attrs]
        prompts[str(item_id)] = (
            "You are an expert in curriculum and learning pathway design.\n"
            f"Dataset: {dataset_name}\n"
            f"Course title: {id2title.get(str(item_id), 'Unknown Course')}\n"
            f"Course domain fields: {fields}\n\n"
            "Analyze this course from a learner-memory perspective. Explain: "
            "(1) what immediate curiosity or short-term motivation it can trigger, "
            "(2) what working-memory demands it places on learners in terms of prerequisite knowledge, "
            "cognitive load, practice style, and pacing, and "
            "(3) what long-term competencies or career-oriented value it can contribute.\n\n"
            "Your response must be in English without subtitles, bullet points, or Chinese text. "
            "Translate any Chinese course names to English in your analysis."
        )
    return prompts


def generate_multilevel_memory_analysis_prompt(multilevel_memory_data, datamap, dataset_name):
    prompts = {}
    for key, memory in multilevel_memory_data.items():
        prompts[key] = (
            "You are an expert in cognitive learning dynamics and recommendation reasoning.\n"
            f"Dataset: {dataset_name}\n"
            f"Sensory memory courses: {memory.get('sensory_memory', [])}\n"
            f"Working memory courses: {memory.get('working_memory', [])}\n"
            f"Long-term memory courses: {memory.get('long_term_memory', [])}\n\n"
            "Provide a Memory Transition Reflection by analyzing three transition processes: "
            "attention selection from sensory memory to working memory, consolidation from working memory to long-term memory, "
            "and retrieval influence from long-term memory to current learning choices. "
            "Explain which interests are likely temporary, which are becoming stable, and which long-term strengths may facilitate or "
            "interfere with future course selection. "
            "Conclude with an integrated judgment of the learner's current stage and near-term recommendation direction.\n\n"
            "Your response must be in English without subtitles, bullet points, or Chinese text. "
            "Translate any Chinese course names to English in your analysis."
        )
    return prompts


def main(proc_dir):
    sequence_data = load_json(os.path.join(proc_dir, "sequential_data.json"))
    item2attribute = load_json(os.path.join(proc_dir, "item2attributes.json"))
    datamap = load_json(os.path.join(proc_dir, "datamaps.json"))

    uid_set = [int(uid) for uid in sequence_data.keys()]
    random.shuffle(uid_set)
    split = int(len(uid_set) * 0.8)
    train_uids, test_uids = uid_set[:split], uid_set[split:]

    lm_hist_idx = {}
    for uid in uid_set:
        seq_len = len(sequence_data[str(uid)][0])
        lm_hist_idx[str(uid)] = list(range(1, seq_len))

    item_set = {int(item) for item in datamap["id2item"].keys() if int(item) != 0}
    rank_train = generate_rank_data(sequence_data, lm_hist_idx, train_uids, item_set)
    rank_test = generate_rank_data(sequence_data, lm_hist_idx, test_uids, item_set)
    rerank_train = generate_rerank_data(sequence_data, lm_hist_idx, train_uids, item_set)
    rerank_test = generate_rerank_data(sequence_data, lm_hist_idx, test_uids, item_set)

    cutoffs = collect_cutoff_indices(rank_train, rank_test, rerank_train, rerank_test)
    memory = build_causal_multilevel_memory(sequence_data, item2attribute, datamap, cutoffs)

    save_pickle(rank_train, os.path.join(proc_dir, "rank.train"))
    save_pickle(rank_test, os.path.join(proc_dir, "rank.test"))
    save_pickle(rerank_train, os.path.join(proc_dir, "rerank.train"))
    save_pickle(rerank_test, os.path.join(proc_dir, "rerank.test"))
    save_json({"train": train_uids, "test": test_uids, "lm_hist_idx": lm_hist_idx}, os.path.join(proc_dir, "train_test_split.json"))
    save_json(memory, os.path.join(proc_dir, "causal_multilevel_memory.json"))
    save_json(generate_hist_prompt_multilevel_memory(sequence_data, item2attribute, datamap, lm_hist_idx, memory, "MOOCCubeX"), os.path.join(proc_dir, "prompt.hist.multilevel_memory"))
    save_json(generate_item_prompt_multilevel_memory(item2attribute, datamap, "MOOCCubeX"), os.path.join(proc_dir, "prompt.item.multilevel_memory"))
    save_json(generate_multilevel_memory_analysis_prompt(memory, datamap, "MOOCCubeX"), os.path.join(proc_dir, "prompt.memory_analysis"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", default="data/MOOCCubeX/proc_data")
    args = parser.parse_args()
    main(args.proc_dir)
