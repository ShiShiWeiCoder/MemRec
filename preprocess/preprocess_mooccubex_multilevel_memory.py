import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


DEFAULT_SEED = 1234
SENSORY_MEMORY_LEN = 5
WORKING_MEMORY_LEN = 15
LONG_TERM_MIN_INTERACTIONS = 5
LONG_TERM_MIN_TIMESPAN = 0


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)


def parse_json_lines(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data, file_path):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def correct_title(title):
    return title.strip() if title else "Unknown Course"


def normalize_timestamp(value, fallback_index):
    if value in (None, ""):
        return float(fallback_index)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float(fallback_index)


def load_courses_new(course_file):
    courses = parse_json_lines(course_file)
    meta = {}
    for course in courses:
        raw_id = course.get("id")
        if raw_id is None:
            continue
        course_id = raw_id if str(raw_id).startswith("C_") else f"C_{raw_id}"
        meta[course_id] = {
            "name": correct_title(course.get("name")),
            "field": course.get("field") or course.get("category") or "Unknown Field",
        }
    return meta


def load_users(user_file):
    users = parse_json_lines(user_file)
    interactions = {}
    user_attrs = {}
    timestamps = {}
    for user in users:
        uid = str(user.get("id") or user.get("user_id"))
        course_order = user.get("course_order") or user.get("courses") or []
        time_order = user.get("time_order") or user.get("timestamps") or []
        pairs = []
        for idx, course_id in enumerate(course_order):
            item_id = str(course_id) if str(course_id).startswith("C_") else f"C_{course_id}"
            ts = normalize_timestamp(time_order[idx] if idx < len(time_order) else None, idx)
            pairs.append((item_id, ts, 1))
        pairs.sort(key=lambda x: x[1])
        interactions[uid] = [(item_id, rating) for item_id, _, rating in pairs]
        timestamps[uid] = {item_id: ts for item_id, ts, _ in pairs}
        user_attrs[uid] = {"anonymous_profile": user.get("profile", {})}
    return interactions, user_attrs, timestamps


def extract_multilevel_memory(user_items, meta_infos, user_timestamps=None):
    memory_data = {}
    for uid, interactions in user_items.items():
        positive_items = [item for item, rating in interactions if rating > 0]
        sensory = positive_items[-SENSORY_MEMORY_LEN:]
        working = positive_items[-WORKING_MEMORY_LEN:]

        field_counter = Counter()
        field_items = defaultdict(list)
        field_times = defaultdict(list)
        for item in positive_items:
            field = meta_infos.get(item, {}).get("field", "Unknown Field")
            field_counter[field] += 1
            field_items[field].append(item)
            if user_timestamps and uid in user_timestamps and item in user_timestamps[uid]:
                field_times[field].append(user_timestamps[uid][item])

        long_term = []
        for field, count in field_counter.most_common():
            if count < LONG_TERM_MIN_INTERACTIONS:
                continue
            if LONG_TERM_MIN_TIMESPAN > 0 and len(field_times[field]) > 1:
                timespan_days = (max(field_times[field]) - min(field_times[field])) / 86400.0
                if timespan_days < LONG_TERM_MIN_TIMESPAN:
                    continue
            long_term.extend(field_items[field][-20:])

        if not long_term:
            long_term = positive_items[:: max(1, len(positive_items) // 10)][:15]

        memory_data[uid] = {
            "sensory_memory": sensory,
            "working_memory": working,
            "long_term_memory": long_term[:20],
            "long_term_fields": [field for field, _ in field_counter.most_common(5)],
        }
    return memory_data


def check_Kcore(user_items, u_core, i_core):
    user_count = {u: sum(1 for _, r in items if r > 0) for u, items in user_items.items()}
    item_count = Counter()
    for items in user_items.values():
        for item, rating in items:
            if rating > 0:
                item_count[item] += 1
    return min(user_count.values() or [0]) >= u_core and min(item_count.values() or [0]) >= i_core


def filter_Kcore(user_items, u_core, i_core):
    filtered = dict(user_items)
    while not check_Kcore(filtered, u_core, i_core):
        item_count = Counter()
        for items in filtered.values():
            for item, rating in items:
                if rating > 0:
                    item_count[item] += 1
        filtered = {
            u: [(item, rating) for item, rating in items if rating <= 0 or item_count[item] >= i_core]
            for u, items in filtered.items()
        }
        filtered = {
            u: items
            for u, items in filtered.items()
            if sum(1 for _, rating in items if rating > 0) >= u_core
        }
    return filtered


def id_map(user_items, user_attrs, user_multilevel_memory):
    user2id, item2id = {}, {"<PAD>": 0}
    sequential_data = {}
    memory_id_data = {}

    for uid in sorted(user_items):
        mapped_uid = len(user2id) + 1
        user2id[uid] = mapped_uid
        item_seq, rating_seq = [], []
        for item, rating in user_items[uid]:
            if item not in item2id:
                item2id[item] = len(item2id)
            item_seq.append(item2id[item])
            rating_seq.append(rating)
        sequential_data[mapped_uid] = [item_seq, rating_seq]

        memory = user_multilevel_memory.get(uid, {})
        memory_id_data[mapped_uid] = {
            key: [item2id[item] for item in value if item in item2id]
            for key, value in memory.items()
            if key.endswith("_memory")
        }

    datamaps = {
        "user2id": user2id,
        "id2user": {v: k for k, v in user2id.items()},
        "item2id": item2id,
        "id2item": {v: k for k, v in item2id.items()},
    }
    return sequential_data, datamaps, memory_id_data


def get_attribute_mooc(meta_infos, data_maps):
    attribute2id = {"Unknown Field": 1}
    id2attribute = {1: "Unknown Field"}
    item2attributes = {}
    itemid2title = {}

    for mapped_id, raw_item in data_maps["id2item"].items():
        if raw_item == "<PAD>":
            item2attributes[mapped_id] = [0]
            itemid2title[mapped_id] = "<PAD>"
            continue
        meta = meta_infos.get(raw_item, {})
        field = meta.get("field", "Unknown Field")
        if field not in attribute2id:
            next_id = len(attribute2id) + 1
            attribute2id[field] = next_id
            id2attribute[next_id] = field
        item2attributes[mapped_id] = [attribute2id[field]]
        itemid2title[mapped_id] = meta.get("name", "Unknown Course")

    data_maps.update(
        {
            "attribute2id": attribute2id,
            "id2attribute": id2attribute,
            "itemid2title": itemid2title,
        }
    )
    return item2attributes, data_maps


def preprocess(course_file, user_file, processed_dir):
    meta_infos = load_courses_new(course_file)
    user_items, user_attrs, user_timestamps = load_users(user_file)

    user_items = filter_Kcore(user_items, u_core=5, i_core=5)
    memory_data = extract_multilevel_memory(user_items, meta_infos, user_timestamps)
    sequential_data, data_maps, memory_id_data = id_map(user_items, user_attrs, memory_data)
    item2attributes, data_maps = get_attribute_mooc(meta_infos, data_maps)

    os.makedirs(processed_dir, exist_ok=True)
    save_json(sequential_data, os.path.join(processed_dir, "sequential_data.json"))
    save_json(item2attributes, os.path.join(processed_dir, "item2attributes.json"))
    save_json(data_maps, os.path.join(processed_dir, "datamaps.json"))
    save_json(memory_id_data, os.path.join(processed_dir, "multilevel_memory.json"))
    save_json(
        {
            "user_num": len(data_maps["user2id"]),
            "item_num": len(data_maps["item2id"]),
            "attribute_num": len(data_maps["attribute2id"]),
            "attribute_ft_num": 1,
            "rating_num": 2,
            "dense_dim": 0,
            "rerank_list_len": 10,
        },
        os.path.join(processed_dir, "stat.json"),
    )


def compute_adaptive_window(total_interactions):
    return {
        "sensory": max(3, min(8, int(total_interactions * 0.08))),
        "working": max(10, min(30, int(total_interactions * 0.22))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_file", required=True)
    parser.add_argument("--user_file", required=True)
    parser.add_argument("--output_dir", default="data/MOOCCubeX/proc_data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    preprocess(args.course_file, args.user_file, args.output_dir)
