"""Distribution and transition features from paper Section IV-B."""

from __future__ import annotations

import math
from collections import Counter


def _memory_ids(memory):
    if not memory:
        return []
    return memory[0] if isinstance(memory[0], list) else memory


def _domains(item_ids, item_to_fields):
    counts = Counter()
    for item_id in item_ids:
        for field in item_to_fields.get(str(item_id), []):
            counts[str(field)] += 1
    return counts, set(counts)


def _entropy(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _jaccard(left, right):
    return _ratio(len(left & right), len(left | right))


def compute_memory_features(memory_by_cutoff, item_to_fields):
    """Return 6-D distribution and 9-D transition features per cutoff."""

    distribution = {}
    transition = {}
    for key, memory in memory_by_cutoff.items():
        sensory_counts, sensory = _domains(
            _memory_ids(memory.get("sensory_memory", [])), item_to_fields
        )
        working_counts, working = _domains(
            _memory_ids(memory.get("working_memory", [])), item_to_fields
        )
        long_counts, long_term = _domains(
            _memory_ids(memory.get("long_term_memory", [])), item_to_fields
        )
        distribution[key] = [
            _entropy(sensory_counts),
            _entropy(working_counts),
            _entropy(long_counts),
            _jaccard(sensory, working),
            _jaccard(working, long_term),
            _jaccard(sensory, long_term),
        ]

        all_domains = sensory | working | long_term
        emerging = sensory - working - long_term
        consolidation = sensory & working
        transition[key] = [
            float(len(emerging)),
            _ratio(len(emerging), len(sensory)),
            float(len(consolidation)),
            _ratio(len(consolidation), len(sensory)),
            float(len(long_term)),
            _ratio(len(long_term), len(all_domains)),
            _ratio(len(sensory & working), len(sensory)),
            _ratio(len(working & long_term), len(working)),
            _ratio(len((sensory & long_term) - working), len(sensory)),
        ]
    return distribution, transition
