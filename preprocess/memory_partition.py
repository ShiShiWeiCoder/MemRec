"""Paper-aligned, point-in-time memory partitioning for MemRec."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


SECONDS_PER_DAY = 86400.0
DEFAULT_MIN_HISTORY = 5


@dataclass(frozen=True)
class MemoryPartitionConfig:
    time_window_days: float = 30.0
    sensory_ratio: float = 0.08
    working_ratio: float = 0.22
    sensory_tightening: float = 4.0
    long_term_threshold: int = 7
    long_term_min_timespan_days: float = 30.0

    def validate(self) -> None:
        if not 0 < self.sensory_ratio <= self.working_ratio <= 1:
            raise ValueError("expected 0 < sensory_ratio <= working_ratio <= 1")
        if self.time_window_days <= 0:
            raise ValueError("time_window_days must be positive")
        if not 3 <= self.sensory_tightening <= 6:
            raise ValueError("sensory_tightening must be in [3, 6]")
        if self.long_term_threshold <= 0:
            raise ValueError("long_term_threshold must be positive")
        if self.long_term_min_timespan_days < 0:
            raise ValueError("long_term_min_timespan_days cannot be negative")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping | None) -> "MemoryPartitionConfig":
        values = values or {}
        aliases = {
            "long_term_min_timespan": "long_term_min_timespan_days",
        }
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in normalized.items() if key in allowed})
        config.validate()
        return config


def _split_pairs(records: Sequence[tuple]) -> list[list]:
    return [
        [record[1] for record in records],
        [record[2] for record in records],
    ]


def align_filtered_timestamps(
    original_items: Sequence[tuple],
    original_timestamps: Sequence[float],
    filtered_items: Sequence[tuple],
) -> list[float]:
    """Keep timestamps aligned when K-core filtering removes interactions."""

    if len(original_items) != len(original_timestamps):
        raise ValueError("original items and timestamps must have identical lengths")
    aligned = []
    target_index = 0
    for item, timestamp in zip(original_items, original_timestamps):
        if target_index < len(filtered_items) and item == filtered_items[target_index]:
            aligned.append(float(timestamp))
            target_index += 1
    if target_index != len(filtered_items):
        raise ValueError("filtered interactions are not an ordered subsequence")
    return aligned


def build_temporal_train_test_split(
    sequence_data: Mapping,
    train_ratio: float = 0.9,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> dict:
    """Split every eligible user's chronological sequence at one cutoff."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if min_history < 0:
        raise ValueError("min_history cannot be negative")

    eligible_users = []
    temporal_cutoffs = {}
    for user_id, sequence in sequence_data.items():
        if len(sequence) != 2 or len(sequence[0]) != len(sequence[1]):
            raise ValueError(f"invalid item/rating sequence for user {user_id}")
        sequence_length = len(sequence[0])
        cutoff = max(min_history, math.floor(train_ratio * sequence_length))
        if cutoff >= sequence_length:
            continue
        normalized_user_id = int(user_id) if str(user_id).isdigit() else user_id
        eligible_users.append(normalized_user_id)
        temporal_cutoffs[str(user_id)] = cutoff

    return {
        "train": eligible_users,
        "test": list(eligible_users),
        "temporal_cutoffs": temporal_cutoffs,
        "train_ratio": train_ratio,
        "min_history": min_history,
    }


def temporal_sample_bounds(
    sequence_length: int,
    cutoff: int,
    split: str,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[int, int]:
    """Return the half-open sample interval for one temporal split."""

    cutoff = int(cutoff)
    if not min_history <= cutoff < sequence_length:
        raise ValueError(
            "temporal cutoff must retain the minimum history and at least one test item"
        )
    if split == "train":
        return min_history, cutoff
    if split == "test":
        return cutoff, sequence_length
    raise ValueError("split must be 'train' or 'test'")


def partition_memory_at_cutoff(
    item_ids: Sequence,
    ratings: Sequence,
    timestamps: Sequence[float],
    item_to_fields: Mapping,
    field_names: Mapping | None = None,
    config: MemoryPartitionConfig | None = None,
    rating_threshold: float = 0,
) -> dict:
    """Apply paper equations (6)-(9) to history strictly before one cutoff.

    Timestamps must be Unix-like seconds and aligned with ``item_ids``. The
    caller is responsible for slicing all three sequences at ``seq_idx``.
    """

    config = config or MemoryPartitionConfig()
    config.validate()
    if not (len(item_ids) == len(ratings) == len(timestamps)):
        raise ValueError(
            "item, rating, and timestamp sequences must have identical lengths"
        )
    if any(float(current) < float(previous) for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("timestamps must be sorted in nondecreasing order")

    positive = [
        (position, item_id, rating, float(timestamp))
        for position, (item_id, rating, timestamp) in enumerate(
            zip(item_ids, ratings, timestamps)
        )
        if rating > rating_threshold
    ]
    if not positive:
        return {
            "sensory_memory": [[], []],
            "working_memory": [[], []],
            "long_term_memory": [[], []],
            "long_term_fields": [],
        }

    n_interactions = len(positive)
    latest_timestamp = positive[-1][3]
    sensory_rank_limit = math.ceil(config.sensory_ratio * n_interactions)
    working_rank_limit = math.ceil(config.working_ratio * n_interactions)
    sensory_seconds = (
        config.time_window_days * SECONDS_PER_DAY / config.sensory_tightening
    )
    working_seconds = config.time_window_days * SECONDS_PER_DAY

    sensory = []
    working = []
    for positive_index, record in enumerate(positive):
        reverse_rank = n_interactions - positive_index
        elapsed = max(0.0, latest_timestamp - record[3])
        if reverse_rank <= sensory_rank_limit and elapsed <= sensory_seconds:
            sensory.append(record)
        elif (
            sensory_rank_limit < reverse_rank <= working_rank_limit
            and elapsed <= working_seconds
        ):
            working.append(record)

    field_records = {}
    for record in positive:
        for field in item_to_fields.get(str(record[1]), item_to_fields.get(record[1], [])):
            field_records.setdefault(str(field), []).append(record)

    qualifying_fields = []
    selected_positions = set()
    minimum_span_seconds = config.long_term_min_timespan_days * SECONDS_PER_DAY
    for field, records in field_records.items():
        span_seconds = records[-1][3] - records[0][3]
        if len(records) < config.long_term_threshold:
            continue
        if span_seconds < minimum_span_seconds:
            continue
        qualifying_fields.append(field)
        for index in (0, len(records) // 2, len(records) - 1):
            selected_positions.add(records[index][0])

    long_term = [record for record in positive if record[0] in selected_positions]
    field_names = field_names or {}
    long_term_fields = [
        field_names.get(field, field_names.get(int(field), field))
        if field.isdigit()
        else field_names.get(field, field)
        for field in qualifying_fields
    ]

    return {
        "sensory_memory": _split_pairs(sensory),
        "working_memory": _split_pairs(working),
        "long_term_memory": _split_pairs(long_term),
        "long_term_fields": long_term_fields,
    }
