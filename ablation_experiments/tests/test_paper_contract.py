#!/usr/bin/env python3
"""Focused regression tests for the public MemRec paper contract."""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RS"))
sys.path.insert(0, str(ROOT / "preprocess"))

from knowledge_encoding.encode_analysis_bert import load_stream, select_jobs
from generate_mooccube_multilevel_memory import (
    generate_ctr_data as generate_mooccube_ctr_data,
    generate_rank_data as generate_mooccube_rank_data,
    generate_rerank_data as generate_mooccube_rerank_data,
)
from generate_mooccubex_multilevel_memory import (
    generate_ctr_data as generate_mooccubex_ctr_data,
    generate_rank_data as generate_mooccubex_rank_data,
    generate_rerank_data as generate_mooccubex_rerank_data,
)
from layers import MultilevelMemoryHEA
from memory_features import compute_memory_features
from memory_partition import (
    SECONDS_PER_DAY,
    MemoryPartitionConfig,
    align_filtered_timestamps,
    build_temporal_train_test_split,
    partition_memory_at_cutoff,
)
from models import (
    AutoInt,
    DCN,
    DIEN,
    DLCM,
    DeepFM,
    DeepInterestNet,
    EGRerank,
    FiBiNet,
    FiGNN,
    GSF,
    LambdaRank,
    MIR,
    PEAR,
    PIER,
    PRM,
    RankFormer,
    SetRank,
    xDeepFM,
)
from utils import evaluate_rerank, save_paper_metrics


class MemoryPartitionTests(unittest.TestCase):
    def test_temporal_windows_and_no_future_input(self):
        days = [0, 1, 2, 3, 4, 5, 6, 7, 29, 30]
        memory = partition_memory_at_cutoff(
            list(range(1, 11)),
            [1] * 10,
            [day * SECONDS_PER_DAY for day in days],
            {},
        )
        self.assertEqual(memory["sensory_memory"][0], [10])
        self.assertEqual(memory["working_memory"][0], [8, 9])
        self.assertNotIn(11, memory["sensory_memory"][0])
        self.assertNotIn(11, memory["working_memory"][0])

    def test_long_term_requires_frequency_and_span(self):
        item_ids = list(range(1, 8))
        ratings = [1] * 7
        fields = {str(item_id): ["domain-a"] for item_id in item_ids}
        config = MemoryPartitionConfig()
        qualifying = partition_memory_at_cutoff(
            item_ids,
            ratings,
            [day * SECONDS_PER_DAY for day in (0, 5, 10, 15, 20, 25, 30)],
            fields,
            config=config,
        )
        self.assertEqual(qualifying["long_term_memory"][0], [1, 4, 7])
        self.assertEqual(qualifying["long_term_fields"], ["domain-a"])

        short_burst = partition_memory_at_cutoff(
            item_ids,
            ratings,
            [day * SECONDS_PER_DAY for day in (0, 4, 8, 12, 16, 20, 29)],
            fields,
            config=config,
        )
        self.assertEqual(short_burst["long_term_memory"][0], [])
        self.assertEqual(short_burst["long_term_fields"], [])

    def test_duplicate_interactions_keep_timestamp_alignment(self):
        original = [("a", 1), ("a", 1), ("b", 1)]
        filtered = [("a", 1), ("b", 1)]
        self.assertEqual(
            align_filtered_timestamps(original, [1, 2, 3], filtered),
            [1.0, 3.0],
        )

    def test_rejects_unsorted_timestamps(self):
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            partition_memory_at_cutoff([1, 2], [1, 1], [2, 1], {})


class TemporalDatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.sequence_data = {
            "1": [list(range(1, 21)), [1] * 20],
            "2": [list(range(21, 26)), [1] * 5],
        }
        self.split = build_temporal_train_test_split(self.sequence_data)
        self.item_set = list(range(1, 101))

    def test_every_eligible_user_is_split_chronologically(self):
        self.assertEqual(self.split["train"], [1])
        self.assertEqual(self.split["test"], [1])
        self.assertEqual(self.split["temporal_cutoffs"], {"1": 18})
        self.assertEqual(self.split["min_history"], 5)

        for generator in (generate_mooccube_ctr_data, generate_mooccubex_ctr_data):
            with self.subTest(generator=generator.__module__):
                train_ctr = generator(
                    self.sequence_data, self.split["train"],
                    self.split["temporal_cutoffs"], "train"
                )
                test_ctr = generator(
                    self.sequence_data, self.split["test"],
                    self.split["temporal_cutoffs"], "test"
                )
                self.assertEqual(
                    [sample[1] for sample in train_ctr], list(range(5, 18))
                )
                self.assertEqual([sample[1] for sample in test_ctr], [18, 19])
                self.assertGreaterEqual(test_ctr[0][1], 5)

    def test_candidate_groups_do_not_cross_temporal_cutoff(self):
        cutoff = self.split["temporal_cutoffs"]["1"]
        positive_items = set(self.sequence_data["1"][0])
        generators = (
            (generate_mooccube_rank_data, 50),
            (generate_mooccube_rerank_data, 10),
            (generate_mooccubex_rank_data, 50),
            (generate_mooccubex_rerank_data, 10),
        )
        for generator, expected_list_length in generators:
            with self.subTest(generator=generator.__name__):
                train_data = generator(
                    self.sequence_data, self.split["train"], self.item_set,
                    self.split["temporal_cutoffs"], "train"
                )
                test_data = generator(
                    self.sequence_data, self.split["test"], self.item_set,
                    self.split["temporal_cutoffs"], "test"
                )
                for _, seq_idx, candidates, labels in train_data:
                    self.assertLess(seq_idx, cutoff)
                    self.assertEqual(len(candidates), expected_list_length)
                    selected = [item for item, label in zip(candidates, labels) if label]
                    self.assertTrue(all(item <= cutoff for item in selected))
                    negatives = [item for item, label in zip(candidates, labels) if not label]
                    self.assertTrue(positive_items.isdisjoint(negatives))
                for _, seq_idx, candidates, labels in test_data:
                    self.assertGreaterEqual(seq_idx, cutoff)
                    self.assertEqual(len(candidates), expected_list_length)
                    selected = [item for item, label in zip(candidates, labels) if label]
                    self.assertTrue(all(item > cutoff for item in selected))
                    negatives = [item for item, label in zip(candidates, labels) if not label]
                    self.assertTrue(positive_items.isdisjoint(negatives))


class FeatureTests(unittest.TestCase):
    def test_distribution_and_transition_dimensions_and_values(self):
        memory = {
            "u:4": {
                "sensory_memory": [[1, 2], [1, 1]],
                "working_memory": [[2, 3], [1, 1]],
                "long_term_memory": [[3, 4], [1, 1]],
            }
        }
        fields = {"1": ["a"], "2": ["b"], "3": ["b"], "4": ["c"]}
        distribution, transition = compute_memory_features(memory, fields)
        self.assertEqual(len(distribution["u:4"]), 6)
        self.assertEqual(len(transition["u:4"]), 9)
        expected_distribution = [1.0, 0.0, 1.0, 0.5, 0.5, 1.0 / 3.0]
        for actual, expected in zip(distribution["u:4"], expected_distribution):
            self.assertAlmostEqual(actual, expected)
        expected_transition = [1.0, 0.5, 1.0, 0.5, 2.0, 2.0 / 3.0, 0.5, 1.0, 0.0]
        for actual, expected in zip(transition["u:4"], expected_transition):
            self.assertAlmostEqual(actual, expected)


class BertContractTests(unittest.TestCase):
    def test_all_jobs_and_answer_only_encoding(self):
        args = SimpleNamespace(
            output_path=None,
            stream="all",
            data_dir="/tmp/proc",
            output_prefix="bert_newprompt",
        )
        self.assertEqual(
            select_jobs(args),
            [
                ("hist", "/tmp/proc/bert_newprompt.hist"),
                ("item", "/tmp/proc/bert_newprompt.item"),
                ("analysis", "/tmp/proc/bert_newprompt.analysis"),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "user.klg")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"u:4": {"prompt": "do not encode", "ans": "encoded text"}}, handle)
            self.assertEqual(load_stream(path, "hist"), {"u:4": "encoded text"})

    def test_empty_llm_answers_fail_before_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "memory_analysis.klg")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"u:4": {"prompt": "p", "ans": ""}}, handle)
            with self.assertRaisesRegex(ValueError, "empty answers"):
                load_stream(path, "analysis")


class FusionTests(unittest.TestCase):
    def test_paper_film_and_two_plus_three_moe(self):
        torch.manual_seed(7)
        module = MultilevelMemoryHEA(
            [2, 3, [128, 32], 2],
            768,
            dropout=0.0,
            enhanced_gating=True,
            reflection_mode=True,
            transition_feature_dim=9,
            analysis_dim=768,
            fusion_mode="film",
        )
        self.assertEqual(len(module.share_expt_net), 2)
        self.assertEqual(len(module.spcf_expt_net), 2)
        self.assertTrue(all(len(experts) == 3 for experts in module.spcf_expt_net))
        self.assertEqual(module.film_user[0].out_features, 1536)
        self.assertEqual(module.film_user[-1].out_features, 1536)
        self.assertFalse(hasattr(module, "film_dropout"))
        self.assertEqual(module.gate_net[0][0].in_features, 768 + 6 + 9)

        user = torch.randn(2, 768, requires_grad=True)
        course = torch.randn(2, 768, requires_grad=True)
        mtr = torch.randn(2, 768, requires_grad=True)
        memory = {
            "enhanced_gating_features": torch.randn(2, 6),
            "transition_features": torch.randn(2, 9),
        }
        captured_gate_input = []
        hook = module.gate_net[0].register_forward_pre_hook(
            lambda _module, inputs: captured_gate_input.append(inputs[0].detach())
        )
        outputs = module([user, course, mtr], memory)
        hook.remove()
        self.assertEqual([tuple(value.shape) for value in outputs], [(2, 32), (2, 32)])
        self.assertEqual(tuple(torch.cat(outputs, dim=-1).shape), (2, 64))

        normalized_mtr = torch.nn.functional.normalize(mtr, p=2, dim=-1)
        gamma, beta = module.film_user(normalized_mtr).chunk(2, dim=-1)
        expected_user = gamma * user + beta
        torch.testing.assert_close(captured_gate_input[0][:, :768], expected_user)

        sum(value.square().mean() for value in outputs).backward()
        self.assertIsNotNone(mtr.grad)
        self.assertIsNotNone(module.film_user[0].weight.grad)
        self.assertIsNotNone(module.gate_net[0][0].weight.grad)
        self.assertIsNotNone(module.share_expt_net[0].fc[0].weight.grad)
        self.assertIsNotNone(module.spcf_expt_net[0][0].fc[0].weight.grad)


class _Dataset:
    item_num = 20
    attr_num = 8
    attr_ft_num = 1
    rating_num = 2
    dense_dim = 0
    aug_vec_dim = 8
    max_list_len = 4


def _args():
    return SimpleNamespace(
        task="rerank",
        augment=True,
        max_hist_len=3,
        embed_dim=4,
        final_mlp_arch=[12, 8],
        dropout=0.0,
        hidden_size=8,
        rnn_dp=0.0,
        output_dim=1,
        convert_dropout=0.0,
        convert_type="MultilevelMemoryHEA",
        auxi_loss_weight=0.0,
        export_num=2,
        specific_export_num=3,
        convert_arch=[8, 4],
        device="cpu",
        memory_mode=True,
        enable_memory_attention=True,
        memory_fusion_type="attention",
        enhanced_gating=True,
        reflection_mode=True,
        transition_feature_dim=9,
        analysis_vec_dim=8,
        analysis_reduction_dim=0,
        enable_knowledge_reduction=False,
        unified_reduction=False,
        knowledge_reduction_dim=8,
        knowledge_reduction_dropout=0.0,
        analysis_as_expert=False,
        fusion_mode="film",
        skip_user_profile=False,
        skip_course_profile=False,
        skip_analysis=False,
        enable_ls_attention=False,
        dcn_deep_arch=[8],
        dcn_cross_num=2,
        deepfm_deep_arch=[8],
        num_attn_layers=1,
        num_attn_heads=1,
        attn_size=4,
        res_conn=False,
        attn_scale=False,
        reduction_ratio=2,
        bilinear_type="field_interaction",
        cin_layer_units=[4, 4],
        direct=False,
        gnn_layer_num=2,
        reuse_graph_layer=False,
        dien_gru="GRU",
        n_head=2,
        attn_dp=0.0,
        ff_dim=32,
        n_layers=1,
        temperature=1.0,
    )


def _input(augmentation_value):
    batch_size, list_len, hist_len, vector_dim = 2, 4, 3, 8
    return {
        "iid_list": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "aid_list": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        "lb_list": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
        "hist_iid_seq": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "hist_aid_seq": torch.tensor([[1, 2, 3], [2, 3, 4]]),
        "hist_rate_seq": torch.ones(batch_size, hist_len, dtype=torch.long),
        "hist_seq_len": torch.full((batch_size,), hist_len, dtype=torch.long),
        "hist_aug_vec": torch.full((batch_size, vector_dim), augmentation_value),
        "item_aug_vec_list": [
            torch.full((batch_size, vector_dim), augmentation_value)
            for _ in range(list_len)
        ],
        "memory_analysis_aug_vec": torch.full(
            (batch_size, vector_dim), augmentation_value
        ),
        "enhanced_gating_features": torch.zeros(batch_size, 6),
        "transition_features": torch.zeros(batch_size, 9),
        "multilevel_memory_mode": torch.ones(batch_size),
    }


class BackboneTests(unittest.TestCase):
    def test_all_public_backbones_consume_memrec_vector(self):
        public_models = [
            ("DeepFM", lambda: DeepFM(_args(), _Dataset())),
            ("xDeepFM", lambda: xDeepFM(_args(), _Dataset())),
            ("DCN", lambda: DCN(_args(), "v1", _Dataset())),
            ("FiBiNet", lambda: FiBiNet(_args(), _Dataset())),
            ("FiGNN", lambda: FiGNN(_args(), _Dataset())),
            ("AutoInt", lambda: AutoInt(_args(), _Dataset())),
            ("DIN", lambda: DeepInterestNet(_args(), _Dataset())),
            ("DIEN", lambda: DIEN(_args(), _Dataset())),
            ("DLCM", lambda: DLCM(_args(), _Dataset())),
            ("PRM", lambda: PRM(_args(), _Dataset())),
            ("SetRank", lambda: SetRank(_args(), _Dataset())),
            ("MIR", lambda: MIR(_args(), _Dataset())),
            ("GSF", lambda: GSF(_args(), _Dataset())),
            ("EGRerank", lambda: EGRerank(_args(), _Dataset())),
            ("LambdaRank", lambda: LambdaRank(_args(), _Dataset())),
            ("RankFormer", lambda: RankFormer(_args(), _Dataset())),
            ("PEAR", lambda: PEAR(_args(), _Dataset())),
            ("PIER", lambda: PIER(_args(), _Dataset())),
        ]
        for name, factory in public_models:
            with self.subTest(backbone=name):
                torch.manual_seed(11)
                model = factory().eval()
                with torch.no_grad():
                    baseline = model(_input(0.0))["logits"]
                    augmented = model(_input(1.0))["logits"]
                self.assertEqual(tuple(baseline.shape), (2, 4))
                self.assertFalse(
                    torch.allclose(baseline, augmented),
                    f"{name} output did not change when MemRec vectors changed",
                )


class MetricReportTests(unittest.TestCase):
    def test_map_at_k_penalizes_missing_relevant_items(self):
        results = evaluate_rerank(
            labels=[[1, 1, 0, 0]],
            preds=[[0.9, 0.1, 0.8, 0.7]],
            scope_number=[3],
            is_rank=True,
        )
        self.assertAlmostEqual(results[0][0], 0.5)

    def test_machine_readable_paper_schema(self):
        results = (
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            0.7,
            0.8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.json")
            report = save_paper_metrics(path, "rank", "DIN", [5, 10], results, seed=1)
            self.assertEqual(
                set(report["metrics"]),
                {
                    "MAP@5",
                    "MAP@10",
                    "NDCG@5",
                    "NDCG@10",
                    "HR@5",
                    "HR@10",
                    "MRR",
                    "AUC",
                },
            )
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
