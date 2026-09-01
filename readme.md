# MemRec: Multi-Level Memory Augmented Course Recommendation

This repository contains the anonymized key implementation for a multi-level memory augmented recommendation model on MOOC-style course recommendation data. The released code focuses on the reproducible components needed for review: MOOCCubeX preprocessing, memory construction, LLM-based memory reflection, text encoding, and Rank/Rerank training.

## Anonymous Review Notice

This repository is prepared for anonymous review. Please host it with an anonymous service such as Anonymous GitHub or Anonymous OSF during the review period.

Before sharing the repository, remove contributor names, email addresses, affiliations, private machine paths, API keys, model service credentials, commit metadata, and any institution-specific information from README files, comments, configuration files, scripts, logs, and Git history. Do not upload raw private data, generated logs, checkpoints, or reviewer-identifying metadata.

## Public Code Layout

```text
readme.md
.gitignore
preprocess/
  pre_utils.py
  preprocess_mooccubex_multilevel_memory.py
  generate_mooccubex_multilevel_memory.py
knowledge-generating/
  llm_generating_mooccubex_multilevel_memory.py
knowledge_encoding/
  encode_analysis_bert.py
RS/
  dataset.py
  layers.py
  models.py
  optimization.py
  utils.py
  rank/main_rank_multilevel_memory.py
  rerank/main_rerank_multilevel_memory.py
scripts/
  run_portable_training.py
```

Generated folders such as `data/`, `logs/`, `figures/`, `__pycache__/`, model checkpoints, and local experiment notes are intentionally excluded from the public repository.

## Requirements

The code was developed with Python 3.8+ and PyTorch. A typical environment includes:

```bash
pip install torch transformers numpy scikit-learn tqdm requests
```

Optional encoders can be loaded from local paths or from Hugging Face model names, depending on the review environment.

## Data Layout

Raw and processed datasets are not included in this anonymous code release. After obtaining the dataset according to its license, place the files under:

```text
data/MOOCCubeX/raw_data/
```

The preprocessing scripts generate the following processed files under `data/MOOCCubeX/proc_data/`:

```text
sequential_data.json
sequential_timestamps.json
item2attributes.json
datamaps.json
train_test_split.json
multilevel_memory.json
causal_multilevel_memory.json
memory_partition_config.json
enhanced_gating_features.json
transition_features.json
rank.train / rank.test
rerank.train / rerank.test
```

## Pipeline

### 1. Preprocess Interactions and Build Memory Slots

```bash
python preprocess/preprocess_mooccubex_multilevel_memory.py \
  --adaptive_window \
  --sensory_ratio 0.08 \
  --working_ratio 0.22 \
  --time_window_days 30 \
  --sensory_tightening 4 \
  --long_term_threshold 7 \
  --long_term_min_timespan 30 \
  --output_dir data/MOOCCubeX/proc_data
```

### 2. Generate Rank/Rerank Candidate Lists and LLM Prompts

```bash
python preprocess/generate_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data
```

Rank uses fixed 50-candidate lists with up to 5 positive courses. Rerank uses fixed 10-candidate lists with up to 4 positive courses. Negatives are sampled from the global course set while excluding the user's positive courses.

### 3. Generate Memory Reflection Text

The default script calls a local OpenAI-compatible or Ollama-style LLM endpoint. Configure the endpoint and model in the script or through environment variables before running.

```bash
python knowledge-generating/llm_generating_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data \
  --output_dir data/MOOCCubeX/knowledge_multilevel_memory
```

This produces all three JSON-formatted knowledge streams required by equation (13):

```text
data/MOOCCubeX/knowledge_multilevel_memory/user_multilevel_memory.klg
data/MOOCCubeX/knowledge_multilevel_memory/item_multilevel_memory.klg
data/MOOCCubeX/knowledge_multilevel_memory/memory_analysis.klg
```

### 4. Encode User, Course, and Reflection Text

```bash
python knowledge_encoding/encode_analysis_bert.py \
  --knowledge_dir data/MOOCCubeX/knowledge_multilevel_memory \
  --data_dir data/MOOCCubeX/proc_data \
  --output_prefix bert_newprompt \
  --model_path bert-base-uncased
```

The command fails if any knowledge stream is absent and writes the complete training contract:

```text
bert_newprompt.hist
bert_newprompt.item
bert_newprompt.analysis
```

### 5. Train Rank Model

```bash
python RS/rank/main_rank_multilevel_memory.py \
  --data_dir data/MOOCCubeX/proc_data \
  --task rank \
  --algo DIN \
  --augment true \
  --aug_prefix bert_newprompt \
  --convert_type MultilevelMemoryHEA \
  --export_num 2 \
  --specific_export_num 3 \
  --convert_arch 128,32 \
  --convert_dropout 0.2 \
  --lr 1e-3 \
  --enhanced_gating \
  --reflection_mode \
  --fusion_mode film \
  --metric_scope 5,10 \
  --metrics_output results/paper_metrics/mooccubex_rank_din.json
```

### 6. Train Rerank Model

```bash
python RS/rerank/main_rerank_multilevel_memory.py \
  --data_dir data/MOOCCubeX/proc_data \
  --task rerank \
  --algo DLCM \
  --augment true \
  --aug_prefix bert_newprompt \
  --convert_type MultilevelMemoryHEA \
  --export_num 2 \
  --specific_export_num 3 \
  --convert_arch 128,32 \
  --convert_dropout 0.2 \
  --lr 1e-3 \
  --enhanced_gating \
  --reflection_mode \
  --fusion_mode film \
  --metric_scope 1,3,5 \
  --metrics_output results/paper_metrics/mooccubex_rerank_dlcm.json
```

## Evaluation Protocol

The public MemRec pipeline uses a per-user chronological 9:1 split, fixed candidate lists, and leakage-free negative samples. Every retained user has at least five interactions before the first supervised sample, and no candidate group crosses the train/test cutoff. Rank is evaluated on 50-candidate sampled lists and writes MAP@5/10, NDCG@5/10, HR@5/10, MRR, and AUC. Rerank is evaluated on 10-candidate sampled lists and writes MAP@1/3/5, NDCG@1/3/5, HR@1/3/5, and MRR. The public training entry point emits a machine-readable JSON report through `--metrics_output`.

## Reproducibility Notes

Set random seeds for reproducible runs. Keep generated LLM outputs and encoded vectors fixed when comparing configurations. Large generated artifacts, raw datasets, model checkpoints, and logs should be distributed separately from the anonymous code repository if the data license and review policy allow it.
