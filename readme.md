# MemRec: Memory-Guided Course Recommendation

MemRec is a memory-guided knowledge augmentation framework for course recommendation. It models learner histories through the Atkinson-Shiffrin memory view: sensory memory captures immediate exploration, working memory captures the current learning task, and long-term memory captures stable domain interests and career-oriented expertise.

The system uses large language models as offline reflection engines rather than as end-to-end recommenders. It generates structured semantic knowledge from memory-separated histories and course metadata, encodes the generated text with BERT, and injects the resulting vectors into ranking and reranking backbones through Memory Transition Reflection (MTR), FiLM-style conditioning, and memory-aware expert routing.

## System Overview

MemRec follows a five-stage pipeline:

1. Separate each learner history into sensory, working, and long-term memory layers.
2. Build distribution and transition features that describe memory concentration, overlap, emerging exploration, consolidation, and retrieval influence.
3. Generate three types of LLM-based knowledge: a user multi-level memory profile, a course cognitive attribute description, and an MTR reflection.
4. Encode generated texts with BERT-base-uncased and mean pooling.
5. Append the final memory-guided augmentation vector to standard Rank or Rerank backbone inputs.

The framework is model-agnostic. In the paper, MemRec is integrated with six ranking backbones, including DeepFM, DCN, FiBiNet, AutoInt, DIN, and DIEN, and four reranking backbones, including DLCM, PRM, RankFormer, and PIER.

## Repository Structure

```text
readme.md
preprocess/preprocess_mooccubex_multilevel_memory.py
preprocess/generate_mooccubex_multilevel_memory.py
knowledge-generating/llm_generating_mooccubex_multilevel_memory.py
knowledge_encoding/encode_analysis_bert.py
RS/dataset.py
RS/layers.py
RS/models.py
RS/rank/main_rank_multilevel_memory.py
RS/rerank/main_rerank_multilevel_memory.py
```

## Requirements

The code uses Python 3.8+ with common scientific and deep learning packages:

```bash
pip install torch numpy scikit-learn tqdm transformers requests
```

## Data Layout and Preprocessing

Dataset sources:

- MOOCCube: http://moocdata.cn/data/MOOCCube
- MOOCCubeX: https://github.com/THU-KEG/MOOCCubeX

After obtaining MOOCCubeX under the relevant license, place raw files under:

```text
data/MOOCCubeX/raw_data/
```

The preprocessing stage writes processed files under:

```text
data/MOOCCubeX/proc_data/
```

The preprocessing stage sorts interactions chronologically, applies core filtering, constructs memory slots under point-wise temporal cutoffs, and writes processed files such as:

```text
sequential_data.json
item2attributes.json
datamaps.json
stat.json
train_test_split.json
causal_multilevel_memory.json
rank.train / rank.test
rerank.train / rerank.test
prompt.hist.multilevel_memory
prompt.item.multilevel_memory
prompt.memory_analysis
```

The default memory parameters match the paper setting: sensory ratio `0.08`, working ratio `0.22`, long-term frequency threshold `7`, and long-term span threshold `30` days.

## Pipeline

### 1. Preprocess Interactions

```bash
python preprocess/preprocess_mooccubex_multilevel_memory.py \
  --course_file data/MOOCCubeX/raw_data/course_new.json \
  --user_file data/MOOCCubeX/raw_data/user.json \
  --output_dir data/MOOCCubeX/proc_data \
  --sensory_ratio 0.08 \
  --working_ratio 0.22 \
  --long_term_threshold 7 \
  --long_term_min_timespan 30
```

### 2. Generate Candidate Lists and Prompts

```bash
python preprocess/generate_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data
```

Rank uses fixed 50-candidate lists with up to five positive courses. Rerank uses fixed 10-candidate lists with up to four positive courses. Memory and user-side knowledge for each sample are constructed only from interactions before the sample cutoff.

### 3. Generate Memory Reflection Text

The paper uses Llama-3.1-8B-Instruct deployed locally with Ollama. The script accepts any compatible local generation endpoint.

```bash
python knowledge-generating/llm_generating_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data \
  --output_dir data/MOOCCubeX/knowledge_multilevel_memory
```

### 4. Encode Reflection Text

Generated user profiles, course attributes, and MTR reflections are encoded offline. The paper uses BERT-base-uncased with maximum length 512 and mean pooling, producing 768-dimensional vectors.

```bash
python knowledge_encoding/encode_analysis_bert.py \
  --knowledge_dir data/MOOCCubeX/knowledge_multilevel_memory \
  --output_path data/MOOCCubeX/proc_data/bert_newprompt.analysis \
  --model_path bert-base-uncased
```

### 5. Train Rank and Rerank Models

MemRec appends a 64-dimensional memory-guided augmentation vector to the original recommendation features and optimizes the augmentation module jointly with the backbone. Offline memory separation, LLM generation, and BERT encoding do not receive gradients.

```bash
python RS/rank/main_rank_multilevel_memory.py \
  --data_dir data/MOOCCubeX/proc_data \
  --task rerank \
  --algo DeepFM \
  --augment true \
  --aug_prefix bert_newprompt \
  --convert_type MultilevelMemoryHEA \
  --reflection_mode
```

```bash
python RS/rerank/main_rerank_multilevel_memory.py \
  --data_dir data/MOOCCubeX/proc_data \
  --task rerank \
  --algo DLCM \
  --augment true \
  --aug_prefix bert_newprompt \
  --convert_type MultilevelMemoryHEA \
  --reflection_mode
```

## Reproducibility Settings

The reported experiments use fixed preprocessing, candidate lists, and negative samples across all compared methods.

- Random seed: `1234` for model training; `12345` for candidate generation and prompt construction.
- Number of runs: one fixed-seed run per configuration unless otherwise stated.
- Batch size: `512`.
- Training epochs: `20`.
- Early stopping: not enabled in the released scripts; models are trained with the fixed epoch budget above.
- Optimizer: AdamW.
- Learning rates: `1e-3` for the recommendation backbone and `5e-4` for the MemRec expert network.
- Memory parameters: default `w_s=0.08`, `w_w=0.22`, `tau=7`, and `T_min=30` days.
- Candidate lists: Rank uses 50 candidates with up to 5 positives; Rerank uses 10 candidates with up to 4 positives.
- Hyperparameter ranges considered in the paper: sensory window `w_s` from `0.04` to `0.16`, long-term frequency threshold `tau` in `{3, 7, 10, 15}`, long-term span threshold including `15` and `30` days, and MoE expert configurations `(1,1)`, `(1,2)`, `(2,3)`, `(3,3)`, `(2,5)`, and `(4,4)`.
- Text encoder and LLM robustness checks: BERT-base-uncased, BGE-M3, and Sentence-BERT all-MiniLM-L6-v2 for encoding; Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Mistral-7B-Instruct for generation.
- Computing infrastructure: one NVIDIA V100 GPU.

## Method Components

MemRec contains three main technical components:

- Multi-level memory separation maps interaction histories to sensory, working, and long-term memory according to recency, current-session concentration, domain frequency, and time-span persistence.
- Memory Transition Reflection asks the LLM to reason about attention selection from sensory to working memory, consolidation from working to long-term memory, and retrieval influence from long-term expertise to new learning.
- FiLM-conditioned semantic modulation and memory-aware MoE routing transform the user, course, and MTR vectors into the final augmentation vector used by Rank and Rerank models.

Large generated files, raw datasets, checkpoints, logs, and figures are intentionally kept outside the repository.
