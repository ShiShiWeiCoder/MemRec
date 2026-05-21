# MemRec: Anonymous Key Code Release

This repository contains an anonymized, partial code release for a multi-level memory augmented recommendation framework on MOOC-style course recommendation data.

The release is prepared for anonymous review. It intentionally keeps only the core pipeline files and redacts selected implementation details. Raw datasets, generated knowledge files, checkpoints, logs, figures, local scripts, and experiment-only utilities are not included.

## Scope

Included paths:

```text
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

Some core routines are represented by short placeholders to protect unpublished implementation details while preserving the module boundaries, command-line interfaces, and data flow.

## Requirements

The code uses Python 3.8+ with common scientific and deep learning packages:

```bash
pip install torch numpy scikit-learn tqdm transformers requests
```

## Data Layout

Datasets are not included. After obtaining data under the relevant license, place raw files under:

```text
data/MOOCCubeX/raw_data/
```

The preprocessing stage writes processed files under:

```text
data/MOOCCubeX/proc_data/
```

Expected processed artifacts include interaction sequences, item attributes, train/test candidate lists, prompt files, transition features, and multi-level memory metadata.

## Pipeline

### 1. Preprocess Interactions

```bash
python preprocess/preprocess_mooccubex_multilevel_memory.py \
  --course_file data/MOOCCubeX/raw_data/course_new.json \
  --user_file data/MOOCCubeX/raw_data/user.json \
  --output_dir data/MOOCCubeX/proc_data
```

### 2. Generate Candidate Lists and Prompts

```bash
python preprocess/generate_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data
```

### 3. Generate Memory Reflection Text

The LLM script uses a local OpenAI-compatible or Ollama-style endpoint. Configure the endpoint through environment variables or command-line arguments.

```bash
python knowledge-generating/llm_generating_mooccubex_multilevel_memory.py \
  --proc_dir data/MOOCCubeX/proc_data \
  --output_dir data/MOOCCubeX/knowledge_multilevel_memory
```

### 4. Encode Reflection Text

```bash
python knowledge_encoding/encode_analysis_bert.py \
  --knowledge_dir data/MOOCCubeX/knowledge_multilevel_memory \
  --output_path data/MOOCCubeX/proc_data/bert_newprompt.analysis \
  --model_path bert-base-uncased
```

### 5. Train Rank and Rerank Models

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

## Notes for Reviewers

This is a code-structure release rather than a full artifact package. It omits large generated files and selected unpublished details. The public files are intended to show the preprocessing, memory construction, reflection generation, encoding, and Rank/Rerank integration flow without revealing all implementation details.
