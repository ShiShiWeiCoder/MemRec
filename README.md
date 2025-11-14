# MemRec: 基于多级记忆增强的推荐系统

MemRec 是一个基于 Atkinson-Shiffrin 记忆模型的多级记忆增强推荐系统，通过模拟人类记忆的三个层次（感觉记忆、工作记忆、长期记忆）来提升推荐效果。

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [工作流程](#工作流程)
- [数据格式](#数据格式)
- [模型支持](#模型支持)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [依赖环境](#依赖环境)
- [许可证](#许可证)

## 🎯 项目简介

MemRec 将认知心理学中的 Atkinson-Shiffrin 记忆模型引入推荐系统，通过分析用户行为在不同记忆层次上的表现，构建更精准的用户画像和物品表示。系统支持 Coursera 和 MOOC 两个教育推荐数据集，实现了从数据预处理到模型训练的全流程。

### 多级记忆模型

- **感觉记忆 (Sensory Memory)**: 捕捉用户的即时需求和最近浏览行为
- **工作记忆 (Working Memory)**: 分析用户当前学习会话和短期学习目标
- **长期记忆 (Long-term Memory)**: 识别用户的职业发展方向和长期兴趣偏好

## ✨ 核心特性

- 🧠 **多级记忆建模**: 基于 Atkinson-Shiffrin 记忆模型的三级记忆层次分析
- 📚 **多数据集支持**: 支持 Coursera 和 MOOC 教育推荐数据集
- 🔄 **完整工作流**: 数据预处理 → 知识编码 → Rank 粗排 → Rerank 精排
- 🤖 **LLM 知识增强**: 使用大语言模型生成多级记忆增强的知识表示
- 🎯 **多模型支持**: 支持多种推荐模型（DeepFM, DIN, DIEN, DLCM, PRM 等）
- 🔧 **灵活配置**: 丰富的超参数配置选项，支持知识降维、多头注意力融合等

## 📁 项目结构

```
pythonProject3/
├── knowledge_encoding/          # 知识编码模块
│   └── llm_encoding_multilevel_memory.py    # 使用BERT等模型编码多级记忆知识
├── preprocess/                  # 数据预处理模块
│   ├── preprocess_coursera_multilevel_memory.py    # Coursera数据集预处理
│   ├── preprocess_mooc_multilevel_memory.py        # MOOC数据集预处理
│   ├── generate_coursera_multilevel_memory.py      # 生成Coursera多级记忆提示词
│   └── generate_mooc_multilevel_memory.py          # 生成MOOC多级记忆提示词
├── RS/                          # 推荐系统模块
│   ├── rank/                    # Rank阶段（粗排）
│   │   ├── main_rank_multilevel_memory.py          # Rank模型训练主程序
│   │   └── run_rank_multilevel_memory.py           # Rank模型运行脚本
│   └── rerank/                  # Rerank阶段（精排）
│       ├── main_rerank_multilevel_memory.py        # Rerank模型训练主程序
│       └── run_rerank_multilevel_memory.py         # Rerank模型运行脚本
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers numpy pandas scikit-learn tqdm
```

### 2. 数据准备

将数据集放置在 `data/` 目录下：

```
data/
├── coursera/
│   ├── enrolled_course.csv      # 用户学习历史
│   └── Coursera_2.csv           # 课程元数据
└── mooc/
    ├── course_new.json          # 课程元数据
    └── user_new.json            # 用户数据
```

### 3. 数据预处理

#### Coursera 数据集

```bash
cd preprocess
python preprocess_coursera_multilevel_memory.py \
    --k_core_user 3 \
    --k_core_item 3 \
    --sensory_memory_len 1 \
    --working_memory_len 2 \
    --test_ratio 0.1

python generate_coursera_multilevel_memory.py
```

#### MOOC 数据集

```bash
cd preprocess
python preprocess_mooc_multilevel_memory.py
python generate_mooc_multilevel_memory.py
```

### 4. 知识编码

使用 BERT 等模型编码多级记忆增强的知识：

```bash
cd knowledge_encoding
python llm_encoding_multilevel_memory.py
```

主要参数：
- `MODEL_NAME`: 模型名称（如 'bert-base-uncased', 'bert-chinese' 等）
- `AGGREGATE_TYPE`: 聚合方式（'avg', 'last', 'cls' 等）
- `BATCH_SIZE`: 批次大小

### 5. 模型训练

#### Rank 阶段（粗排）

```bash
cd RS/rank
python main_rank_multilevel_memory.py \
    --data_dir data/mooc/proc_data/ \
    --algo DeepFM \
    --augment true \
    --aug_prefix bert-base-uncased_avg_augment_multilevel_memory \
    --memory_mode true \
    --epoch_num 20 \
    --batch_size 512 \
    --lr 1e-4
```

#### Rerank 阶段（精排）

```bash
cd RS/rerank
python main_rerank_multilevel_memory.py \
    --data_dir data/mooc/proc_data/ \
    --algo DLCM \
    --augment true \
    --aug_prefix bert-base-uncased_avg_augment_multilevel_memory \
    --memory_mode true \
    --epoch_num 20 \
    --batch_size 512 \
    --lr 1e-4
```

## 🔄 工作流程

MemRec 的完整工作流程包括以下步骤：

```
1. 数据预处理
   ├── 加载原始数据（Coursera/MOOC）
   ├── K-core 过滤
   ├── 提取多级记忆（感觉记忆、工作记忆、长期记忆）
   ├── 负采样
   ├── ID 映射
   └── 训练/测试集划分

2. 生成多级记忆提示词
   ├── 生成物品多级记忆提示词
   ├── 生成用户历史多级记忆提示词
   └── 生成多级记忆分析提示词

3. 知识编码
   ├── 使用 LLM（如 GPT）生成多级记忆增强知识
   ├── 使用 BERT 等模型编码知识为向量
   └── 保存编码后的向量文件

4. Rank 阶段（粗排）
   ├── 从大量候选中筛选 Top-K（如 15 个）
   └── 使用 CTR 模型（DeepFM, DIN, DIEN 等）

5. Rerank 阶段（精排）
   ├── 对 Rank 阶段的候选进行精细排序
   └── 使用排序模型（DLCM, PRM, SetRank 等）
```

## 📊 数据格式

### 预处理后的数据文件

预处理完成后，`proc_data/` 目录下会生成以下文件：

- `sequential_data.json`: 用户序列数据
- `item2attributes.json`: 物品属性映射
- `datamaps.json`: ID 映射关系
- `train_test_split.json`: 训练/测试集划分
- `multilevel_memory.json`: 多级记忆数据
- `stat.json`: 数据统计信息

### 知识编码后的文件

- `{model}_{agg}_augment_multilevel_memory.item`: 物品知识向量
- `{model}_{agg}_augment_multilevel_memory.hist`: 用户历史知识向量
- `{model}_{agg}_augment_multilevel_memory.analysis`: 多级记忆分析向量

### 训练数据文件

- `ctr.train/test`: CTR 训练/测试数据
- `rank.train/test`: Rank 训练/测试数据（粗排）
- `rerank.train/test`: Rerank 训练/测试数据（精排）

## 🎯 模型支持

### Rank 阶段模型（粗排）

- **特征交互模型**: DeepFM, xDeepFM, DCN, FiBiNet, FiGNN, AutoInt
- **用户行为模型**: DIN (Deep Interest Network), DIEN (Deep Interest Evolution Network)

### Rerank 阶段模型（精排）

- **列表级排序模型**: DLCM, PRM, SetRank, MIR, GSF, EGRerank, LambdaRank

### 知识编码模型

- **BERT 系列**: bert-base-uncased, bert-base-cased, bert-large-uncased
- **中文 BERT**: bert-base-chinese, bert-large-chinese, roberta-chinese, macbert
- **多语言 BERT**: bert-base-multilingual-cased

## ⚙️ 配置说明

### 多级记忆参数

- `--memory_mode`: 是否启用多级记忆增强模式（true/false）
- `--memory_specific_export_num`: 记忆特定专家数量（默认 3）
- `--memory_fusion_type`: 多级记忆融合类型（'attention'/'mlp'）
- `--enable_memory_attention`: 是否启用多头注意力融合（true/false）
- `--memory_attn_heads`: 多头注意力头数（默认 4）

### 知识降维参数

- `--enable_knowledge_reduction`: 是否启用知识降维（true/false）
- `--knowledge_reduction_dim`: 降维目标维度（默认 128）
- `--knowledge_reduction_dropout`: 降维层 dropout 率（默认 0.3）

### 训练参数

- `--epoch_num`: 训练轮数（默认 20）
- `--batch_size`: 批次大小（默认 512）
- `--lr`: 学习率（默认 1e-4）
- `--patience`: 早停耐心值（默认 5）
- `--metric_scope`: 评估指标范围（如 '3,5,10'）

### 数据集特定参数

#### Coursera
- `--k_core_user`: 用户 K-core 阈值（默认 3）
- `--k_core_item`: 物品 K-core 阈值（默认 3）
- `--sensory_memory_len`: 感觉记忆长度（默认 1）
- `--working_memory_len`: 工作记忆长度（默认 2）

#### MOOC
- `--user_core`: 用户 K-core 阈值
- `--item_core`: 物品 K-core 阈值

## 📈 实验结果

系统支持多种评估指标：

- **MAP@K**: 平均精度均值
- **NDCG@K**: 归一化折损累积增益
- **HR@K**: 命中率
- **MRR**: 平均倒数排名
- **AUC**: ROC 曲线下面积（Rank 阶段）

## 🔧 依赖环境

### Python 版本
- Python 3.7+

### 主要依赖
- PyTorch >= 1.8.0
- transformers >= 4.0.0
- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0

### 设备支持
- CPU
- CUDA GPU
- Apple Silicon (MPS)

## 📝 使用示例

### 完整流程示例（MOOC 数据集）

```bash
# 1. 数据预处理
cd preprocess
python preprocess_mooc_multilevel_memory.py
python generate_mooc_multilevel_memory.py

# 2. 知识编码
cd ../knowledge_encoding
python llm_encoding_multilevel_memory.py

# 3. Rank 训练
cd ../RS/rank
python main_rank_multilevel_memory.py \
    --data_dir ../../data/mooc/proc_data/ \
    --algo DeepFM \
    --augment true \
    --memory_mode true

# 4. Rerank 训练
cd ../rerank
python main_rerank_multilevel_memory.py \
    --data_dir ../../data/mooc/proc_data/ \
    --algo DLCM \
    --augment true \
    --memory_mode true
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**: 使用 LLM 生成知识时，需要配置相应的 API 密钥。知识编码阶段会从 Hugging Face 下载预训练模型，请确保网络连接正常。
