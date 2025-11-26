# MemRec: 基于多级记忆增强的推荐系统

MemRec 是一个基于 Atkinson-Shiffrin 记忆模型的多级记忆增强推荐系统，通过模拟人类记忆的三个层次（感觉记忆、工作记忆、长期记忆）来提升推荐效果。系统将认知心理学中的记忆模型引入推荐系统，通过分析用户行为在不同记忆层次上的表现，构建更精准的用户画像和物品表示，从而提升推荐系统的准确性和个性化程度。

## 运行环境

### Python 版本
- Python 3.7+

### 依赖包安装

```bash
pip install torch transformers numpy pandas scikit-learn tqdm
```

## 实验步骤

### 第一步：下载数据集

下载数据集并放置在 `data/` 目录下：

- **Coursera数据集**: https://www.kaggle.com/datasets/leewanhung/coursera-dataset?select=course.csv
  - 需要文件：`enrolled_course.csv`（用户学习历史）和 `Coursera_2.csv`（课程元数据）
  - 放置位置：`data/coursera/`

- **MOOC数据集**: https://github.com/THU-KEG/MOOCCubeX
  - 需要文件：`course.json`（课程元数据）和 `user.json`（用户数据）
  - 放置位置：`data/mooc/`

### 第二步：数据预处理

对原始数据进行预处理，包括K-core过滤、多级记忆提取、负采样等操作，生成训练和测试所需的数据文件。

**Coursera数据集**（在 preprocess 文件夹下运行）：
```bash
python coursera.py
python generate.py --dataset coursera
```

**MOOC数据集**（在 preprocess 文件夹下运行）：
```bash
python mooc.py
python generate.py --dataset mooc
```

### 第三步：知识生成

本项目已包含大模型生成的知识文件（`.klg`文件），无需运行大模型生成代码。每个数据集对应三个知识文件，位于 `data/{dataset}/knowledge_multilevel_memory/` 目录下：
- `item.klg`: 物品多级记忆知识
- `user.klg`: 用户历史多级记忆知识
- `memory_analysis.klg`: 多级记忆分析知识

这些知识文件通过大语言模型生成，包含了物品和用户在多级记忆层次上的语义表示和分析结果。

### 第四步：知识编码

使用BERT等预训练模型将多级记忆知识编码为向量表示，为后续的推荐模型提供知识增强特征。

**在 encoding 文件夹下运行**：
```bash
python encoder.py --dataset coursera
# 或
python encoder.py --dataset mooc
```

### 第五步：模型训练

#### Rank阶段（粗排）

从大量候选中筛选Top-K候选，使用特征交互模型或用户行为模型进行粗排。

**在 training 文件夹下运行**：
```bash
python rank_run.py --dataset coursera
# 或
python rank_run.py --dataset mooc
```

#### Rerank阶段（精排）

对Rank阶段的候选进行精细排序，使用列表级排序模型优化最终推荐结果。

**在 training 文件夹下运行**：
```bash
python rerank_run.py --dataset coursera
# 或
python rerank_run.py --dataset mooc
```

训练脚本会自动搜索最优超参数，并保存最佳模型。

## 数据集来源

- **Coursera数据集**: https://www.kaggle.com/datasets/leewanhung/coursera-dataset?select=course.csv
- **MOOC数据集**: https://github.com/THU-KEG/MOOCCubeX
