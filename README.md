# Zero2Hero

一个面向「从 0 到 1 学习大模型」的实践仓库，覆盖：

- LLM 基础实现与训练（`minimind`、`TinyLLM`）
- VLM 多模态实践（`minimind-v`、`Qwen3-SmVL`）
- RL 对齐损失家族（`GRPO-Family`）
- Transformer 基础组件拆解（`base`）
- 研究/实验性项目（`MMSearch`、`nanoMOE`、`toys`）

> 推荐配合视频学习：[Build a nanoGPT](https://www.bilibili.com/video/BV1qWwke5E3K?spm_id_from=333.788.videopod.sections&vd_source=4b89af53720f562b658eda17f36f478f)

---

## 1. 仓库定位

`Zero2Hero` 不是单一项目，而是一组互补的学习/实验工程：

- 想学 **LLM 训练全流程**：优先看 `minimind`
- 想学 **VLM 多模态拼接与训练**：看 `Qwen3-SmVL`、`minimind-v`
- 想学 **RLHF/GRPO 类损失**：看 `GRPO-Family`
- 想学 **Attention/MoE/RoPE 基础实现**：看 `base`

---

## 2. 环境要求

- Python: `>=3.11`
- 深度学习框架：`torch==2.4.1`
- 主要库：`transformers==4.54.1`、`trl>=0.15.2`、`datasets>=3.4.1`

依赖定义位于根目录 `pyproject.toml`。

### 安装方式（推荐）

```bash
# 进入仓库根目录
cd Zero2Hero

# 若使用 uv
uv sync

# 或使用 pip
pip install -e .
```

---

## 3. 项目结构总览

```text
Zero2Hero/
├── base/                 # Transformer基础组件实现（交叉熵/MHA+KVCache/MoE/RoPE）
├── GRPO-Family/          # GRPO、DAPO、CISPO、GSPO、GiGPO 损失Notebook
├── minimind/             # 迷你LLM完整链路：预训练/SFT/DPO/LoRA/推理
├── minimind-v/           # 迷你VLM项目（README已给背景，代码仍在迭代）
├── Qwen3-SmVL/           # SmolVLM + Qwen3 的中文多模态“拼接微调”
├── tiny-universe/        # Datawhale TinyLLM 课程工程
├── MMSearch/             # MMSearch 论文/基准相关
├── nanoMOE/              # MoE构建实验Notebook
└── toys/                 # 研究/实验性质内容（如 mHC）
```

---

## 4. 快速开始（建议顺序）

### Step 1：先跑通 `minimind`（最完整）

```bash
cd minimind

# 训练 tokenizer（脚本在 scripts 目录）
python scripts/train_tokenizer.py

# 预训练（多卡示例）
bash scripts/pretrain.sh

# 全参数 SFT（多卡示例）
bash scripts/full_sft.sh

# 推理评估
python eval_model.py --load 0 --model_mode 1
```

> `scripts/pretrain_moe.sh`、`scripts/full_sft_moe.sh` 可用于 MoE 版本。

### Step 2：阅读 `base` 理解核心模块

- `base/mha/mha_kvcache.py`：多头注意力 + KV Cache
- `base/moe/moe.py`：MoE + 负载均衡辅助损失示例
- `base/rpoe/rope.py`：RoPE 旋转位置编码实现
- `base/cross_entropy/cross_entropy.py`：二分类交叉熵示例

### Step 3：扩展到 VLM / RL

- VLM：`Qwen3-SmVL`
- RL 损失：`GRPO-Family`

---

## 5. 各子项目说明

### 5.1 minimind

来源：`https://github.com/jingyaogong/minimind`

核心特点：

- 迷你 LLM 的完整训练链路：预训练、SFT、DPO、LoRA
- 内置 tokenizer：`model/minimind_tokenizer/`
- 支持 MoE（通过 `--use_moe True`）
- 推理入口：`eval_model.py`

关键文件：

- `train_pretrain.py`
- `train_full_sft.py`
- `train_dpo.py`
- `train_lora.py`
- `eval_model.py`

### 5.2 minimind-v

来源：`https://github.com/jingyaogong/minimind-v`

定位：迷你多模态视觉语言模型（VLM）项目，包含预训练与模型结构代码。

当前目录中可见：

- `train_pretrain.py`
- `model/model_vlm.py`
- `model/VLMConfig.py`
- `model/dataset.py`

说明：`eval_vlm.py` 当前为空文件，评估脚本仍待补充。

### 5.3 Qwen3-SmVL

来源：`https://github.com/ShaohonChen/Qwen3-SmVL`

定位：将 `SmolVLM2-256M` 与 `Qwen3-0.6B` 做连接器拼接，进行中文多模态训练。

关键文件：

- `train.py`：训练主入口（支持 `yaml` 参数）
- `utils.py`：加载并替换 processor / connector / text model
- `cocoqa_train.yaml`：训练参数示例
- `chat_template.jinja`：对话模板

示例启动：

```bash
cd Qwen3-SmVL
python train.py cocoqa_train.yaml
```

### 5.4 GRPO-Family

包含 5 个强化学习损失 Notebook：

- `grpo/grpo_loss.ipynb`
- `dapo/dapo_loss.ipynb`
- `cispo/cispo_loss.ipynb`
- `gspo/gspo_loss.ipynb`
- `gigpo/gigpo_loss.ipynb`

另有：`grpo/grpo_analysis.ipynb`。

适合用途：理解不同策略优化目标与实现差异。

### 5.5 tiny-universe / TinyLLM

来源：`https://github.com/datawhalechina/tiny-universe`

`tiny-universe/TinyLLM/README.md` 给出了最短路径：

```bash
python train_vocab.py --download True --vocab_size 4096
python preprocess.py
python train.py
python sample.py --prompt "One day, Lily met a Shoggoth"
```

适合场景：快速体验从 tokenizer 到采样推理的完整闭环。

### 5.6 MMSearch

定位：MMSearch 论文相关基准内容（README 当前主要为论文引用）。

### 5.7 nanoMOE

定位：MoE 相关实验 Notebook（`build_moe.ipynb`）。

### 5.8 toys

包含实验性内容（如 `toys/mHC/mHC.ipynb`），用于跟踪/验证新想法。

---

## 6. 推荐学习路径

### 路线 A（LLM 基础）

1. `base`（理解机制）
2. `minimind`（跑训练和推理）
3. `tiny-universe/TinyLLM`（对比不同最小实现）

### 路线 B（对齐与强化学习）

1. `minimind`（先有可训练语言模型）
2. `GRPO-Family`（理解各损失）
3. 回到 `minimind/train_dpo.py` 做联动实验

### 路线 C（多模态）

1. `minimind-v`（了解轻量 VLM 结构）
2. `Qwen3-SmVL`（实践模型拼接与多模态训练）

---

## 7. 注意事项

- 多个脚本默认数据路径是作者本地路径（如 `/home/zxy/...`），请按你的环境修改。
- 多卡脚本依赖 `torchrun` 和 CUDA 环境；单卡可直接运行 Python 入口并下调 batch size。
- 部分子目录当前是学习/占位性质（例如 README 简短、Notebook 为主），建议以代码与脚本为准。

---

## 8. 致谢与来源

- `minimind`: https://github.com/jingyaogong/minimind
- `minimind-v`: https://github.com/jingyaogong/minimind-v
- `Qwen3-SmVL`: https://github.com/ShaohonChen/Qwen3-SmVL
- `tiny-universe`: https://github.com/datawhalechina/tiny-universe
- `MMSearch`: https://github.com/CaraJ7/MMSearch