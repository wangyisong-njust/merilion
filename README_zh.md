# MERaLiON-2 模型压缩与推理优化

本项目对 [MERaLiON-2](https://huggingface.co/MERaLiON) 多模态语音识别（ASR）大模型进行结构化剪枝、量化和推理优化，目标是在保持识别准确率的同时，大幅降低模型参数量和推理延迟，使其能够在 CPU 等资源受限的边缘设备上高效部署。

---

## 模型架构

### MERaLiON-2 基础架构

MERaLiON-2 是由新加坡国立研究基金会支持开发的多模态大语言模型，专注于英语语音识别任务。其架构由三个核心组件构成：

```
音频输入
   ↓
Speech Encoder（Whisper 架构）
   ↓
Audio Adapter（线性投影层）
   ↓
Text Decoder（Gemma2 架构）
   ↓
转录文本输出
```

**Speech Encoder**
- 基于 OpenAI Whisper 的 Transformer 编码器
- 将原始音频特征（梅尔频谱）映射为音频嵌入向量
- 以 30 秒为单位分块处理长音频
- 推理过程保持 FP32/FP16 精度，不参与量化

**Audio Adapter**
- 将音频嵌入对齐到文本解码器的维度空间
- 轻量线性层，参数量少

**Text Decoder（MERaLiON-2-3B）**
- 基于 Gemma2 的 Decoder-only Transformer
- 默认配置（未剪枝时）：
  - 层数：26 层
  - 隐层维度：`hidden_size = 2048`
  - 注意力头数：`num_attention_heads = 8`（GQA，`num_key_value_heads = 4`）
  - FFN 中间维度：`intermediate_size = 8192`
  - 激活函数：GeGLU（门控线性单元）

---

## 压缩方案与技术架构

### 1. 结构化剪枝（LLM-Pruner）

本项目基于 [LLM-Pruner](https://arxiv.org/abs/2305.11627) 框架对 MERaLiON-2 文本解码器进行结构化剪枝。

#### 剪枝框架

剪枝分三个阶段：

**阶段一：发现阶段（Discovery Stage）**
- 通过依赖图（`DependencyGraph`）自动分析层间耦合关系
- 识别最小可移除单元（Group），确保剪枝后模型结构合法
- 注意力头剪枝必须以完整头为单位（Q/K/V 联动）

**阶段二：估计阶段（Estimation Stage）**
- 使用 **Taylor 重要性估计**衡量每个 Group 对模型性能的贡献
- 通过少量标定样本（约 20 条音频）计算一阶梯度和近似二阶海森矩阵
- 选项：`param_mix`（混合一二阶，默认）、`param_first`、`l1`、`l2`

**阶段三：恢复阶段（Recover Stage）**
- 剪枝后通过 LoRA 微调恢复模型性能（见下节）

#### 中间块（Midblock）剪枝策略

仅对 Transformer 的中间层施加剪枝，保留首尾层的完整表达能力：

```
Layer 0 ~ start-1     ← 保持完整（输入特征提取）
Layer start ~ end-1   ← 按 pruning_ratio 剪枝（中间块）
Layer end ~ N-1       ← 保持完整（输出层）
```

典型配置示例（25% 剪枝，3B 模型 26 层）：
- `--block_mlp_layer_start 3 --block_mlp_layer_end 23`
- `--block_attention_layer_start 3 --block_attention_layer_end 23`

剪枝后，各层维度不再统一（非均匀剪枝），通过自定义模型代码（`meralion2_bl/`）使用 `DynamicCache` 代替原始 `HybridCache` 支持动态 KV 缓存。

#### 剪枝脚本

```bash
python meralion.py \
  --base_model MERaLiON/MERaLiON-2-3B \
  --pruning_ratio 0.25 \
  --block_wise \
  --block_mlp_layer_start 3 --block_mlp_layer_end 23 \
  --block_attention_layer_start 3 --block_attention_layer_end 23 \
  --pruner_type taylor \
  --num_examples 20 \
  --device cuda \
  --save_model \
  --save_ckpt_log_name meralion_pruned
```

---

### 2. LoRA 微调恢复（Post-Training）

剪枝后使用低秩适配（LoRA）快速恢复模型性能：

- 仅对文本解码器（Gemma2 部分）的线性层插入 LoRA 适配器
- 数据集：IMDA 英语 ASR 数据（Part 1）及混合语料
- 支持多 GPU 训练（DeepSpeed / gradient accumulation）

```bash
python post_training_meralion.py \
  --prune_model meralion_checkpoints/meralion_pruned \
  --data_path imda_asr \
  --lora_r 16 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --output_dir meralion_tune_log/my_tune
```

微调完成后合并 LoRA 权重：

```bash
python merge_meralion.py \
  --ckpt meralion_checkpoints/meralion_pruned \
  --lora_ckpt meralion_tune_log/my_tune/checkpoint-final \
  --save_path merged_model
```

---

### 3. 量化方案

针对 CPU 推理，项目实现了多种量化方案（`infer_cpu.py`），可按精度/速度权衡选择：

| 方案 | 精度 | 加速比 | 硬件需求 | 命令行标志 |
|------|------|--------|----------|------------|
| 无量化 | FP32 | 1× | 任意 | （默认） |
| INT8 动态量化 | 权重 INT8，激活 FP32 | ~1.5–2× | 任意 | `--int8` |
| INT8 权重仅（torchao） | 权重 INT8，激活 FP32 | ~1.3–1.8× | 任意，torch.compile 兼容 | `--int8ao` |
| **W8A8**（INT8 × INT8） | 权重+激活均 INT8 | ~2–3× | AVX-512 VNNI / AMX | `--w8a8` |
| INT4 权重仅 | 权重 INT4，激活 FP32 | ~2–4× | 任意（实验性） | `--int4` |

**W8A8 量化**是本项目的重点优化，利用现代 x86 CPU 的 VNNI/AMX 指令集实现真正的 INT8 矩阵乘法：

- 仅量化 `text_decoder.model`（Gemma2 Transformer 块）的线性层权重和运行时激活
- `speech_encoder`、`audio_adapter` 和 `lm_head` 保持 FP32，避免精度损失
- 使用 [torchao](https://github.com/pytorch/ao) 的 `Int8DynamicActivationInt8WeightConfig` 实现

---

### 4. torch.compile 推理加速

通过 `torch.compile` 对推理计算图进行编译优化：

- **`max-autotune` 模式**（默认）：针对 CPU GEMM 内核做自动调优，适合 INT8/FP32 推理
- **`reduce-overhead` 模式**：减少 Python 调度开销，适合较小批量

```bash
# 启用 compile，指定模式
python infer_cpu.py --model merged_model --w8a8 --compile --compile_mode max-autotune
python infer_cpu.py --model merged_model --int8ao --compile --compile_mode reduce-overhead
```

注意：INT4 和 W8A8 量化的核心计算由 torchao 内核处理，不依赖 compile 也可加速；但二者组合使用可进一步提升吞吐量。

---

### 5. .mera 打包格式（边缘分发）

为方便在边缘设备上分发和加载模型，项目实现了 `.mera` 单文件打包格式（`pack_model.py`）：

**格式规范（MERA v1）**

```
Bytes  0–3:    Magic: b"MERA"
Bytes  4–7:    版本号: uint32 LE = 1
Bytes  8–15:   Header 长度: uint64 LE（含对齐填充）
Bytes 16–N:    Header JSON（UTF-8），零填充到 64 字节对齐边界
Bytes  N+:     张量数据块，每块对齐到 64 字节
```

**Header JSON 内容**

```json
{
  "format_version": 1,
  "model_config": { ... },           // config.json 内容
  "configs": {                        // 辅助配置（tokenizer 等）
    "tokenizer.json": { ... },
    "processor_config.json": { ... }
  },
  "source_files": {                   // 模型 Python 代码（如 processor）
    "processing_meralion2.py": "..."
  },
  "storage": "int8",                  // 或 "float16"
  "tensors": {                        // 张量索引表
    "text_decoder.model.layers.0.mlp.gate_proj.weight": {
      "dtype": "int8", "shape": [1024, 2048],
      "offset": 1234, "nbytes": 2097152
    },
    "text_decoder.model.layers.0.mlp.gate_proj.weight_scale": {
      "dtype": "float32", "shape": [1024],
      "offset": 3331586, "nbytes": 4096
    }
  }
}
```

**量化策略**
- `text_decoder.model.layers.*.*.weight`：INT8 每输出通道对称量化 + FP32 scale
- 其他所有张量（speech encoder、audio adapter 等）：FP16

打包和加载：

```bash
# 打包
python pack_model.py --model merged_model --output model.mera

# 加载推理
python infer_cpu.py --model model.mera --w8a8 --compile --audio sample.wav
```

---

## 项目结构

```
merilion/
├── LLMPruner/                    # 核心剪枝框架库
│   ├── torch_pruning/            # 依赖图、重要性估计、剪枝算法
│   ├── models/                   # 各 LLM 架构适配（LLaMA、ChatGLM 等）
│   ├── pruner/                   # MERaLiON 专用剪枝器
│   ├── evaluator/                # PPL 评估模块
│   └── peft/                     # LoRA 实现
├── meralion2_bl/                 # 修改后的 HuggingFace 模型代码
│   ├── modeling_gemma2.py        # 支持 DynamicCache 的 Gemma2 实现
│   └── modeling_whisper.py       # Whisper 音频编码器
├── scripts/                      # 执行脚本
│   └── meralion.sh               # MERaLiON 剪枝 + 微调流程
├── audiobench/                   # ASR 评估数据集处理
├── vllm_inference/               # vLLM 推理集成
├── meralion_checkpoints/         # 剪枝模型检查点
├── meralion_tune_log/            # LoRA 微调日志和权重
├── meralion.py                   # MERaLiON 剪枝主脚本
├── post_training_meralion.py     # MERaLiON LoRA 微调
├── merge_meralion.py             # 剪枝模型 + LoRA 合并
├── infer_cpu.py                  # CPU 推理（量化 + compile）
├── pack_model.py                 # .mera 格式打包工具
├── make_demo_html.py             # 交互式 HTML 演示生成
└── run_demo.sh                   # 一键基准测试 + 演示生成
```

---

## 快速上手

### 环境依赖

```bash
pip install torch torchao transformers peft deepspeed
pip install jiwer  # WER 计算
```

### 端到端流程

```bash
# 1. 剪枝（25% 参数，中间块策略）
python meralion.py \
  --base_model MERaLiON/MERaLiON-2-3B \
  --pruning_ratio 0.25 \
  --block_wise \
  --block_mlp_layer_start 3 --block_mlp_layer_end 23 \
  --block_attention_layer_start 3 --block_attention_layer_end 23 \
  --save_model --save_ckpt_log_name meralion_pruned

# 2. LoRA 微调恢复
python post_training_meralion.py \
  --prune_model meralion_checkpoints/meralion_pruned \
  --output_dir meralion_tune_log/my_tune

# 3. 合并权重
python merge_meralion.py \
  --ckpt meralion_checkpoints/meralion_pruned \
  --lora_ckpt meralion_tune_log/my_tune/checkpoint-final \
  --save_path merged_model

# 4. 打包为 .mera 格式
python pack_model.py --model merged_model --output model.mera

# 5. CPU 推理（W8A8 + compile）
python infer_cpu.py \
  --model model.mera \
  --w8a8 --compile \
  --audio sample.wav
```

### 一键演示

```bash
bash run_demo.sh
```

生成包含 Pareto 图和音频样本对比的交互式 HTML 演示页面。

---

## 主要修改记录

| 提交 | 修改内容 |
|------|----------|
| `a4f0b5d` | 新增 W8A8 量化（INT8 权重 × INT8 动态激活），支持 VNNI/AMX CPU 加速 |
| `809ce90` | 实现 `.mera` 单文件打包格式，用于边缘设备分发 |
| `db8381a` | 将 torch.compile 默认模式切换为 `max-autotune`，提升 CPU GEMM 性能 |
| `267c4a9` | 新增 `--compile_mode` 命令行参数，支持运行时选择编译模式 |
| `1f208c7` | 修复缓存类型：原始模型使用 `HybridCache`，剪枝模型使用 `DynamicCache` |
| `6c2ce7c` | 修复 `<bos>` 缺失问题：两条推理路径均使用 `apply_chat_template` |
| `8617a66` | 新增 WER 文本归一化（小写 + 去标点），规范评估标准 |
| `ba8f208` | 修复 `HybridCache` 溢出：预分配 prefill + max_new_tokens 容量 |
| `e8a3622` | 新增 `run_demo.sh`：一键基准测试 + HTML 演示生成 |

---

## 评估指标

推理评估使用 WER（词错误率）和延迟，数据集为 IMDA Part 1 ASR 测试集（indices 11000–15999）。

```bash
# 在数据集上评估，输出 WER + 延迟
python infer_cpu.py \
  --model model.mera \
  --w8a8 --compile \
  --dataset /path/to/imda_test \
  --output wer+latency
```

---

## 致谢

- [LLM-Pruner](https://arxiv.org/abs/2305.11627)（Ma et al., NeurIPS 2023）— 结构化剪枝框架
- [MERaLiON-2](https://huggingface.co/MERaLiON) — 基础模型
- [torchao](https://github.com/pytorch/ao) — INT8/INT4 量化内核
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) — 依赖图剪枝引擎
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 评估工具

---

## 引用

如使用本项目的剪枝方法，请引用：

```bibtex
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```
