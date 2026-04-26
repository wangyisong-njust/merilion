# MERaLiON-2 CPU 推理项目 — 交接文档

## 1. 项目目标

通过 **结构化剪枝 + 量化 + torch.compile** 让 MERaLiON-2-3B 在 CPU 上做 ASR 推理达到可用速度（<5s/句），同时保持 WER 不大幅下降。

当前 3B 模型已跑通并形成 Pareto 前沿，扩展到 MERaLiON-2-10B 和 LLaMA-2-7B 验证方法通用性。

## 2. 完整流水线

```
原始模型 (BF16)
    ↓  Step 1: 结构化剪枝  (meralion.py)
剪枝后模型 (meralion_checkpoints/)
    ↓  Step 2: LoRA 微调恢复  (post_training_meralion.py)
LoRA 权重 (meralion_tune_log/)
    ↓  Step 3: 合并 LoRA  (merge_lora.py)
合并后 FP32 模型
    ↓  Step 4: CPU 量化推理  (infer_cpu.py)
JSON 结果
    ↓  Step 5: HTML Demo  (make_demo_html.py)
demo.html 可视化
```

## 3. 目录结构

| 目录 | 用途 |
|---|---|
| `meralion_checkpoints/` | 剪枝后裸模型（未微调） |
| `meralion_tune_log/` | LoRA adapter + 合并后的完整模型 |
| `meralion2_bl/` | 改写的 Gemma2 HF 模型代码（支持 midblock 参数） |
| `LLMPruner/` | 剪枝算法实现 |
| `prune_log/` | 剪枝过程日志 |

---

## 4. Step 1：结构化剪枝 (`meralion.py`)

按层选择性裁剪 attention heads 和 MLP 中间维度。

### 核心参数

| 参数 | 含义 |
|---|---|
| `--base_model` | 原始模型路径（HF 目录） |
| `--pruning_ratio 0.5` | 默认裁剪率（被 attn/mlp 专用参数覆盖） |
| `--text_attn_pruning_ratio 0.5` | 文本解码器 attention 头裁剪 50% |
| `--text_mlp_pruning_ratio 0.5` | 文本解码器 MLP 中间维度裁剪 50% |
| `--whisper_attn_pruning_ratio` | 语音编码器 attention 裁剪（可选） |
| `--whisper_mlp_pruning_ratio` | 语音编码器 MLP 裁剪（可选） |
| `--block_attention_layer_start/end` | 裁剪的 attention 层范围 `[start, end)` |
| `--block_mlp_layer_start/end` | 裁剪的 MLP 层范围 `[start, end)` |
| `--pruner_type taylor` | 重要性度量（taylor / magnitude / l2） |
| `--taylor param_mix` | Taylor 具体形式（param_first / param_second / param_mix） |
| `--num_examples 20` | 校准集样本数（用于计算重要性） |
| `--block_wise` | 块级裁剪（vs. `--channel_wise` / `--layer_wise`） |
| `--save_model_path` | 裁剪后模型保存路径 |

### 中间层保护机制

不是所有层都被裁，**头部（0 ~ start-1）和尾部（end ~ 最后）保留完整维度**，只裁中间段。这是因为：
- 头部处理低层特征，改动易破坏语义
- 尾部接近输出，改动直接影响 token 生成

裁剪后 `config.json` 里写入：
```json
"midblock_ratio": 0.5,   // 1 - pruning_ratio
"midblock_start": 3,
"midblock_end":   22
```

`meralion2_bl/modeling_gemma2.py` 根据这三个参数动态缩小对应层的权重矩阵。

---

## 5. Step 2：LoRA 微调 (`post_training_meralion.py`)

裁剪后精度会掉，用 LoRA 在 ASR 数据上微调恢复。

### 核心参数

| 参数 | 含义 |
|---|---|
| `--base_model` | 剪枝后模型路径 |
| `--output_dir` | LoRA adapter 输出目录 |
| `--lora_r 16` | LoRA 秩（rank） |
| `--lora_alpha 16` | LoRA 缩放系数（通常与 rank 相同） |
| `--lora_dropout 0.05` | LoRA 层 dropout |
| `--lora_target_modules` | 插 LoRA 的模块（默认全部 attn+mlp proj） |
| `--learning_rate 5e-5` | 学习率（10B 建议用 1e-5） |
| `--num_epochs 3` | 训练轮数（通常 2-3 足够） |
| `--batch_size 8` | 全局 batch size |
| `--micro_batch_size 2` | 每张卡每步的 batch（10B 设为 1） |
| `--cutoff_len 256` | token 最大长度 |

两卡 DDP：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 post_training_meralion.py ...
```

---

## 6. Step 3：合并 LoRA (`merge_lora.py`)

把 LoRA adapter 权重加回基础模型，得到一个可独立使用的完整模型。

```bash
python merge_lora.py \
    --base    meralion_checkpoints/<pruned-model> \
    --adapter meralion_tune_log/<lora-dir> \
    --output  meralion_tune_log/<lora-dir>   # 合并权重直接写回同目录
```

合并后 `model*.safetensors` 和 `adapter_config.json` 共存，`infer_cpu.py` 会优先用合并权重。

---

## 7. Step 4：CPU 推理 + 量化 (`infer_cpu.py`)

核心量化选项（互斥）：

| 选项 | 方法 | 兼容 compile | 速度 | 精度 |
|---|---|---|---|---|
| `--no_quant` | FP32 | ✓ | 慢 | 最好 |
| `(默认)` | INT8 dynamic (`quantize_dynamic`) | ✗ | 中 | 波动大 |
| `--int8ao` | torchao INT8 weight-only | ✓ | 中 | 稳定 |
| `--int4` | torchao INT4 weight-only | ✓ | **最快** | 可接受 |
| `--w8a8` | torchao W8A8（权重+激活都 INT8） | ✓ | 最快 | 实验中 |

### 其他参数

| 参数 | 含义 |
|---|---|
| `--dataset` | IMDA_PART1 ASR 数据集路径 |
| `--num_samples 20` | benchmark 样本数 |
| `--max_new_tokens 128` | 生成长度上限 |
| `--compile_mode` | `max-autotune`（默认）/ `reduce-overhead` / `default` |
| `--no_compile` | 跳过 torch.compile |
| `--trust_remote_code` | 原始模型走原生 HF 路径（**未剪枝模型专用**） |
| `--save_samples` | 在 JSON 中保存每条样本的预测文本 |
| `--audio_dir demo_audio` | 导出原始 wav（供 HTML demo 嵌入） |

### 关键数据路径

- 未裁剪原模型：加 `--trust_remote_code`，走 `transcribe_native()`（手写 prefill/decode 循环 + HybridCache）
- 裁剪后模型：走 `transcribe()`（`model.generate()` + 预建 DynamicCache）

### 结果 JSON 关键字段

```json
{
  "wer": 0.0151,
  "avg_latency_s": 19.88,
  "avg_decode_tps": 1.17,
  "ram_mb": 10500,
  "samples": [...]
}
```

---

## 8. Step 5：HTML Demo (`make_demo_html.py`)

从多个 JSON 生成一页可视化，含：
- 性能总表（延迟、加速比、WER、ΔWER、decode 速度、RAM）
- Pareto 曲线（speedup vs WER）
- 20 条样本（音频 + 下拉菜单切换配置）

```bash
python make_demo_html.py \
    --configs \
        "Original FP32:demo_fp32_original.json" \
        "mid3-22 INT8:demo_int8_mid3-22.json" \
        ... \
    --output demo.html
```

第一个配置自动重命名为 "Original Model"，其余为 "Compression Plan A/B/C/D"。

---

## 9. 配置命名约定

`v3-td50-mid3-22` 解码：
- `v3`：流水线第三版（带 Whisper 编码器可选剪枝 + 2-GPU DDP 微调）
- `td50`：**t**ext **d**ecoder 整体裁剪率 50%
- `mid3-22`：`block_layer_start=3`，`block_layer_end=22`（即裁层 3~21）

---

## 10. 3B 当前结果（20 样本，Pareto 前沿）

| 配置 | 延迟 | 加速比 | WER | Decode 速度 |
|---|---|---|---|---|
| Original FP32 | 19.88s | 1.00× | 1.51% | 1.17 tok/s |
| mid3-22 INT8 | 8.05s | 2.47× | 3.02% | ~2.5 tok/s |
| mid4-23 INT4+compile | 4.11s | 4.83× | 3.77% | ~5 tok/s |
| mid3-23 INT4+compile | 3.99s | 4.98× | 4.15% | ~5 tok/s |
| mid3-22 INT4+compile | 3.92s | 5.07× | 4.53% | ~5 tok/s |

结论：**INT4+compile 全面优于 INT8**（更快、RAM 更低、WER 更稳定）。

---

## 11. 扩展到新模型

已有脚本（同比例缩放层范围）：

| 脚本 | 模型 | 层数 | 配置 |
|---|---|---|---|
| `run_prune_10b.sh` | MERaLiON-2-10B-ASR | 42 | mid5-36 / mid5-37 / mid6-37 |
| `run_prune_llama2_7b.sh` | LLaMA-2-7B | 32 | mid4-27 / mid4-28 / mid5-28 |

缩放公式（保持相同理论裁剪率）：
```
new_start = round(orig_start / orig_layers * new_layers)
new_end   = round(orig_end   / orig_layers * new_layers)
```

### LLaMA-2 注意事项

`meralion.py` 默认访问 `model.model.text_decoder.model.layers[]`（MERaLiON 多模态结构）。LLaMA-2 是纯文本，结构是 `model.model.layers[]`。如果脚本报错，需要：
1. 在 `meralion.py` 里加模型类型判断
2. 或改用原版 LLM-Pruner 的 `hf_prune.py`

`infer_cpu.py` 也是为 MERaLiON（音频+文本）写的，纯文本模型需要单独的 benchmark 脚本。

---

## 12. 常见坑

1. **`<bos>` token 必须存在**：Gemma2 训练时每个 prompt 前都有 `<bos>`。手动拼 prompt 会漏掉 → 生成退化为 `<Speaker1>`。必须用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
2. **HybridCache 默认大小不够**：`model.generate(cache_implementation="hybrid")` 按 `max_new_tokens` 建 cache，prefill 137 token 会溢出。需要预建 `HybridCache(max_cache_len=seq_len + max_new_tokens)`
3. **INT8 dynamic quant 不能与 torch.compile 共存**：`DynamicQuantizedLinear` Dynamo 无法 trace，输出乱码。想要 compile 就用 `--int8ao`
4. **原模型走 `transcribe_native()`，裁剪模型走 `transcribe()`**：因为 HF 原生 Gemma2 代码里的 HybridCache 逻辑会对 DynamicCache 兼容性差；裁剪模型（KV head 数异构）必须用 DynamicCache

---

## 13. 关键运行入口

| 脚本 | 功能 |
|---|---|
| `benchmark_cpu_v3_all.sh` | 3B 全部 v3 配置的完整 CPU benchmark（FP32/INT8/INT4） |
| `run_demo.sh` | 生成 HTML demo（跑 5 个 Pareto 配置并打包） |
| `run_quant_bench_mid3_22.sh` | mid3-22 单配置的 GPU W4A16-AWQ + CPU benchmark |
| `run_v3_mid50_multi.sh` | 8 卡并行跑 4 个 v3 配置的完整 prune+tune+eval |
| `run_prune_10b.sh` | 10B 三配置完整流水线 |
| `run_prune_llama2_7b.sh` | LLaMA-2-7B 三配置完整流水线 |

---

## 14. Speculative Decoding 实验（2026-04）

目标：在 GPU 上用 spec decoding 加速 decode，两条路：
- **N-gram**（`infer_gpu.py --speculative --corpus ngram_corpus.pkl`）：`NGramDraft` 用 train 集 build 的 prefix→next_tok 字典
- **Model-draft**（`infer_gpu_spec_draft_fast.py`）：pruned 3B 做 draft，原模型做 verifier

### 14.1 最终结果（IMDA_PART1, 20 samples, L40 GPU, BF16）

| 验证器 | 方案 | γ | lat (s) | tps | acc | speedup |
|-------|-----|---|---------|-----|-----|---------|
| 3B | no-spec（baseline） | — | 0.54 | 37.1 | — | 1.00× |
| 3B | ngram + corpus | 5 | 0.62 | 37.2 | 9.7% | ≈1.00× |
| 3B | model-draft（3B→3B-pruned） | 5 | 0.94 | 22.2 | 70.3% | **0.58×** ✗ |
| 3B | model-draft + torch.compile | 5 | 1.07 | 20.2 | 70.3% | 0.50× ✗ |
| 3B | ngram + torch.compile | 5 | 78.3 | 0.4 | 9.7% | **broken** |
| **10B-ASR** | no-spec（baseline） | — | 0.87 | 22.9 | — | 1.00× |
| **10B-ASR** | ngram + corpus | 8 | 0.71 | 31.3 | 6.9% | 1.23× ✓ |
| **10B-ASR** | ngram + ref corpus | 16 | 0.69 | 32.5 | 3.9% | 1.26× |
| **10B-ASR** | model-draft（10B→3B-pruned） | 5 | 1.05 | 20.4 | 69.7% | 0.83× ✗ |
| **10B-ASR** | ngram + **model corpus** (10k mid3-23 outputs) | 8 | **0.66** | **34.1** | 8.3% | **1.32× ✓** |
| **3B** | **Medusa v2** (K=4 heads, verifier-output trained) | 4 | **0.35** | **57.3** | 38.8% | **1.51× ✓** |

### 14.2 关键结论

1. **Spec decoding 只在大模型上划算**。3B 时 FA2 + HybridCache 的 baseline 已经打满，任何 overhead 都亏；10B 开始才有余地。
2. **N-gram > model-draft** 在此配置。Ngram 是 dict lookup（几乎零成本），失败就 fallback。Model-draft 的 γ 次 draft forwards 墙钟开销大 —— draft wall-clock 比例远大于 FLOPs 比例（SDPA vs FA2、非均匀 KV heads 的 cache 管理）。
3. **torch.compile 在这个 custom Gemma2 上全线失败**：
   - Ngram + compile：78s/sample（100× 变慢）—— spec 的 K+1 verifier 输入 shape 变化，dynamo recompile 地狱
   - Model-draft + compile：10% 变慢 —— `skipping cudagraphs due to mutated inputs`（KV cache in-place update）+ `cache_size_limit (8)` 被打爆
   - 即使加了 `mark_static_address` + 持久 input buffer 也救不回来

### 14.3 工程清单（全都试过）

全部保留在 `infer_gpu_spec_draft_fast.py`：
- `PrunedPreallocCache`（继承 `HybridCache`，每层按实际 kv_heads 预分配，避免 `DynamicCache` 的 per-step `torch.cat`）
- Draft argmax GPU 驻留，γ 次循环后一次 sync（省 γ−1 次 CPU↔GPU sync）
- EOS 截断移到循环外（多跑几次 draft forward 但保 GPU pipeline 不断）
- 静态形状 cache：update() 返回全 buffer + attention_mask=None（Gemma2Model._update_causal_mask 自动按 max_cache_len 生成静态 mask）
- FA2 on/off 可切
- `--compile_draft` 开关（仍保留，有兴趣可试新 torch 版本）

### 14.4 N-gram corpus 优化

N-gram 的接受率主要受 **corpus 与 verifier 实际输出分布的匹配度**影响：

| Corpus 来源 | 样本数 | Prefixes | 最佳 acc | speedup |
|-----|--------|----------|---------|---------|
| Reference transcription (tokenize 参考文本) | 50k | 463k | 10.3% | 1.26× |
| mid3-23 输出（跑过模型的 token 序列）| 10k | 150k | 11.5% | 1.32× |

`build_ngram_from_model.py` 跑 draft（或 verifier）做 inference，收集其 **实际输出 token**，从这些 token 建 n-gram。比从 reference tokenize 高 15-20% 接受率，因为避免了 tokenizer/casing/标点不对齐。

用法（3-GPU 并行 shard）：
```bash
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((i+1)) python build_ngram_from_model.py \
    --model <model> --dataset <train_data> \
    --num_samples 10000 --shard_id $i --num_shards 3 \
    --output_shard ngram_model_shard_$i.pkl &
done
python build_ngram_from_model.py --mode merge \
  --shards ngram_model_shard_*.pkl --output ngram_corpus_model.pkl
```

进一步可以：
- 扩到 50k samples（3-GPU ~3h）
- 用 10B 自己生成（比 mid3-23 准但更贵：50k samples 约 15h on 3 GPUs）
- 保留 top-K 候选（当前只存 top-1）+ tree attention

### 14.5 Medusa 实现（做了，1.51× 胜出！）

自研 Medusa heads（不依赖 vLLM）。K=4 小 MLP head 挂在 verifier 最后一层 hidden state 上，每个 head 预测未来一个 offset 的 token。推理时 heads 并行产生 K 个 draft token，verifier 一次 batched 验证。

#### 关键文件
| 文件 | 作用 |
|---|---|
| `medusa_model.py` | `MedusaHead` + `MedusaHeads` 类（ResBlock + 共享 lm_head） |
| `collect_medusa_data.py` | **关键**：跑 verifier 实际 inference，**保存每步 hidden state + 下一个 token**（10k samples × avg 19 tok = ~900MB, BF16）。支持 `--shard_id/--num_shards` 多卡并行 |
| `train_medusa_v2.py` | 从 cached 数据训 heads，val split 选 best checkpoint |
| `train_medusa.py` | **v1（已废弃）** 用 reference 文本直接训，val acc 好但 inference acc 崩到 4% —— 分布失配 |
| `infer_gpu_medusa.py` | Inference 集成（greedy 接受，线性非树） |
| `medusa_heads_v2_best.pt` | 训练好的 heads（21M 参数，step 1782 val_loss=4.424） |

#### V1 vs V2 关键差异

| 方面 | v1 (失败) | v2 (成功) |
|---|---|---|
| 训练 target | IMDA reference 文本 tokenize | 3B 自己跑音频 → 输出 token |
| Hidden state 来源 | Pure text decoder forward | 真实 inference（speech encoder + adapter → text decoder） |
| Val acc (head 0/1/2/3) | 40/30/30/35% | 76/47/35/27% |
| **Inference acc** | **4.2%** | **38.8%** |
| **Latency** | 0.59s (vs baseline 0.53s, 慢 11%) | **0.35s** (vs baseline 0.53s, **1.51× 快**) |

**教训**：Medusa 训练数据必须匹配 inference 时的 hidden state 分布。对多模态（音频→文本），必须跑一遍真实 inference 收集 hidden state，不能用纯文本 text_decoder forward。

#### Medusa × 量化（3B, IMDA_PART1, 20 samples, L40）

heads 只在 bf16 hidden state 上训练了一次，下面所有量化变体都共用 `medusa_heads_v2_best.pt`：

| Quant | Baseline lat / tps | +Medusa lat / tps | Speedup (lat) | Acc | WER | VRAM |
|-------|---|---|---|---|---|---|
| **bf16** | 0.53 / 38.0 | **0.35 / 57.3** | **1.51×** | 38.8% | 1.51% | 7.2 GB |
| fp16 | 0.53 / 37.5 | 0.35 / 56.3 | 1.51× | 38.8% | 1.51% | 7.2 GB |
| int8 (bnb) | 1.94 / 10.3 | 1.14 / 17.5 | 1.70× | 37.5% | 1.51% | 5.2 GB |
| int4 (bnb) | 1.00 / 20.1 | 0.73 / 27.4 | 1.37× | 36.8% | **2.64%** | 4.2 GB |
| mlx4 | 0.91 / 22.0 | 0.56 / 35.7 | **1.62×** | 38.4% | 1.51% | **4.4 GB** |

关键发现：
- **Accept rate 跨量化稳定在 37-39%** —— heads trained on bf16 hidden state 对 mlx4/int8 的量化噪声鲁棒
- **bf16+medusa 是绝对最快**（0.35s, 57 tok/s）；**mlx4+medusa 是 VRAM-speed Pareto**（0.56s, 4.4 GB, WER 不变）
- int8+medusa speedup 最大（1.70×）只因 baseline 太慢，绝对值没用
- int4 保留其 WER 退化（2.64% vs 1.51%），和 Medusa 无关

#### Pipeline 命令

```bash
# 1. 收集 hidden states + 输出 token 配对（3 GPU 并行 ~25 min for 10k samples）
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((i+1)) python collect_medusa_data.py \
    --model /home/.../MERaLiON-2-3B \
    --dataset /home/.../IMDA_PART1_train \
    --num_samples 10000 --shard_id $i --num_shards 3 \
    --output_shard medusa_data_shard_$i.pt &
done

# 2. 训练 heads（单卡 ~3 min, val split 选 best）
python train_medusa_v2.py \
  --model /home/.../MERaLiON-2-3B \
  --data_shards medusa_data_shard_*.pt \
  --num_heads 4 --num_layers 1 \
  --batch_size 8 --grad_accum 2 --lr 1e-3 --epochs 3 \
  --output_best medusa_heads_v2_best.pt

# 3. Inference
python infer_gpu_medusa.py \
  --model /home/.../MERaLiON-2-3B \
  --heads medusa_heads_v2_best.pt \
  --dataset /home/.../IMDA_PART1_eval \
  --num_samples 20
```

#### 调试教训：head indexing 要对得上训练 offset

训练时 head k 学预测 `tokens[i+k+1]`（offset +k+1 从 "naturally predicted" token）。推理时 h_last 是 "产生 next_tok 的那个 hidden state"，所以 `head_k(h_last)` 预测 offset +k+1 后的 token：
- head 0 → 第 1 个 draft（next_tok 之后的第 1 个）
- head 1 → 第 2 个 draft
- head K-1 → 第 K 个 draft

**v1 bug**：以为 head 0 和 verifier's 自己的 +1 冗余，跳过了 head 0 用 heads 1..K-1 → draft 位置错位 → acc 暴跌到 6%。修成 heads 0..K-1 后 acc 飙到 38.8%。

### 14.6 若想进一步加速（未做）

1. **更多 Medusa heads**（K=6-8）—— 但 head 3 已经只有 27% acc，递减收益
2. **Tree attention**（每个 head 保留 top-K 候选，多分支并行 verify）—— 复杂度高，Medusa-2 paper 有
3. **EAGLE**（递归 draft，比 Medusa acc 更高）—— 需重训
4. **改用 vLLM** —— PagedAttention + 原生 spec decoding + tensor parallel
5. **量化 verifier 到 W4A16**（AWQ/GPTQ）让 decode 更 memory-bound → 给 spec 更多 headroom

### 14.6 Minibatch 与量化收益的关系（概念）

N-gram 和 model-draft 都是 batch=1 decode，完全 memory-bandwidth-bound。量化（W4A16）在这种场景收益最大（weights 4× 变小 = 4× decode 加速），是因为 weight loading 是瓶颈。

反之，batch 增大到 compute-bound（L40 上大约 B≥32）：
- 量化 kernel 必须 dequant 回 FP16 再走 tensor core → 没带宽收益
- 反而 dequant 本身变成额外开销

**重要含义**：对于单条 ASR inference（典型场景），量化始终有价值；但如果跑 offline batch 服务，batch 大到一定程度后 W4A16 不再加速。

### 14.7 关键文件

| 文件 | 作用 |
|---|---|
| `infer_gpu.py` | Ngram spec 主入口（`--speculative --corpus ... [--compile]`） |
| `infer_gpu_spec_draft.py` | Model-draft 原版（保留作 A/B 对照，见 fast 版注释） |
| `infer_gpu_spec_draft_fast.py` | Model-draft 优化版（prealloc cache / static shape / 持久 buffer / `--compile_draft`） |
| `build_ngram_corpus.py` | 从 IMDA train 集 reference 文本构造 prefix→next_tok 字典 |
| `build_ngram_from_model.py` | **优先用**：从模型输出 token 构造 corpus（匹配 verifier 实际分布，acc +15-20%）。支持 `--shard_id`/`--num_shards` 多卡并行 |
| `ngram_corpus.pkl` | Reference-based corpus（PART1 50k samples, 463k prefixes, 6MB） |
| `ngram_corpus_model.pkl` | Model-based corpus（mid3-23 输出 10k samples, 150k prefixes, 2MB）|
| `run_draft_spec_bench.sh` | Model-draft bench（BF16 / BnB4 draft × BF16 / INT8 verifier） |

### 14.8 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| Pruned draft NaN logits | speech encoder FP16 溢出（max ≈ 65504） | `input_features_d` dtype 从 `speech_encoder.parameters()` 取（BF16 而非 WQLinear scales 的 FP16） |
| AWQ4 pruned draft NaN 依然 | `model.load_state_dict(_non_td_sd, strict=False)` 静默丢 speech encoder 权重 | 直接复用 `_pruned_full` 做 shell，不重建 |
| AWQ GEMM overflow | Gemma2 MLP gate×up ≈ 1e6 超 FP16 max，WQLinear kernel 要 FP16 I/O | 放弃 AWQ4 draft，改用 BnB int4（compute_dtype=bf16 不溢出）|
| `awq_ext` is None | env 没装 GEMM C 扩展 | 包装 WQLinear_GEMM 用 `WQLinear_GEMM.forward` 而非 `awq_ext.gemm_forward_cuda` 直调 |
| `can't set attribute 'key_cache'` | HybridCache 把 `key_cache/value_cache/max_cache_len` 定义为只读 property | 我们存在 `_key_bufs/_value_bufs/_max_cache_len`，override property 暴露 |
| Compile 每步 recompile | `cache_position=torch.tensor([d_pos])` 每步新建 tensor object → dynamo guard 失效 | 用持久 `_d_cpos` buffer + `.fill_()`；`mark_static_address` |
| `build_ngram_from_model.py` `model.generate()` 报 `index_copy_ shape mismatch` | 剪枝模型 mid-block kv_heads 非均匀，HF 默认 HybridCache 按 config 的均匀值预分配 | 手写 decode loop，显式用 DynamicCache |

## 15. EAGLE 实验（2026-04，A100）

### 15.1 最终结果（IMDA_PART1, 20 samples, A100 sm_80, K=4 chain）

| Verifier | TPS Speedup | Latency Speedup | VRAM | WER | 备注 |
|---|---|---|---|---|---|
| bf16 (FA2) | 1.81× | 1.81× | 7.16 G | 6.74% | EAGLE chain baseline |
| W4A16 GPTQ + Marlin | 1.70× | 1.70× | ~4 G | 6.74% | Marlin 不擅长 batch=1 |
| W4A16 GPTQ + ExllamaV2 | 1.77× | 1.80× | 5.83 G | 6.74% | 接近 bf16 |
| **W4A16 GPTQ + ExllamaV1** | **1.83×** | **1.90×** | **5.83 G** | **6.74%** | **best** |

**关键发现**：W4A16+ExllamaV1 的 EAGLE 比 bf16+EAGLE 还略快（1.83× vs 1.81×），同时省 19% VRAM。原因：exllama v1 专为 batch=1 单 token decode 优化，在小矩阵 / 低 launch overhead 上比 cuBLAS bf16 dispatch 还省。Marlin 反过来在 batch≥4 才显出来，对我们这种 batch=1 占多数的 spec decoding 场景反而不划算。

WER 0% 退化（6.74% 跨所有 quant 配置一致）—— RTN-style W4A16 校准对 audio-conditioned 输出影响微乎其微。

### 15.2 训练改进（依次叠加，每次只动一个变量）

| 改动 | 单步 val_acc | 多步 val_acc (k=0/1/2/3) | TPS Speedup |
|---|---|---|---|
| 初版（teacher-forcing, alpha=0.1） | 70% | — | 1.5× |
| Scheduled sampling (p_max=0.5, alpha=0.5) | 73% | — | 1.7× |
| Multi-step unroll training (D=4) | 76% | 0.76 / 0.54 / 0.39 / 0.29 | 1.8× |
| ↑ + 数据从 10k → 30k+（多 dataset） | 76% | — | 2.11× ← 但有 train/test leakage |
| ↑ 修 leakage（bench 取 [0,N) 不打乱） | — | — | **1.81×**（真值） |

**Train/test 泄露的来源**：原 bench 用 `shuffle(seed=42)` 然后取 `[10500-N, 10500)`，对应原始 index 是随机的，跟训练区 `[30, 30+N_per_dataset)` 大概率重合。修法：bench 改为不打乱，取 `[0, N_bench)`；train `start_idx >= N_bench` 即可保证不重合。

### 15.3 走过没显著收益的方向

- **2-layer EAGLE**：参数 88M → 130M 但 acc 反而降。10k 样本喂不饱 130M，论文也是单层。
- **Tree attention（B=2 step-1）**：accept rate 跟 chain 接近，但 verifier 必须从 FA2 切到 SDPA（4D mask），attention 慢 30-50% 把 tree 收益吃光。绝对加速 1.36× < chain 1.81×。要值得做需要 FlashInfer kernel（300-500 行集成代码）。
- **HF transformers + compressed-tensors W4A16**：load 成功但走 `CompressedLinear.forward` 的 dequant fallback，0.33× 慢路径。HF transformers 没有 Marlin 的内置 dispatch，必须 vLLM 才能用上。

### 15.4 复现命令

最终最佳配置：
```bash
# 1. 训练 EAGLE（10k+ 样本，30 min in 3 GPU 并行）
NUM_SAMPLES_PER_DATASET=15000 UNROLL_DEPTH=4 HIDDEN_LOSS_ALPHA=0.5 \
SCHED_SAMPLING_MAX=0.5 EPOCHS=10 \
bash run_eagle_train.sh

# 2. 量化 verifier 到 W4A16 GPTQ-Marlin format（用 auto-gptq）
$PYTHON_PATH quantize_gptq_marlin.py \
    --src   /path/to/MERaLiON-2-3B \
    --out   quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin \
    --bits 4 --group_size 128 --n_calib 128

# 3. Bench EAGLE chain + W4A16 ExllamaV1 verifier
VERIFIER_QUANT=gptq_marlin GPTQ_KERNEL=exllama \
GPTQ_MARLIN_MODEL=quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin \
BF16_MODEL=/path/to/MERaLiON-2-3B \
SKIP_COLLECT=1 SKIP_TRAIN=1 FORCE_BENCH=1 \
bash run_eagle_train.sh
```

### 15.5 关键文件

| 文件 | 作用 |
|---|---|
| `eagle_model.py` | EAGLE 网络（fuse + Gemma2DecoderLayer + final_ln + shared embed/lm_head）|
| `train_eagle.py` | EAGLE 训练（teacher-forcing / scheduled sampling / multi-step unroll）|
| `infer_gpu_eagle.py` | EAGLE chain inference（K=4 默认，--quant gptq_marlin 走 W4A16 ExllamaV1）|
| `infer_gpu_eagle_tree.py` | Tree-attention 变体（保留作 reference，currently slower than chain on FA2 → SDPA）|
| `quantize_gptq_marlin.py` | 量化 text_decoder 为 GPTQ format with marlin-compatible flags |
| `load_gptq_marlin.py` | 绕过 optimum/gptqmodel，直接用 auto-gptq from_quantized 加载，支持 marlin/exllama/exllamav2 三种 kernel |
| `patch_autogptq_marlin_only.py` | auto-gptq 0.7.1 的 cuda_64/256 跟 torch 2.6 不兼容，patch setup.py 只编 marlin+exllama+exllamav2 |
| `setup_cuda_includes.sh` | conda env 装 nvcc 但缺 dev headers，把 pkgs 里的 CUDA include 一次性加进 CPATH |
| `requirements_marlin.txt` | 干净 env 配方（vllm 0.8.5 + transformers 4.51.3 + compressed-tensors 0.9.3 + …）|

### 15.6 EAGLE 加速进一步推（未做）

1. **更多训练数据**（30k → 100k+）：当前数据上限可能限制 deep-step accuracy（k=3 才 29%）
2. **Logit distillation**：现在只用 hard CE label，加 KL(verifier_logits || eagle_logits) 应该提升深步 acc
3. **EAGLE-2 论文的 dynamic tree**：动态选 top-N 候选，不是固定深度。要换 inference flow
4. **CUDA Graph / torch.compile + Marlin**：HF transformers 上不直接支持；要走 vLLM 路径或自己手写 cuda graph 包装 verifier decode。Marlin 的 1.37×（vLLM 实测）就是这条路拿到的

### 15.7 踩坑记录（EAGLE / Marlin / 环境）

| 问题 | 原因 | 修复 |
|---|---|---|
| EAGLE k>1 加速断崖式下降 | Inference 用 `pos_ids = cur_pos+k`（绝对位置），但训练用 `arange(T-1)` 从 0 开始；EAGLE 自己的 KV cache 每轮重置 → RoPE 应用错误位置 | `pos_ids_d = [[k]]`（EAGLE 内部协调系统）|
| EAGLE val_acc 76% 但 1.5× 上不去 | 单步 acc 看不出 unroll 训练效果，需要 multi-step val metric | `run_eval` 加 D=4 步 autoregressive accuracy 测量 |
| 2-layer EAGLE 反而变慢 | 130M 参数，10k 训练样本不够 | 退回单层 |
| HF transformers 加载 compressed-tensors W4A16 慢 0.33× | `CompressedLinear.forward` = dequant + 普通 F.linear，没 Marlin 路径，且 `register_offload_parameter` hook 引入 host-device 搬运 | 改用 GPTQ format + auto-gptq + marlin/exllama kernel |
| auto-gptq 装上但 marlin kernel 调不动（`'function' object has no attribute 'mul'`）| pip wheel 没编 CUDA ext | 源码编译，只保留 marlin+exllama+exllamav2，跳过 cuda_64/256（torch 2.6 API 不兼容）|
| auto-gptq cuda_64 编译报 `vec.type()` → ScalarType 转换失败 | torch 2.6 移除了 DeprecatedTypeProperties → ScalarType 隐式转换 | `patch_autogptq_marlin_only.py` 直接跳过这两个 kernel |
| 缺 `crt/host_defines.h` / `cusparse.h` / `nv/target` / `cublas_v2.h` | conda env 装了 nvcc 但没装完整 toolkit dev headers | `setup_cuda_includes.sh`：扫 pkgs cache 把所有 cuda-*-dev 的 include 加进 CPATH |
| `libc10.so: cannot open shared object file` | torch 的 lib 没在 LD_LIBRARY_PATH | `export LD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH` |
| `optimum.gptq → gptqmodel → ModuleNotFoundError: pcre` | optimum 1.24+ 切到 gptqmodel 后端，gptqmodel 依赖 python-pcre（要系统 libpcre dev）| 绕过 HfQuantizer，写 `load_gptq_marlin.py` 直接调 auto-gptq from_quantized |
| `Marlin kernel requires torch_dtype=torch.float16` | auto-gptq 三个 W4A16 kernel 都要求 fp16 I/O | 整个 MERaLiON 加载用 fp16 而非 bf16 |
| pip 把 torch / transformers / tokenizers / hf_hub 顺手升级搞坏 env | gptqmodel + autoawq_kernels 拉了不兼容版本 | 全程 `pip install --no-deps`，并 pin `transformers==4.51.3 / tokenizers==0.21.x / hf_hub==0.30.x / triton==3.2.0`；最终重建 env，见 `requirements_marlin.txt` |
| Bench 测出 2.11× speedup 太好了 | bench 用 `shuffle(seed=42)` 然后位置 `[10500-N, 10500)` 跟训练区 `[30, 30+N)` 大概率重合 | bench 改为不打乱，取 `[0, N)`；start_idx ≥ bench N |
