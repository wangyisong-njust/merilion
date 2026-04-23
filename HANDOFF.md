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

### 14.5 若想进一步加速（未做）

1. **训小 draft**（蒸馏 ~500M，计算比 1:6-1:20）+ 自研 Medusa/EAGLE 头
2. **改用 vLLM** —— PagedAttention + 原生 spec decoding + tensor parallel
3. **量化 verifier 到 W4A16**（AWQ/GPTQ）让 decode 更 memory-bound → 给 spec 更多 headroom

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
