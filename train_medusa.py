"""Train Medusa heads on MERaLiON-2-3B, frozen verifier.

We only need the text decoder's last-layer hidden states to train the heads.
For ASR our training target is the TRANSCRIPTION, so we tokenize the reference
text and feed it directly into `model.text_decoder` (bypassing the audio path
entirely — the heads don't care about audio).

Loss: for each head k∈[1..K], CE between head_k(h[:, :-k]) and target
      tokens at position t+k.  Sum across heads.

Usage:
    python train_medusa.py \\
        --model /home/kaixin/programs/LLM_base_model/MERaLiON-2-3B \\
        --dataset /home/kaixin/meralion_datasets/train/ASR/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 30000 --start_idx 30 --seq_len 256 \\
        --num_heads 4 --num_layers 1 \\
        --batch_size 4 --grad_accum 4 --lr 5e-4 \\
        --epochs 2 \\
        --output /home/kaixin/yisong/merilion/medusa_heads.pt
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medusa_model import MedusaHeads, attach_medusa


class TranscriptDataset(Dataset):
    """Tokenize reference transcripts from an IMDA dataset split."""
    def __init__(self, dataset_path, tokenizer, num_samples, start_idx, seq_len):
        from datasets import load_from_disk
        data = load_from_disk(os.path.abspath(dataset_path))
        end = min(start_idx + num_samples, len(data))
        self.samples = []
        for i in range(start_idx, end):
            s = data[i]
            oa = s.get("other_attributes") or {}
            ref = oa.get("Transcription") or oa.get("transcription")
            if ref is None:
                ans = s.get("answer")
                ref = ans.get("text") if isinstance(ans, dict) else ans
            if not ref or not str(ref).strip():
                continue
            # Use BOS + transcript text so the model sees a familiar prefix.
            ids = tokenizer.encode(str(ref), add_special_tokens=True)
            if len(ids) < 4:
                continue
            # Trim/pad to seq_len; we'll mask pads in loss.
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            self.samples.append(ids)
        print(f"  dataset loaded: {len(self.samples)} usable samples "
              f"(from [{start_idx},{end}))")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch, pad_id):
    """Right-pad to max length in batch.  Returns (input_ids, attention_mask)."""
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn     = torch.zeros_like(input_ids)
    for i, ids in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return input_ids, attn


def medusa_loss(heads: MedusaHeads, h: torch.Tensor,
                input_ids: torch.Tensor, attn_mask: torch.Tensor,
                vocab_size: int):
    """
    h:         (B, T, H) last hidden state from frozen verifier
    input_ids: (B, T) tokens
    attn_mask: (B, T)
    Returns scalar loss = sum over heads of head_k's CE on offset-k targets.
    """
    K = heads.num_heads
    logits_all = heads(h)                    # (K, B, T, V)
    total_loss = 0.0
    per_head = []
    per_head_acc = []
    for k in range(K):
        # head_k predicts token at offset (k+1)
        # predictions at positions t = 0..T-(k+2)
        # targets at positions   t = (k+1)..T-1
        if input_ids.size(1) <= k + 1:
            continue
        pred = logits_all[k, :, :-(k+1), :].contiguous()          # (B, T-k-1, V)
        tgt  = input_ids[:, (k+1):].contiguous()                  # (B, T-k-1)
        m    = attn_mask[:, (k+1):].contiguous().bool()           # (B, T-k-1)
        # Only count positions where both pred position (t) and tgt position (t+k+1) are non-pad
        m = m & attn_mask[:, :-(k+1)].contiguous().bool()
        if m.sum() == 0:
            continue
        pred_flat = pred.view(-1, vocab_size)
        tgt_flat  = tgt.view(-1)
        loss_k_each = F.cross_entropy(pred_flat, tgt_flat, reduction="none")
        loss_k = (loss_k_each * m.view(-1).float()).sum() / m.sum().float()
        total_loss = total_loss + loss_k
        per_head.append(loss_k.item())
        # Top-1 acc on masked positions
        pred_tok = pred_flat.argmax(-1)
        correct = (pred_tok == tgt_flat) & m.view(-1)
        per_head_acc.append(correct.sum().item() / m.sum().item())
    return total_loss, per_head, per_head_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--start_idx", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="override epochs with an absolute step budget")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="medusa_heads.pt")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)

    print(f"Loading verifier from {args.model} …")
    t0 = time.time()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infer_gpu import load_model_gpu
    model, processor = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = processor.tokenizer
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print(f"Attaching {args.num_heads} Medusa heads (residual layers={args.num_layers}) …")
    heads = attach_medusa(model, num_heads=args.num_heads, num_layers=args.num_layers)
    n_trainable = sum(p.numel() for p in heads.heads.parameters())
    print(f"  trainable params: {n_trainable/1e6:.2f} M")

    print(f"Loading dataset {args.dataset} …")
    ds = TranscriptDataset(args.dataset, tokenizer,
                           args.num_samples, args.start_idx, args.seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    n_updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = args.max_steps if args.max_steps > 0 else n_updates_per_epoch * args.epochs
    print(f"  dataloader: {len(loader)} microbatches, grad_accum={args.grad_accum}")
    print(f"  updates per epoch: {n_updates_per_epoch}  |  total updates: {total_updates}")

    optim = torch.optim.AdamW(
        heads.heads.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    # Cosine with linear warmup
    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_updates - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1 + math.cos(math.pi * progress))

    vocab_size = model.text_decoder.lm_head.out_features
    print(f"  vocab_size: {vocab_size}")

    global_step = 0
    microstep   = 0
    epoch_losses = []
    t_start = time.time()
    for epoch in range(10**9):
        if global_step >= total_updates:
            break
        for input_ids, attn_mask in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            # Forward verifier to get last hidden states (no grad, full prec).
            with torch.no_grad():
                td_out = model.text_decoder(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # last hidden state
                h = td_out.hidden_states[-1]   # (B, T, H)

            # Heads are the only trainable part.
            heads.heads.train()
            loss, per_head, per_head_acc = medusa_loss(
                heads, h, input_ids, attn_mask, vocab_size)
            if not torch.is_tensor(loss) or loss.item() == 0:
                continue
            (loss / args.grad_accum).backward()

            microstep += 1
            if microstep % args.grad_accum == 0:
                # apply LR schedule
                lr_scale = lr_at(global_step)
                for g in optim.param_groups:
                    g["lr"] = args.lr * lr_scale
                torch.nn.utils.clip_grad_norm_(heads.heads.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    accs = ", ".join(f"{a:.3f}" for a in per_head_acc)
                    dt = time.time() - t_start
                    print(f"  [ep {epoch} step {global_step:4d}/{total_updates}]  "
                          f"loss={loss.item():.3f}  "
                          f"per_head_loss=[{', '.join(f'{x:.3f}' for x in per_head)}]  "
                          f"per_head_acc=[{accs}]  "
                          f"lr={args.lr * lr_scale:.2e}  "
                          f"elapsed={dt/60:.1f}m  "
                          f"eta={dt*(total_updates-global_step)/global_step/60:.1f}m",
                          flush=True)

                if global_step % args.save_every == 0:
                    ckpt = {
                        "heads_state": heads.trainable_state_dict(),
                        "config": {
                            "num_heads": args.num_heads,
                            "num_layers": args.num_layers,
                            "hidden_size": heads.hidden_size,
                        },
                        "step": global_step,
                    }
                    torch.save(ckpt, args.output)
                    print(f"  saved checkpoint → {args.output} at step {global_step}")

                if global_step >= total_updates:
                    break
        epoch_losses.append(loss.item())

    ckpt = {
        "heads_state": heads.trainable_state_dict(),
        "config": {
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "hidden_size": heads.hidden_size,
        },
        "step": global_step,
    }
    torch.save(ckpt, args.output)
    print(f"\ntraining done.")
    print(f"  final saved → {args.output}")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
