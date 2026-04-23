"""Train Medusa heads on (hidden_state, token) pairs captured during
*actual inference* — fixes train/inference distribution mismatch that
caused v1 heads to get only 4% accept rate despite 30-50% training acc.

Data: list of dicts {"tokens": [int, ...], "hiddens": Tensor[T, H]}
per sample from `collect_medusa_data.py`.

For training position i in a sample (0 <= i < T):
  hiddens[i]     → head_k predicts tokens[i + k + 1]  (for k = 0..K-1)
  The target exists only if i + k + 1 < T.  We mask out-of-range positions.

Held-out val split selects best checkpoint by val CE loss.
"""
import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medusa_model import MedusaHeads, attach_medusa


class MedusaDataset(Dataset):
    """Loads shard .pt files produced by collect_medusa_data.py."""
    def __init__(self, shard_paths, min_tokens: int = 5):
        self.samples = []
        for p in shard_paths:
            print(f"  loading {p}")
            shard = torch.load(p, weights_only=False)
            for s in shard:
                if len(s["tokens"]) >= min_tokens:
                    self.samples.append(s)
        print(f"  total: {len(self.samples)} samples  "
              f"tokens: {sum(len(s['tokens']) for s in self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return s["hiddens"], torch.tensor(s["tokens"], dtype=torch.long)


def collate(batch):
    """Pad (hiddens, tokens) to max T in batch; return with mask."""
    max_T = max(h.shape[0] for h, _ in batch)
    H     = batch[0][0].shape[1]
    hiddens = torch.zeros((len(batch), max_T, H), dtype=batch[0][0].dtype)
    tokens  = torch.zeros((len(batch), max_T),    dtype=torch.long)
    mask    = torch.zeros((len(batch), max_T),    dtype=torch.bool)
    for i, (h, t) in enumerate(batch):
        T = h.shape[0]
        hiddens[i, :T] = h
        tokens[i, :T]  = t
        mask[i, :T]    = 1
    return hiddens, tokens, mask


def medusa_loss(heads: MedusaHeads, hiddens, tokens, mask, vocab_size: int):
    """
    hiddens: (B, T, H)  mixed pad, covered by mask
    tokens:  (B, T)
    mask:    (B, T) bool — True for real positions
    Loss: sum_k CE(head_k(hiddens[:, :-(k+1)]), tokens[:, (k+1):])
          restricted to positions where both ends are in mask.
    """
    K = heads.num_heads
    logits_all = heads(hiddens)          # (K, B, T, V)
    total = 0.0
    per_head_loss, per_head_acc = [], []
    for k in range(K):
        off = k + 1
        if hiddens.size(1) <= off:
            continue
        pred = logits_all[k, :, :-off, :].contiguous()
        tgt  = tokens[:, off:].contiguous()
        m    = mask[:, off:].bool() & mask[:, :-off].bool()
        if m.sum() == 0:
            continue
        p_flat = pred.view(-1, vocab_size)
        t_flat = tgt.view(-1)
        m_flat = m.view(-1)
        ce = F.cross_entropy(p_flat, t_flat, reduction="none")
        loss_k = (ce * m_flat.float()).sum() / m_flat.float().sum()
        total = total + loss_k
        per_head_loss.append(loss_k.item())
        with torch.no_grad():
            acc = ((p_flat.argmax(-1) == t_flat) & m_flat).sum().float() / m_flat.float().sum()
            per_head_acc.append(acc.item())
    return total, per_head_loss, per_head_acc


def evaluate(heads, loader, device, vocab_size):
    heads.eval()
    n = 0
    sum_loss = 0.0
    per_head_correct = [0] * heads.num_heads
    per_head_total   = [0] * heads.num_heads
    with torch.no_grad():
        for hiddens, tokens, mask in loader:
            hiddens = hiddens.to(device, non_blocking=True)
            tokens  = tokens.to(device, non_blocking=True)
            mask    = mask.to(device, non_blocking=True)
            K = heads.num_heads
            logits_all = heads(hiddens)
            for k in range(K):
                off = k + 1
                if hiddens.size(1) <= off:
                    continue
                pred = logits_all[k, :, :-off, :]
                tgt  = tokens[:, off:]
                m    = mask[:, off:].bool() & mask[:, :-off].bool()
                if m.sum() == 0:
                    continue
                p_flat = pred.reshape(-1, vocab_size)
                t_flat = tgt.reshape(-1)
                m_flat = m.reshape(-1)
                ce = F.cross_entropy(p_flat, t_flat, reduction="none")
                l = (ce * m_flat.float()).sum() / m_flat.float().sum()
                sum_loss += l.item()
                n += 1
                correct = ((p_flat.argmax(-1) == t_flat) & m_flat).sum().item()
                per_head_correct[k] += correct
                per_head_total[k]   += m_flat.sum().item()
    mean_loss = sum_loss / max(1, n)
    per_head_acc = [per_head_correct[k] / max(1, per_head_total[k])
                    for k in range(heads.num_heads)]
    return mean_loss, per_head_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Used to instantiate heads with correct hidden/vocab")
    ap.add_argument("--data_shards", nargs="+", required=True)
    ap.add_argument("--val_frac",    type=float, default=0.05)
    ap.add_argument("--num_heads",   type=int, default=4)
    ap.add_argument("--num_layers",  type=int, default=1)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--grad_accum",  type=int, default=2)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--epochs",      type=int, default=3)
    ap.add_argument("--max_steps",   type=int, default=-1)
    ap.add_argument("--eval_every",  type=int, default=200)
    ap.add_argument("--log_every",   type=int, default=20)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      default="medusa_heads_v2.pt")
    ap.add_argument("--output_best", default="medusa_heads_v2_best.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load the base model only for lm_head and hidden/vocab sizes — we do NOT
    # need to run it during training (hidden states are cached on disk).
    print(f"Loading verifier (for lm_head + shape info) …")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infer_gpu import load_model_gpu
    model, _ = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    heads = attach_medusa(model, num_heads=args.num_heads, num_layers=args.num_layers)
    n_trainable = sum(p.numel() for p in heads.heads.parameters())
    print(f"  trainable params: {n_trainable/1e6:.2f} M")
    vocab_size = model.text_decoder.lm_head.out_features

    # Build dataset and split
    print("Loading training data …")
    ds_full = MedusaDataset(args.data_shards, min_tokens=5)
    n_val = max(1, int(len(ds_full) * args.val_frac))
    indices = list(range(len(ds_full)))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)
    print(f"  train={len(ds_train)}  val={len(ds_val)}")

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True)

    n_updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = args.max_steps if args.max_steps > 0 else n_updates_per_epoch * args.epochs
    print(f"  train microbatches={len(train_loader)}  "
          f"updates/epoch={n_updates_per_epoch}  total_updates={total_updates}")

    optim = torch.optim.AdamW(
        heads.heads.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_updates - args.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1 + math.cos(math.pi * prog))

    best_val_loss = float("inf")
    global_step = 0
    microstep = 0
    t_start = time.time()
    for epoch in range(10**9):
        if global_step >= total_updates:
            break
        heads.heads.train()
        for hiddens, tokens, mask in train_loader:
            hiddens = hiddens.to(device, non_blocking=True)
            tokens  = tokens.to(device, non_blocking=True)
            mask    = mask.to(device, non_blocking=True)
            loss, per_head_loss, per_head_acc = medusa_loss(
                heads, hiddens, tokens, mask, vocab_size)
            if not torch.is_tensor(loss):
                continue
            (loss / args.grad_accum).backward()

            microstep += 1
            if microstep % args.grad_accum == 0:
                for g in optim.param_groups:
                    g["lr"] = args.lr * lr_at(global_step)
                torch.nn.utils.clip_grad_norm_(heads.heads.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    phs = ", ".join(f"{x:.3f}" for x in per_head_loss)
                    pha = ", ".join(f"{x:.3f}" for x in per_head_acc)
                    dt = time.time() - t_start
                    eta = dt * (total_updates - global_step) / max(1, global_step) / 60
                    print(f"  [ep {epoch} step {global_step:5d}/{total_updates}] "
                          f"loss={loss.item():.3f}  ph_loss=[{phs}]  "
                          f"ph_acc=[{pha}]  lr={optim.param_groups[0]['lr']:.2e}  "
                          f"elapsed={dt/60:.1f}m  eta={eta:.1f}m", flush=True)

                if global_step % args.eval_every == 0 or global_step == total_updates:
                    val_loss, val_acc = evaluate(heads, val_loader, device, vocab_size)
                    va_s = ", ".join(f"{x:.3f}" for x in val_acc)
                    print(f"  [val @ step {global_step}]  val_loss={val_loss:.3f}  "
                          f"val_ph_acc=[{va_s}]", flush=True)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt = {
                            "heads_state": heads.trainable_state_dict(),
                            "config": {
                                "num_heads": args.num_heads,
                                "num_layers": args.num_layers,
                                "hidden_size": heads.hidden_size,
                            },
                            "step": global_step,
                            "val_loss": val_loss,
                            "val_acc":  val_acc,
                        }
                        torch.save(ckpt, args.output_best)
                        print(f"    ↳ new best val_loss, saved → {args.output_best}")
                    heads.heads.train()

                if global_step >= total_updates:
                    break

    # final save
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
    print(f"\nfinal saved → {args.output}  (best saved → {args.output_best}, val={best_val_loss:.3f})")


if __name__ == "__main__":
    main()
