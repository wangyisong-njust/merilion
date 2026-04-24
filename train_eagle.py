"""Train an EAGLE draft model on the audio-conditioned shards produced by
collect_medusa_data.py.

For each sample (tokens: [t_0..t_{T-1}], hiddens: Tensor[T, H]):

    input_ids_t  = tokens[t-1]         # previous token
    prev_h_t     = hiddens[t-1]        # verifier's hidden at t-1
    target_t     = tokens[t]           # next token to predict
    target_h_t   = hiddens[t]          # next hidden (for MSE loss)

Loss = CE(logits, target_t)  +  alpha · MSE(h_draft, target_h_t)

All verifier parameters are frozen; only EAGLE.fuse + EAGLE.layer +
EAGLE.final_ln are trainable (~85 M params for Gemma2-2B).
"""
import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_model import EAGLE, attach_eagle


class EagleDataset(Dataset):
    def __init__(self, shard_paths, min_tokens: int = 4):
        self.samples = []
        for p in shard_paths:
            print(f"  loading {p}")
            shard = torch.load(p, weights_only=False)
            for s in shard:
                if len(s["tokens"]) >= min_tokens:
                    self.samples.append(s)
        print(f"  total: {len(self.samples)} samples, "
              f"{sum(len(s['tokens']) for s in self.samples):,} tokens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return s["hiddens"], torch.tensor(s["tokens"], dtype=torch.long)


def collate(batch, pad_id):
    max_T = max(h.shape[0] for h, _ in batch)
    H     = batch[0][0].shape[1]
    dtype = batch[0][0].dtype
    hiddens = torch.zeros((len(batch), max_T, H), dtype=dtype)
    tokens  = torch.full((len(batch), max_T), pad_id, dtype=torch.long)
    mask    = torch.zeros((len(batch), max_T),    dtype=torch.bool)
    for i, (h, t) in enumerate(batch):
        T = h.shape[0]
        hiddens[i, :T] = h
        tokens[i, :T]  = t
        mask[i, :T]    = 1
    return hiddens, tokens, mask


def eagle_forward_train(eagle, rotary_emb, hiddens, tokens, mask,
                        device, dtype):
    """
    hiddens (B, T, H), tokens (B, T), mask (B, T).
    Shift so that at position t (1..T-1):
      input_ids_t = tokens[t-1], prev_h_t = hiddens[t-1]
    and targets are tokens[t], hiddens[t].

    Builds a causal attention mask over the T-1 shifted positions for
    the single decoder layer.
    """
    B, T, H = hiddens.shape
    if T < 2:
        return None, None, None
    # Shift
    prev_h     = hiddens[:, :-1].contiguous()   # (B, T-1, H)
    prev_ids   = tokens[:, :-1].contiguous()    # (B, T-1)
    prev_mask  = mask[:, :-1]
    target_ids = tokens[:, 1:].contiguous()
    target_h   = hiddens[:, 1:].contiguous()
    tgt_mask   = mask[:, 1:]

    prev_h   = prev_h.to(device, dtype=dtype, non_blocking=True)
    prev_ids = prev_ids.to(device, non_blocking=True)
    target_ids = target_ids.to(device, non_blocking=True)
    target_h = target_h.to(device, dtype=dtype, non_blocking=True)
    tgt_mask = tgt_mask.to(device, non_blocking=True)

    # RoPE position_embeddings for T-1 positions.
    position_ids = torch.arange(T - 1, device=device).unsqueeze(0)  # (1, T-1)
    position_ids = position_ids.expand(B, -1)
    # rotary_emb accepts hidden_states + position_ids
    position_embeddings = rotary_emb(prev_h, position_ids)

    # Causal 4D mask (B, 1, T-1, T-1) with -inf above diag.
    min_v = torch.finfo(dtype).min
    causal = torch.full((T - 1, T - 1), min_v, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    attn_mask = causal[None, None, :, :]   # broadcast over B

    cache_position = torch.arange(T - 1, device=device)
    logits, h_next, _ = eagle(
        input_ids=prev_ids,
        prev_hidden=prev_h,
        position_ids=position_ids,
        attention_mask=attn_mask,
        cache_position=cache_position,
        past_key_value=None,
        position_embeddings=position_embeddings,
    )
    return logits, h_next, (target_ids, target_h, tgt_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_shards", nargs="+", required=True)
    ap.add_argument("--val_frac",    type=float, default=0.05)
    ap.add_argument("--batch_size",  type=int,   default=4)
    ap.add_argument("--grad_accum",  type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int,  default=100)
    ap.add_argument("--epochs",      type=int,   default=2)
    ap.add_argument("--max_steps",   type=int,   default=-1)
    ap.add_argument("--hidden_loss_alpha", type=float, default=0.1,
                    help="Weight for MSE(h_draft, h_verifier_next) loss")
    ap.add_argument("--eval_every",  type=int,   default=200)
    ap.add_argument("--log_every",   type=int,   default=50)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      default="eagle_final.pt")
    ap.add_argument("--output_best", default="eagle_best.pt")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load verifier (for embed_tokens / lm_head / rotary + config).
    from infer_gpu import load_model_gpu
    verifier, _ = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=args.device)
    verifier.eval()
    for p in verifier.parameters():
        p.requires_grad_(False)

    eagle, rotary_emb = attach_eagle(verifier, device, dtype=torch.bfloat16)
    n_trainable = sum(p.numel() for p in eagle.parameters())
    print(f"  EAGLE trainable params: {n_trainable/1e6:.2f} M")
    vocab_size = verifier.text_decoder.lm_head.out_features

    # Data
    print("Loading shards …")
    ds_full = EagleDataset(args.data_shards, min_tokens=4)
    n_val = max(1, int(len(ds_full) * args.val_frac))
    indices = list(range(len(ds_full))); random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)
    print(f"  train={len(ds_train)}  val={len(ds_val)}")

    tokenizer_pad_id = 0   # we don't have the tokenizer here; 0 is safe (PAD)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer_pad_id),
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer_pad_id),
        num_workers=2, pin_memory=True)

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    print(f"  updates per epoch={updates_per_epoch}  total={total_updates}")

    optim = torch.optim.AdamW(eagle.parameters(), lr=args.lr,
                              weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_updates - args.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1 + math.cos(math.pi * prog))

    def run_eval(loader):
        eagle.eval()
        sum_ce, sum_mse, total, correct = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for hiddens, tokens, m in loader:
                out = eagle_forward_train(eagle, rotary_emb, hiddens,
                                          tokens, m, device, torch.bfloat16)
                if out[0] is None:
                    continue
                logits, h_next, (tgt_ids, tgt_h, tgt_mask) = out
                m_prev = m[:, :-1].to(tgt_mask.device)
                m_flat = (tgt_mask.bool() & m_prev.bool()).reshape(-1)
                lf = logits.reshape(-1, logits.size(-1))
                tf = tgt_ids.reshape(-1)
                ce_each = F.cross_entropy(lf, tf, reduction="none")
                m_float = m_flat.float()
                sum_ce += (ce_each * m_float).sum().item()
                mse_each = ((h_next - tgt_h) ** 2).mean(-1).reshape(-1)
                sum_mse += (mse_each * m_float).sum().item()
                total += m_float.sum().item()
                correct += ((lf.argmax(-1) == tf) & m_flat).sum().item()
        eagle.train()
        if total == 0:
            return None, None, None
        return sum_ce / total, sum_mse / total, correct / total

    eagle.train()
    best_val = float("inf")
    global_step, microstep = 0, 0
    t_start = time.time()
    for epoch in range(10**9):
        if global_step >= total_updates:
            break
        for hiddens, tokens, m in train_loader:
            out = eagle_forward_train(eagle, rotary_emb, hiddens,
                                      tokens, m, device, torch.bfloat16)
            if out[0] is None:
                continue
            logits, h_next, (tgt_ids, tgt_h, tgt_mask) = out
            m_prev = m[:, :-1].to(tgt_mask.device)
            # Mask invalid positions (prev or target was pad).
            m_valid = tgt_mask.bool() & m_prev.bool()
            m_flat = m_valid.reshape(-1)
            lf = logits.reshape(-1, logits.size(-1))
            tf = tgt_ids.reshape(-1)
            ce_each = F.cross_entropy(lf, tf, reduction="none")
            ce = (ce_each * m_flat.float()).sum() / m_flat.float().sum().clamp(min=1)
            mse_each = ((h_next - tgt_h) ** 2).mean(-1).reshape(-1)
            mse = (mse_each * m_flat.float()).sum() / m_flat.float().sum().clamp(min=1)
            loss = ce + args.hidden_loss_alpha * mse

            (loss / args.grad_accum).backward()
            microstep += 1
            if microstep % args.grad_accum == 0:
                for g in optim.param_groups:
                    g["lr"] = args.lr * lr_at(global_step)
                torch.nn.utils.clip_grad_norm_(eagle.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    with torch.no_grad():
                        acc = ((lf.argmax(-1) == tf) & m_flat).sum().float() / m_flat.float().sum().clamp(min=1)
                    dt = time.time() - t_start
                    eta = dt * (total_updates - global_step) / max(1, global_step) / 60
                    print(f"  [ep {epoch} step {global_step:5d}/{total_updates}] "
                          f"loss={loss.item():.3f}  ce={ce.item():.3f}  mse={mse.item():.4f}  "
                          f"acc={acc.item():.3f}  lr={optim.param_groups[0]['lr']:.2e}  "
                          f"elapsed={dt/60:.1f}m  eta={eta:.1f}m", flush=True)

                if global_step % args.eval_every == 0 or global_step == total_updates:
                    vce, vmse, vacc = run_eval(val_loader)
                    if vce is not None:
                        v_total = vce + args.hidden_loss_alpha * vmse
                        print(f"  [val @ step {global_step}]  val_loss={v_total:.3f}  "
                              f"val_ce={vce:.3f}  val_mse={vmse:.4f}  val_acc={vacc:.3f}",
                              flush=True)
                        if v_total < best_val:
                            best_val = v_total
                            torch.save({
                                "eagle_state":   eagle.trainable_state_dict(),
                                "config":        dict(eagle.config.to_dict()
                                                      if hasattr(eagle.config, 'to_dict')
                                                      else vars(eagle.config)),
                                "step":          global_step,
                                "val_ce":        vce, "val_mse": vmse, "val_acc": vacc,
                            }, args.output_best)
                            print(f"    ↳ new best, saved → {args.output_best}")

                if global_step >= total_updates:
                    break

    torch.save({
        "eagle_state": eagle.trainable_state_dict(),
        "config":      dict(eagle.config.to_dict() if hasattr(eagle.config, 'to_dict') else vars(eagle.config)),
        "step":        global_step,
    }, args.output)
    print(f"\nfinal saved → {args.output}")
    print(f"  best val_loss={best_val:.3f}  at {args.output_best}")


if __name__ == "__main__":
    main()
