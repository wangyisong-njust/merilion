"""Inventory the current env: which Marlin-capable stacks are installed?"""
import importlib


def check(name, attr=None):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        s = f"  ✓ {name:<28} {v}"
        if attr:
            for a in attr.split(","):
                a = a.strip()
                ok = hasattr(m, a) or any(a in x for x in dir(m))
                s += f"  [{a}={'y' if ok else 'n'}]"
        print(s)
        return True
    except ImportError as e:
        print(f"  ✗ {name:<28} NOT installed  ({e})")
        return False
    except Exception as e:
        # Module is installed but import failed (e.g. dep version mismatch).
        # Don't silently say "NOT installed" — show the real error.
        print(f"  ! {name:<28} INSTALLED BUT IMPORT FAILED: {type(e).__name__}: {e}")
        return False


print("─── Core ─────────────────────────────────────────────────")
check("torch")
check("transformers")
check("accelerate")

print("\n─── Marlin stack candidates ──────────────────────────────")
check("compressed_tensors")
check("vllm")
check("awq")                              # autoawq
check("awq_ext")                          # autoawq cuda kernels
check("autoawq_kernels")
check("auto_gptq")
check("gptqmodel")                        # the modern fork

print("\n─── Marlin ops in torch ──────────────────────────────────")
import torch
hits = []
for ns_name in dir(torch.ops):
    if ns_name.startswith("_"):
        continue
    try:
        ns = getattr(torch.ops, ns_name)
        for op in dir(ns):
            if "marlin" in op.lower():
                hits.append(f"torch.ops.{ns_name}.{op}")
    except Exception:
        pass
if hits:
    for h in hits: print(" ", h)
else:
    print("  (none)  → no Marlin C++ kernel registered in this env")

print("\n─── GPU ──────────────────────────────────────────────────")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}")
print(f"  torch CUDA: {torch.version.cuda}")
