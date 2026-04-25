"""Probe whether the compressed_tensors lib in this env has a Marlin kernel
path, and what `CompressedLinear.forward` actually calls."""
import inspect


def main():
    try:
        import compressed_tensors as ct
        print(f"compressed_tensors: {ct.__version__}")
    except ImportError:
        print("compressed_tensors NOT installed"); return

    # Walk submodules looking for Marlin-related classes.
    import pkgutil
    found = []
    for m in pkgutil.walk_packages(ct.__path__, prefix="compressed_tensors."):
        n = m.name.lower()
        if "marlin" in n or "linear" in n:
            found.append(m.name)
    print("\nMarlin / Linear modules:")
    for n in found:
        print(" ", n)

    # Look at CompressedLinear.forward.
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        src = inspect.getsource(CompressedLinear.forward)
        print("\nCompressedLinear.forward source:")
        print(src)
    except Exception as e:
        print(f"  (couldn't read CompressedLinear.forward: {e})")

    # Check torch ops for Marlin.
    import torch
    print("\nTorch ops with 'marlin' in name:")
    if hasattr(torch.ops, "compressed_tensors"):
        for op in dir(torch.ops.compressed_tensors):
            if "marlin" in op.lower():
                print("  torch.ops.compressed_tensors.", op, sep="")
    # vLLM-style ops?
    for ns in ("_C", "marlin", "vllm"):
        if hasattr(torch.ops, ns):
            for op in dir(getattr(torch.ops, ns)):
                if "marlin" in op.lower():
                    print(f"  torch.ops.{ns}.{op}")


if __name__ == "__main__":
    main()
