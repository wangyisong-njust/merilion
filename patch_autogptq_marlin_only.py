"""Patch /tmp/AutoGPTQ/setup.py to build ONLY the marlin CUDA extension.

auto-gptq 0.7.1's old kernels (cuda_64, cuda_256, exllama, exllamav2) use
torch APIs that no longer exist in torch 2.6 (`vec.type()` etc.) and fail
to compile. We only need the marlin kernel for W4A16 inference, so this
script rewrites the `extensions = [...]` block in setup.py to keep only
autogptq_marlin_cuda.

Usage:
  git clone --depth=1 https://github.com/AutoGPTQ/AutoGPTQ.git /tmp/AutoGPTQ
  python patch_autogptq_marlin_only.py    # default path /tmp/AutoGPTQ/setup.py
  cd /tmp/AutoGPTQ && rm -rf build dist *.egg-info
  python setup.py bdist_wheel
  pip install --no-deps --force-reinstall dist/auto_gptq*.whl
"""
import argparse
import re
import sys


NEW_BLOCK = '''extensions = [
        cpp_extension.CUDAExtension(
            'autogptq_marlin_cuda',
            [
                'autogptq_extension/marlin/marlin_cuda.cpp',
                'autogptq_extension/marlin/marlin_cuda_kernel.cu',
                'autogptq_extension/marlin/marlin_repack.cu'
            ]
        ),
    ]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup-py", default="/tmp/AutoGPTQ/setup.py")
    args = ap.parse_args()

    with open(args.setup_py) as f:
        src = f.read()

    # Match from `extensions = [` through the last appended extension's closing
    # bracket, up to (but not including) `additional_setup_kwargs = {`.
    new_src, n = re.subn(
        r"extensions = \[[\s\S]*?additional_setup_kwargs = \{",
        NEW_BLOCK + "\n    additional_setup_kwargs = {",
        src, count=1,
    )
    if n != 1:
        print(f"ERROR: didn't find the extensions block in {args.setup_py}",
              file=sys.stderr)
        sys.exit(1)

    with open(args.setup_py, "w") as f:
        f.write(new_src)
    print(f"patched {args.setup_py} (marlin-only)")

    # Sanity check: should now contain exactly one CUDAExtension call.
    n_cuda = new_src.count("CUDAExtension")
    n_cpp  = new_src.count("CppExtension")
    print(f"  CUDAExtension count: {n_cuda}  (expected 1)")
    print(f"  CppExtension count:  {n_cpp}   (expected 0)")
    if n_cuda != 1 or n_cpp != 0:
        print("WARN: extension counts unexpected — open setup.py and verify.")


if __name__ == "__main__":
    main()
