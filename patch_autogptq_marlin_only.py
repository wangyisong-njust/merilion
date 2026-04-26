"""Patch /tmp/AutoGPTQ/setup.py to build only the kernels we want.

auto-gptq 0.7.1's general CUDA kernels (cuda_64, cuda_256) use
`vec.type()` which torch 2.6 removed; they break the build. We skip
those and keep the kernels we actually use:
  - autogptq_marlin_cuda     (W4A16 marlin; high-throughput at batch>=4)
  - exllama_kernels          (W4A16 exllama; best at batch=1 decode)
  - exllamav2_kernels        (W4A16 exllamav2; fastest single-token decode)

If exllama / exllamav2 also hit torch 2.6 API issues at compile time,
fall back to marlin-only by passing --marlin-only.

Usage:
  git clone --depth=1 https://github.com/AutoGPTQ/AutoGPTQ.git /tmp/AutoGPTQ
  python patch_autogptq_marlin_only.py    # marlin + exllama + exllamav2
  python patch_autogptq_marlin_only.py --marlin-only   # marlin only (fallback)
  cd /tmp/AutoGPTQ && rm -rf build dist *.egg-info
  python setup.py bdist_wheel
  pip install --no-deps --force-reinstall dist/auto_gptq*.whl
"""
import argparse
import re
import sys


MARLIN_EXT = '''        cpp_extension.CUDAExtension(
            'autogptq_marlin_cuda',
            [
                'autogptq_extension/marlin/marlin_cuda.cpp',
                'autogptq_extension/marlin/marlin_cuda_kernel.cu',
                'autogptq_extension/marlin/marlin_repack.cu'
            ]
        ),'''

EXLLAMA_EXT = '''        cpp_extension.CUDAExtension(
            "exllama_kernels",
            [
                "autogptq_extension/exllama/exllama_ext.cpp",
                "autogptq_extension/exllama/cuda_buffers.cu",
                "autogptq_extension/exllama/cuda_func/column_remap.cu",
                "autogptq_extension/exllama/cuda_func/q4_matmul.cu",
                "autogptq_extension/exllama/cuda_func/q4_matrix.cu"
            ]
        ),
        cpp_extension.CUDAExtension(
            "exllamav2_kernels",
            [
                "autogptq_extension/exllamav2/ext.cpp",
                "autogptq_extension/exllamav2/cuda/q_matrix.cu",
                "autogptq_extension/exllamav2/cuda/q_gemm.cu",
            ]
        ),'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup-py", default="/tmp/AutoGPTQ/setup.py")
    ap.add_argument("--marlin-only", action="store_true",
                    help="Skip exllama / exllamav2 kernels (fallback if they "
                         "fail to compile on torch 2.6).")
    args = ap.parse_args()

    if args.marlin_only:
        kernels = MARLIN_EXT
    else:
        kernels = MARLIN_EXT + "\n" + EXLLAMA_EXT
    NEW_BLOCK = f"extensions = [\n{kernels}\n    ]\n"

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
