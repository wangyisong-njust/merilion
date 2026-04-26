#!/usr/bin/env bash
# Source this (don't execute) to populate CPATH with every CUDA dev header
# directory found under /home/jinchao/miniconda3/pkgs.
#
# Usage:
#   source setup_cuda_includes.sh
#
# After sourcing, build auto-gptq:
#   cd /tmp/AutoGPTQ && rm -rf build dist *.egg-info
#   $PYTHON_PATH setup.py bdist_wheel

PKGS_ROOT=${PKGS_ROOT:-/home/jinchao/miniconda3/pkgs}

# All header roots to scan.  Order: most-specific globs first.
CANDIDATES=(
    "$PKGS_ROOT"/cuda-*-dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/cuda-*-dev*/include
    "$PKGS_ROOT"/libcublas*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libcublas*dev*/include
    "$PKGS_ROOT"/libcusparse*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libcusparse*dev*/include
    "$PKGS_ROOT"/libcurand*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libcurand*dev*/include
    "$PKGS_ROOT"/libcusolver*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libcusolver*dev*/include
    "$PKGS_ROOT"/libcufft*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libcufft*dev*/include
    "$PKGS_ROOT"/libnvjitlink*dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/libnvjitlink*dev*/include
    "$PKGS_ROOT"/cuda-cccl*/targets/x86_64-linux/include
    "$PKGS_ROOT"/cuda-cccl*/include
    "$PKGS_ROOT"/cuda-nvtx*/targets/x86_64-linux/include
    "$PKGS_ROOT"/cuda-nvtx*/include
    "$PKGS_ROOT"/cuda-driver-dev*/targets/x86_64-linux/include
    "$PKGS_ROOT"/cuda-driver-dev*/include
    "$PKGS_ROOT"/cuda-profiler-api*/targets/x86_64-linux/include
    "$PKGS_ROOT"/cuda-profiler-api*/include
)

added=()
for pat in "${CANDIDATES[@]}"; do
    for d in $pat; do
        [ -d "$d" ] || continue
        # avoid duplicates
        case ":$CPATH:" in
            *":$d:"*) ;;
            *) CPATH="$d:$CPATH"; added+=("$d") ;;
        esac
    done
done
export CPATH

echo "Added ${#added[@]} CUDA include dir(s) to CPATH:"
for d in "${added[@]}"; do echo "  $d"; done

echo ""
echo "Header lookup test:"
for h in cublas_v2.h cusparse.h curand.h cusolver_common.h cufft.h \
         crt/host_defines.h nv/target cccl/c/types.h cuda_runtime.h; do
    found=""
    IFS=':'
    for p in $CPATH; do
        if [ -e "$p/$h" ]; then found="$p"; break; fi
    done
    unset IFS
    if [ -n "$found" ]; then
        printf "  %-30s ✓ %s\n" "$h" "$found"
    else
        printf "  %-30s ✗ NOT FOUND\n" "$h"
    fi
done
