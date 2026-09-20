#!/bin/bash
# Laptop side: build the `science` image with CUDA JAX (the same pinned stack as the Claude
# container, without Claude) and export it as a tarball Apptainer can convert on a cluster.
#
#   ./container/build_sif.sh
#
# then copy the tarball over and, on the cluster:
#   apptainer build ~/containers/gsoc_t33.sif docker-archive://gsoc_t33_science.tar.gz
#
# The image ships self-contained CUDA-12 wheels, so it does not need a matching CUDA on the
# host -- only an NVIDIA driver >= 525. That covers doslis (535 / L4) and UGent (H100).
# Run it with `apptainer run --nv`, or the GPUs will not be visible.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.env"
OUT_DIR="${OUT_DIR:-$HOME/Desktop/GSoC/containers}"
mkdir -p "$OUT_DIR"

docker build --target science -t "$SCIENCE_IMAGE" \
    --build-arg JAX_SPEC="$CLUSTER_JAX_SPEC" \
    "$HERE"

docker save "$SCIENCE_IMAGE" | gzip > "$OUT_DIR/gsoc_t33_science.tar.gz"
ls -lh "$OUT_DIR/gsoc_t33_science.tar.gz"
