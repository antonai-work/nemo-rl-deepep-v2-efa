#!/usr/bin/env bash
#
# Build the all-PRs-applied nemo-rl-deepep-v2-efa image.
#
# Produces a single image that layers:
#   - nvidia/cuda:12.9.0-devel-ubuntu24.04 (Docker Hub public)
#   - aws-efa-installer 1.48.0
#   - aws-ofi-nccl @ 6e504db (source-built)
#   - NCCL 2.30.4 (pip)
#   - DeepEP V2 + patches 0001-0003 (PR #612)
#   - Megatron-LM + patches 0004-0006 (PR #4632)
#   - NeMo-RL + patch 0007 (PR #2410)
#
# Usage:
#   ./docker/build.sh <image-tag>
#
# Examples:
#   ./docker/build.sh nemo-rl-deepep-v2-efa:latest
#   ./docker/build.sh 123456789012.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-deepep-v2-efa:allprs-$(git rev-parse --short HEAD)
#
# After build, validate with:
#   docker run --rm <image-tag> bash /opt/docker/preflight.sh
#   -> expected: 5/5 checks PASS
#
# To push:
#   docker push <image-tag>
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <image-tag>" >&2
  echo "Example: $0 nemo-rl-deepep-v2-efa:latest" >&2
  exit 1
fi

IMAGE_TAG="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile"

if [ ! -f "${DOCKERFILE}" ]; then
  echo "ERROR: expected Dockerfile at ${DOCKERFILE}" >&2
  exit 1
fi

# Sanity-check all seven patches are present.
for N in 0001 0002 0003 0004 0005 0006 0007; do
  MATCH=$(ls "${REPO_ROOT}/patches/${N}-"*.patch 2>/dev/null | head -1)
  if [ -z "${MATCH}" ]; then
    echo "ERROR: missing patch ${N}-*.patch in ${REPO_ROOT}/patches/" >&2
    exit 1
  fi
done

echo "[build.sh] Dockerfile:    ${DOCKERFILE}"
echo "[build.sh] Build context: ${REPO_ROOT}"
echo "[build.sh] Image tag:     ${IMAGE_TAG}"
echo "[build.sh] Patches:       $(ls "${REPO_ROOT}/patches/"*.patch | wc -l) present"
echo ""

DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${EXTRA_ARGS[@]}" \
  "${REPO_ROOT}"

echo ""
echo "[build.sh] Built ${IMAGE_TAG}"
echo "[build.sh] Validate: docker run --rm ${IMAGE_TAG} bash /opt/docker/preflight.sh"
