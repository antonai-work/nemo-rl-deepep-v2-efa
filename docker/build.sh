#!/usr/bin/env bash
#
# Build the all-PRs-applied nemo-rl-deepep-v2-efa image.
#
# Two build modes:
#
#   --mode fast     (default, ~5-10 min)
#       FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a, adds only
#       the NeMo-RL framework layer. Requires network access to GHCR (and
#       possibly a PAT with read:packages scope until the base image package
#       is flipped to public).
#
#   --mode vanilla  (~45 min cold build)
#       Inline the full 240-line base stack (nvidia/cuda:12.9.0-devel-ubuntu24.04
#       through EFA + aws-ofi-nccl + NCCL + GDRCopy + DeepEP V2 + patches)
#       then add the framework layer. Use when GHCR is unreachable or for
#       end-to-end reproducibility from vanilla NVIDIA CUDA.
#
# Both modes produce byte-identical framework content on top of identical
# base layers. The framework layer stacks:
#   - Megatron-LM + patches 0004-0006 (PR #4632)
#   - NeMo-RL + patch 0007 (PR #2410)
#
# Usage:
#   ./docker/build.sh [--mode fast|vanilla] <image-tag>
#
# Examples:
#   ./docker/build.sh nemo-rl-deepep-v2-efa:latest
#   ./docker/build.sh --mode fast 123456789012.dkr.ecr.us-east-2.amazonaws.com/nemo-rl:allprs-$(git rev-parse --short HEAD)
#   ./docker/build.sh --mode vanilla nemo-rl-deepep-v2-efa:vanilla-check
#
# After build, validate with:
#   docker run --rm <image-tag> bash /opt/docker/preflight.sh
#   -> expected: 5/5 checks PASS
#
# To push:
#   docker push <image-tag>
#
set -euo pipefail

MODE="fast"
IMAGE_TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--mode fast|vanilla] <image-tag> [docker build args...]"
      exit 0
      ;;
    *)
      if [[ -z "${IMAGE_TAG}" ]]; then
        IMAGE_TAG="$1"
        shift
      else
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -z "${IMAGE_TAG}" ]]; then
  echo "Usage: $0 [--mode fast|vanilla] <image-tag>" >&2
  echo "Example: $0 --mode fast nemo-rl-deepep-v2-efa:latest" >&2
  exit 1
fi

case "${MODE}" in
  fast|vanilla)
    ;;
  *)
    echo "ERROR: --mode must be 'fast' or 'vanilla' (got '${MODE}')" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "ERROR: expected Dockerfile at ${DOCKERFILE}" >&2
  exit 1
fi

# Sanity-check all seven patches are present.
for N in 0001 0002 0003 0004 0005 0006 0007; do
  MATCH=$(ls "${REPO_ROOT}/patches/${N}-"*.patch 2>/dev/null | head -1)
  if [[ -z "${MATCH}" ]]; then
    echo "ERROR: missing patch ${N}-*.patch in ${REPO_ROOT}/patches/" >&2
    exit 1
  fi
done

echo "[build.sh] Dockerfile:    ${DOCKERFILE}"
echo "[build.sh] Build context: ${REPO_ROOT}"
echo "[build.sh] Image tag:     ${IMAGE_TAG}"
echo "[build.sh] Build mode:    ${MODE}"
echo "[build.sh] Patches:       $(ls "${REPO_ROOT}/patches/"*.patch | wc -l) present"

# Wave 7d-1 OKR-1: source canonical pins from pins.env at the repo root.
# Single source of truth for fork/branch/SHA so a rebase of
# dmvevents/DeepEP-1@aws-efa-auto-qp-cap-v2 or Megatron-LM@deepep-v2-
# elasticbuffer-support only needs to touch one file per consumer repo.
# Dockerfile retains hardcoded ARG defaults as a fallback.
PINS_ENV="${REPO_ROOT}/pins.env"
PIN_BUILD_ARGS=()
if [[ -f "${PINS_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PINS_ENV}"
  echo "[build.sh] pins.env:      ${PINS_ENV}"
  echo "[build.sh]   DEEPEP_FORK=${DEEPEP_FORK:-<unset>}"
  echo "[build.sh]   DEEPEP_BRANCH=${DEEPEP_BRANCH:-<unset>}"
  echo "[build.sh]   DEEPEP_SHA=${DEEPEP_SHA:-<unset>}"
  echo "[build.sh]   MEGATRON_FORK=${MEGATRON_FORK:-<unset>}"
  echo "[build.sh]   MEGATRON_BRANCH=${MEGATRON_BRANCH:-<unset>}"
  echo "[build.sh]   MEGATRON_SHA=${MEGATRON_SHA:-<unset>}"
  for VAR in DEEPEP_FORK DEEPEP_BRANCH DEEPEP_SHA \
             MEGATRON_FORK MEGATRON_BRANCH MEGATRON_SHA; do
    if [[ -n "${!VAR:-}" ]]; then
      PIN_BUILD_ARGS+=(--build-arg "${VAR}=${!VAR}")
    fi
  done
else
  echo "[build.sh] pins.env:      <missing at ${PINS_ENV}>, using Dockerfile ARG defaults"
fi
echo ""

DOCKER_BUILDKIT=1 docker build \
  --build-arg "BUILD_MODE=${MODE}" \
  "${PIN_BUILD_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${EXTRA_ARGS[@]}" \
  "${REPO_ROOT}"

echo ""
echo "[build.sh] Built ${IMAGE_TAG} (mode=${MODE})"
echo "[build.sh] Validate: docker run --rm ${IMAGE_TAG} bash /opt/docker/preflight.sh"
