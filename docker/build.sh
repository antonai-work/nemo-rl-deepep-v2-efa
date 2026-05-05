#!/usr/bin/env bash
#
# nemo-rl-fullstack - local + CodeBuild build driver.
#
# Builds the "all-PRs-applied" E2E image from the Dockerfile in this
# directory. Mirrors the dynamo-inference build.sh shape so CodeBuild and
# local iteration share one entrypoint (gist
# e03bab9e96d0f8933f6dc15d02894b91, section "Local build (dev iteration)").
#
# Docker build context is the REPO ROOT (not this directory) because the
# Dockerfile COPYs from api-shim/, scripts/shared/, integrations/megatron-
# deepep-v2/, and integrations/nemo-rl-fullstack/.
#
# Usage:
#   ./build.sh [--tag TAG] [--repo-root DIR] [--registry REG] [--push] [--no-cache]
#
# Examples:
#   # Local dev (from integrations/nemo-rl-fullstack):
#   ./build.sh --tag allprs-$(git rev-parse --short HEAD)
#
#   # CodeBuild (from anywhere, pointing at repo root):
#   ./build.sh --tag allprs-${SHA} --repo-root ${CODEBUILD_SRC_DIR} --no-push
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TAG="latest"
REPO_ROOT="${DEFAULT_REPO_ROOT}"
REGISTRY=""
PUSH=false
NO_CACHE=false
IMAGE_NAME="nemo-rl-fullstack"

print_usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Build the nemo-rl-fullstack image (all upstream PRs applied, no shim).

Options:
  -t, --tag TAG              Image tag (default: latest). For CodeBuild use
                             "allprs-\${SHA}".
  -r, --repo-root DIR        Repo root to use as docker build context
                             (default: ../../ relative to this script).
      --registry REG         Optional ECR registry prefix
                             (058264135704.dkr.ecr.us-east-2.amazonaws.com).
  -p, --push                 Push to registry after build.
  -n, --no-cache             Build without Docker cache.
      --no-push              Explicit no-push (default; exists for CodeBuild
                             parity with dynamo-inference --no-extract).
  -h, --help                 Show this message.

Examples:
  # Local rebuild at head SHA:
  $0 --tag allprs-\$(git -C "${DEFAULT_REPO_ROOT}" rev-parse --short HEAD)

  # CodeBuild invocation:
  $0 --tag allprs-\${SHA} --repo-root \${CODEBUILD_SRC_DIR} --no-push
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tag)        TAG="$2";        shift 2 ;;
    -r|--repo-root)  REPO_ROOT="$2";  shift 2 ;;
    --registry)      REGISTRY="$2";   shift 2 ;;
    -p|--push)       PUSH=true;       shift ;;
    --no-push)       PUSH=false;      shift ;;
    -n|--no-cache)   NO_CACHE=true;   shift ;;
    -h|--help)       print_usage; exit 0 ;;
    *)               echo "Unknown arg: $1" >&2; print_usage; exit 1 ;;
  esac
done

CACHE_OPT=""
if [ "$NO_CACHE" = "true" ]; then
  CACHE_OPT="--no-cache"
fi

DOCKERFILE="${REPO_ROOT}/integrations/nemo-rl-fullstack/Dockerfile"
if [ ! -f "${DOCKERFILE}" ]; then
  echo "ERROR: expected Dockerfile at ${DOCKERFILE}" >&2
  exit 1
fi

# The Dockerfile COPYs /tmp/nemo-rl-pr-prep.final.patch from this
# directory (see Dockerfile comments). If the file is not staged, surface
# a helpful error rather than letting docker fail mid-build.
PATCH_FILE="${REPO_ROOT}/integrations/nemo-rl-fullstack/nemo-rl-pr-prep.final.patch"
if [ ! -f "${PATCH_FILE}" ]; then
  echo "WARN: ${PATCH_FILE} is missing." >&2
  echo "      The Dockerfile COPYs this file from integrations/nemo-rl-fullstack/." >&2
  echo "      For CodeBuild runs: ensure the patch is committed to the repo." >&2
  echo "      For local runs: cp /tmp/nemo-rl-pr-prep.final.patch \"${PATCH_FILE}\"" >&2
  exit 1
fi

echo "[build.sh] Dockerfile:        ${DOCKERFILE}"
echo "[build.sh] Context (root):    ${REPO_ROOT}"
echo "[build.sh] Image:             ${IMAGE_NAME}:${TAG}"

# Prereq check: the Dockerfile does FROM deepep-base-v2:latest. The
# CodeBuild pre_build phase re-tags the ECR base image to this name; in
# local dev you have to have built the base separately.
if ! docker image inspect "deepep-base-v2:latest" >/dev/null 2>&1; then
  echo "ERROR: deepep-base-v2:latest is not in local docker. Build it first:" >&2
  echo "       cd ${REPO_ROOT} && ./base/deepep-base-v2/build.sh" >&2
  exit 1
fi

DOCKER_BUILDKIT=1 docker build ${CACHE_OPT} \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}:${TAG}" \
  "${REPO_ROOT}"

if [ -n "${REGISTRY}" ]; then
  docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
  echo "[build.sh] Tagged: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
fi

if [ "${PUSH}" = "true" ]; then
  if [ -z "${REGISTRY}" ]; then
    echo "ERROR: --push requires --registry" >&2
    exit 1
  fi
  docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
  echo "[build.sh] Pushed: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
fi

echo "[build.sh] done."
docker images "${IMAGE_NAME}" --filter "reference=${IMAGE_NAME}:${TAG}"
