# syntax=docker/dockerfile:1.10
# =============================================================================
# nemo-rl-fullstack - All-PRs-applied end-to-end validation image
# =============================================================================
#
# Purpose:
#   Single container image that layers all three of our upstream PRs on
#   top of a pure-upstream substrate. This is the "no shim" test --
#   validates that the three upstream PRs, applied in isolation on top of
#   vanilla everything, produce a working NeMo-RL -> Megatron-Core ->
#   DeepEP V2 training stack on AWS EFA.
#
#   Gates whether we file the three upstream PRs.
#
# Pinned layers (see docs/state-vector-analysis-2026-04-29.md for rationale):
#   Base OS + CUDA       nvidia/cuda:12.9.0-devel-ubuntu24.04   vanilla
#   EFA userspace        aws-efa-installer 1.48.0               vanilla (--build-ngc)
#   aws-ofi-nccl         6e504db3403931cde43a2335adcc73fbc69cccac  vanilla
#                        (upstream fix superseded our PR #1206)
#   NCCL                 nvidia-nccl-cu13 >= 2.30.4             vanilla
#   GDRCopy              v2.5.1                                 vanilla
#   DeepEP V2            dmvevents/DeepEP-1 branch              OUR FORK
#                        aws-efa-auto-qp-cap-v2 @ c84dcac       = vanilla V2 +
#                        PR #612's 3 commits (auto-QP cap, EFA gbs fast path,
#                        kScaleoutUpdateInterval tuning)
#   Megatron-LM          dmvevents/Megatron-LM branch           OUR FORK
#                        deepep-v2-elasticbuffer-support         = vanilla
#                        Megatron (23dd639cf) + Shape Y patch (cbacb0bbf,
#                        3 commits: ElasticBuffer support, EventOverlap
#                        fallback, num_experts-explicit backward dispatch)
#   NeMo-RL              NVIDIA-NeMo/RL@46be4e8 + local patch    OUR PATCH
#                        /tmp/nemo-rl-pr-prep.final.patch        = vanilla +
#                        LD_LIBRARY_PATH fix for EFA+OFI discovery (PR #2359
#                        resurrection), closes issue #1973
#   api-shim             NOT INCLUDED                            by design
#                        DEEP_EP_USE_V2_SHIM=0 at runtime, no /opt/api-shim
#                        path, Megatron calls deep_ep.ElasticBuffer natively.
#
# V1 reference: /home/ubuntu/awesome-inferencing/examples/nemo-rl-deepep-working/
# Shape Y proven base: bench/logs/megatron-shapeY-v2-validation-20260505T165839Z/
#
# Build FROM REPO ROOT (patch file is copied from /tmp into context):
#   cp /tmp/nemo-rl-pr-prep.final.patch \
#      integrations/nemo-rl-fullstack/nemo-rl-pr-prep.final.patch
#   cd /home/ubuntu/deepep-v2-integration
#   docker build -f integrations/nemo-rl-fullstack/Dockerfile \
#     -t 058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-fullstack:allprs-20260505 .
#
# Preflight gate (inside running pod):
#   1. NeMo-RL LD fix is live:
#        ldconfig -p | grep libnccl-net    # expect /opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so
#   2. Shape Y probe is active:
#        python3 -c "from megatron.core.transformer.moe.fused_a2a \
#                    import HAVE_DEEP_EP_V2; print(HAVE_DEEP_EP_V2)"  # expect True
#   3. No shim:
#        python3 -c "import deep_ep; print(deep_ep.Buffer)" \
#          # expect class deep_ep.buffer.Buffer (not CompatBuffer)
#        test -e /opt/api-shim || echo NO_SHIM_OK
# =============================================================================

FROM deepep-base-v2:latest

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# DeepEP V2: the base image already has /opt/DeepEP on the fork branch
# aws-efa-auto-qp-cap-v2. Re-pin exactly so this image's provenance is
# reproducible without relying on the base tag.
# -----------------------------------------------------------------------------
ARG DEEPEP_FORK=https://github.com/dmvevents/DeepEP-1.git
ARG DEEPEP_BRANCH=aws-efa-auto-qp-cap-v2
ARG DEEPEP_SHA=c84dcac613c8df743a6487a312bbc966c745c600

RUN set -eux; \
    cd /opt/DeepEP; \
    git remote set-url origin "${DEEPEP_FORK}"; \
    git fetch origin "${DEEPEP_BRANCH}":"${DEEPEP_BRANCH}" 2>&1 || \
        (git fetch origin "${DEEPEP_BRANCH}" && git branch -f "${DEEPEP_BRANCH}" FETCH_HEAD); \
    git checkout "${DEEPEP_BRANCH}"; \
    git reset --hard "${DEEPEP_SHA}"; \
    git log -1 --oneline; \
    python3 -c "import deep_ep; print('DeepEP V2 (fork):', deep_ep.ElasticBuffer)"

# -----------------------------------------------------------------------------
# Megatron-LM: fork branch deepep-v2-elasticbuffer-support.
# The Shape Y patch lives at cbacb0bbf (3 commits ahead of vanilla Megatron
# release tip 23dd639cf). Install editable so the patched fused_a2a.py is on
# PYTHONPATH as /opt/Megatron-LM/megatron/...
# -----------------------------------------------------------------------------
ARG MEGATRON_FORK=https://github.com/dmvevents/Megatron-LM.git
ARG MEGATRON_BRANCH=deepep-v2-elasticbuffer-support
ARG MEGATRON_SHA=cbacb0bbf2a73ed28ffcfcff61beddd31ed6ea76

RUN set -eux; \
    git clone --depth=50 -b "${MEGATRON_BRANCH}" "${MEGATRON_FORK}" /opt/Megatron-LM; \
    cd /opt/Megatron-LM; \
    git reset --hard "${MEGATRON_SHA}"; \
    git log -1 --oneline

# Megatron-Core + pure-Python deps. --no-deps on megatron-core keeps us
# out of transformer-engine / grouped-gemm CUDA-build traps (our buildkit
# has no GPU, and Shape Y's fused_a2a path doesn't need them to exist at
# import time).
ARG MEGATRON_CORE_VERSION=0.12.0
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "megatron-core==${MEGATRON_CORE_VERSION}" && \
    pip install --no-cache-dir --break-system-packages \
      "numpy<2" pyyaml regex sentencepiece accelerate einops ninja packaging

# Install Megatron-LM editable so its patched `megatron.core.transformer.moe.fused_a2a`
# wins over the pip-installed megatron-core version.
# Use --no-deps so pip does not try to satisfy the heavy extras like
# nvidia-modelopt-torch / flash-attn that are declared in setup.py.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      -e /opt/Megatron-LM || echo "editable install failed, PYTHONPATH fallback still works"

# -----------------------------------------------------------------------------
# NeMo-RL: vanilla upstream + our local LD_LIBRARY_PATH patch.
# The local patch adds /opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib to
# LD_LIBRARY_PATH in the NeMo-RL Dockerfiles AND adds an EFA Qwen3 recipe.
#
# We DO NOT install NeMo-RL from its own Dockerfile here; we layer it on
# top of deepep-base-v2 and apply the LD_LIBRARY_PATH at the ENV line
# below (line 163) instead. The recipe YAML from the patch is still copied
# in so it can be used via `--config examples/configs/recipes/llm/aws-efa-...`.
# -----------------------------------------------------------------------------
ARG NEMORL_GIT_REF=46be4e8

RUN set -eux; \
    git clone https://github.com/NVIDIA-NeMo/RL.git /opt/NeMo-RL; \
    cd /opt/NeMo-RL; \
    git fetch origin "${NEMORL_GIT_REF}" || true; \
    git checkout "${NEMORL_GIT_REF}"; \
    git log -1 --oneline

# Apply the LD fix + recipe patch. The patch targets docker/Dockerfile +
# docker/Dockerfile.ngc_pytorch (files we don't execute) + a new YAML
# recipe under examples/configs/recipes/llm/. Only the YAML recipe
# survives as a runtime artifact here; the LD fix is re-applied to this
# Dockerfile's ENV block further down. We still `git apply` the whole
# patch so the recipe lands on disk.
COPY integrations/nemo-rl-fullstack/nemo-rl-pr-prep.final.patch /tmp/nemo-rl-patch.patch
RUN set -eux; \
    cd /opt/NeMo-RL; \
    git apply --verbose /tmp/nemo-rl-patch.patch; \
    ls -l examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml; \
    rm /tmp/nemo-rl-patch.patch

# NeMo-RL pure-Python deps. Mirror nemo-rl-deepep-v2 exactly. --no-deps
# on packages that would pull CUDA-compiling wheels.
RUN pip install --no-cache-dir --break-system-packages \
      "ray[default]>=2.47,<3" \
      "hydra-core>=1.3" \
      "omegaconf>=2.3" \
      wandb \
      datasets \
      "transformers>=4.51" \
      "tokenizers>=0.15" \
      typer \
      rich \
      protobuf \
      jsonlines \
      nvtx \
      colored \
      tiktoken \
      blobfile \
      debugpy \
      math-verify \
      pyzmq \
      torchdata \
      tensorboard \
      num2words \
      "pillow>=10" \
      "mlflow>=2.17,<3.6" \
 && pip install --no-cache-dir --break-system-packages --no-deps \
      nccl4py \
      pybase64 \
      swanlab \
      decord

# -----------------------------------------------------------------------------
# NO-SHIM INVARIANTS
# The base image's deepep-base-v2 already does NOT carry /opt/api-shim --
# only the *-deepep-v2 overlays add it. We explicitly remove any trace of it
# here so `test -e /opt/api-shim` returns non-zero at preflight.
# We also ensure sitecustomize.py from the shim is not picked up via any
# Python site path.
# -----------------------------------------------------------------------------
RUN rm -rf /opt/api-shim \
 && find / -name 'sitecustomize.py' -path '*/api-shim/*' -delete 2>/dev/null || true \
 && test ! -e /opt/api-shim

# -----------------------------------------------------------------------------
# PYTHONPATH: /opt/Megatron-LM so Megatron's patched fused_a2a.py wins,
#              /opt/NeMo-RL so nemo_rl is importable.
#              NO /opt/api-shim.
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/opt/Megatron-LM:/opt/NeMo-RL

# -----------------------------------------------------------------------------
# LD_LIBRARY_PATH: the NeMo-RL PR's claim -- EFA + OFI lib dirs must be
# on the dynamic linker path for NCCL to load the OFI plugin. This is the
# exact ENV line from /tmp/nemo-rl-pr-prep.final.patch.
# Preserve upstream LD_LIBRARY_PATH (EFA + NCCL from base image).
# -----------------------------------------------------------------------------
ENV LD_LIBRARY_PATH="/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# Runtime env block (validates all three PRs and matches Shape Y recipe).
# EP_EFA_MAX_QPS=2: per DeepEP PR #612 conservative auto-cap for 16-rank EFA.
# -----------------------------------------------------------------------------
ENV DEEP_EP_USE_V2_SHIM=0 \
    NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

# Copy the Shape Y training driver from megatron-deepep-v2 -- already
# proven to exercise the V2 dispatch path.
COPY integrations/megatron-deepep-v2/train_step_shapeY.py /opt/train_step_shapeY.py

# NeMo-RL wrapper driver: imports nemo_rl (to prove the library loads on the
# stack) then runs the Shape Y train step. This exercises the
# NeMo-RL -> Megatron-Core -> DeepEP V2 import chain.
COPY integrations/nemo-rl-fullstack/train_via_nemo_rl.py /opt/train_via_nemo_rl.py

# Preflight script (three checks in one).
COPY integrations/nemo-rl-fullstack/preflight.sh /opt/preflight.sh
RUN chmod +x /opt/preflight.sh

# EFA traffic verifier (needed inside pod for snapshot/delta checks).
COPY scripts/verify_efa_traffic.sh /opt/verify_efa_traffic.sh
RUN chmod +x /opt/verify_efa_traffic.sh

WORKDIR /opt

LABEL org.opencontainers.image.title="nemo-rl-fullstack" \
      org.opencontainers.image.description="All-PRs-applied E2E image: DeepEP V2 PR #612 + Megatron Shape Y PR + NeMo-RL LD fix PR, NO shim" \
      deepep.branch="${DEEPEP_BRANCH}" \
      deepep.sha="${DEEPEP_SHA}" \
      megatron.branch="${MEGATRON_BRANCH}" \
      megatron.sha="${MEGATRON_SHA}" \
      nemo_rl.git_ref="${NEMORL_GIT_REF}" \
      shim.included="false"
