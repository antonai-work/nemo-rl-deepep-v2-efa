# syntax=docker/dockerfile:1.10
# =============================================================================
# nemo-rl-deepep-v2-efa - dual-path build (fast GHCR base | from-vanilla)
# =============================================================================
#
# One multi-stage Dockerfile that supports two build paths:
#
#   --build-arg BUILD_MODE=fast     (default)
#       Start from the pre-built deepep-v2-efa-base image on GHCR. Skips the
#       ~20-minute base stack compile and lands you at the NeMo-RL framework
#       layer in ~5-10 minutes. Requires network access to GHCR (the package
#       may require a PAT with read:packages scope until the org flips it to
#       public; a CodeBuild workflow in the same org can use
#       `permissions: { packages: read }`).
#
#   --build-arg BUILD_MODE=vanilla
#       Inline the full base stack (EFA userspace + NCCL + aws-ofi-nccl +
#       GDRCopy + DeepEP V2 + patches) directly on top of
#       nvidia/cuda:12.9.0-devel-ubuntu24.04, byte-identical to the base repo's
#       Dockerfile at https://github.com/antonai-work/deepep-v2-efa-base.
#       Use when you cannot reach GHCR or want end-to-end reproducibility
#       from the vanilla NVIDIA CUDA image.
#
# Both paths yield identical runtime layers on top of an identical base stack;
# the framework layer (Megatron-LM + NeMo-RL + patches) is single-sourced.
#
# Framework-layer pinned sources (both paths):
#   Megatron-LM    dmvevents/Megatron-LM@1aa3a7ec on branch
#                  deepep-v2-elasticbuffer-support (PR #4632 = vanilla
#                  Megatron 23dd639c + 3 Shape Y commits: ElasticBuffer
#                  support, EventOverlap fallback, num_experts-explicit
#                  backward dispatch). SHA bumped from cbacb0bbf after
#                  the branch was rebased on 2026-05-05; content is the
#                  same 3-commit stack.
#   NeMo-RL        NVIDIA-NeMo/RL@46be4e8 + patch 0007 (PR #2410,
#                  LD_LIBRARY_PATH fix for EFA+OFI discovery, closes #1973).
#   SGLang         no direct pin here; sibling repo sgl-project/sglang PR
#                  #24443 documents the V1-shim D+C contract for inference.
#   api-shim       NOT INCLUDED by design. DEEP_EP_USE_V2_SHIM=0 at runtime;
#                  Megatron calls deep_ep.ElasticBuffer natively via patches
#                  0004-0006.
#
# Build FROM REPO ROOT:
#   # Default fast path:
#   docker build -f docker/Dockerfile -t <tag> .
#   # Equivalent:
#   docker/build.sh --mode fast <tag>
#
#   # Vanilla path:
#   docker build --build-arg BUILD_MODE=vanilla -f docker/Dockerfile -t <tag> .
#   docker/build.sh --mode vanilla <tag>
#
# Preflight gate (inside running container):
#   docker run --rm <image> bash /opt/docker/preflight.sh
#   -> expected: "5/5 checks PASS"
#
# Runtime gate (2-node EFA cluster):
#   torchrun --nproc-per-node=8 --nnodes=2 ... /opt/tests/train_qwen3_moe.py
#   -> expected: monotonic loss decrease + EFA TX >= 1 GB
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
ARG BASE_IMAGE_VANILLA=nvidia/cuda:12.9.0-devel-ubuntu24.04

# =============================================================================
# Stage 1a: base-fast
#   Pull the pre-built base image from GHCR. Contains CUDA 12.9 + EFA 1.48.0 +
#   aws-ofi-nccl @ 6e504db + NCCL 2.30.4 + GDRCopy 2.5.1 + DeepEP V2 + patches
#   0001-0003. All LD paths and runtime env are already set.
# =============================================================================
FROM ${BASE_IMAGE_FAST} AS base-fast

# =============================================================================
# Stage 1b: base-vanilla
#   Byte-identical inline of antonai-work/deepep-v2-efa-base/Dockerfile so this
#   path can run without GHCR access. Do not edit below divergently from the
#   base repo -- fetch via:
#     gh api repos/antonai-work/deepep-v2-efa-base/contents/Dockerfile \
#       --jq '.content' | base64 -d
# =============================================================================
FROM ${BASE_IMAGE_VANILLA} AS base-vanilla

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# System deps + EFA userspace + aws-ofi-nccl (NGC-aware install path)
# -----------------------------------------------------------------------------
# EFA installer 1.48+ ships a --build-ngc path specifically for NVIDIA DL base
# containers: it installs libfabric1-aws + libfabric-aws-dev + libfabric-aws-bin
# debs plus the NGC-flavored libnccl-ofi-ngc-v3 deb. The NGC plugin is built
# with --disable-nccl-net-library so it is NCCL-version-agnostic.
#
# We deliberately do NOT install rdma-core / libibverbs-dev / ibverbs-providers
# from the noble apt repository. Those would land v50 and block the v61
# rdma-core bundled inside the EFA 1.48 tarball. libfabric1-aws hard-depends
# on ibverbs-providers >= 59 so the apt path cannot satisfy it.
ARG EFA_INSTALLER_VERSION=1.48.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget tar xz-utils file \
    autoconf automake libtool pkg-config cmake ninja-build patchelf \
    python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev openmpi-bin \
    libnuma-dev libnl-3-dev libnl-route-3-dev \
    kmod pciutils \
 && rm -rf /var/lib/apt/lists/*

# The 1.48 installer's NGC path runs plain `dpkg -i`. libfabric1-aws declares
# `Depends: ibverbs-providers (>= 59)` but noble ships 50.0. Manually installing
# the v61 debs from the tarball first (with --auto-deconfigure) upgrades the
# preinstalled v50 ibverbs-providers to v61 and then libfabric1-aws can drop
# in. This is the same sequence the NGC CUDA DL Base 26.03 image uses.
RUN set -eux; \
    mkdir -p /var/build-scratch/efa; \
    cd /var/build-scratch/efa; \
    curl -fsSL -o aws-efa-installer.tar.gz \
      "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
    tar -xzf aws-efa-installer.tar.gz; \
    cd aws-efa-installer; \
    arch="$(uname -m)"; \
    dpkg -i --auto-deconfigure \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/ibverbs-providers_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad3_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibmad5_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibnetdisc5_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric1-aws_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-bin_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-dev_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-config_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-profile_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libnccl-ofi-ngc-v3_*.deb; \
    rm -f /opt/amazon/efa/lib/libfabric.a; \
    test -f /opt/amazon/efa/include/rdma/fabric.h; \
    test -f /opt/amazon/efa/lib/libfabric.so.1; \
    test -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so; \
    strings /lib/x86_64-linux-gnu/libefa.so.1 | grep '^EFA_1\.4$' > /dev/null \
        || (echo "FATAL: libefa.so.1 missing EFA_1.4 symbol" && exit 1); \
    ldconfig; \
    rm -rf /var/build-scratch/efa

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64

# -----------------------------------------------------------------------------
# GDRCopy v2.5.1 userspace (kernel module lives on the host)
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --depth=1 --branch v2.5.1 \
      https://github.com/NVIDIA/gdrcopy.git /var/build-scratch/gdrcopy; \
    cd /var/build-scratch/gdrcopy; \
    CUDA=/usr/local/cuda make lib_install; \
    ldconfig; \
    rm -rf /var/build-scratch/gdrcopy

# -----------------------------------------------------------------------------
# PyTorch + NCCL 2.30.4 via pip (DeepEP V2 upstream README recommends this
# over building NCCL from source, so the JIT can find headers automatically)
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu129 \
 || pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128

# numpy<2 is required at runtime by torch.distributed.all_gather_object
# (DeepEP V2's ElasticBuffer.__init__ calls it from get_nccl_comm_handle).
# Without numpy<2 the NCCL comm handle exchange raises "Numpy is not available".
RUN pip install --no-cache-dir --break-system-packages \
      "numpy<2"

# Purge the apt-installed NCCL 2.26.x that the CUDA base image ships. Left in
# place, /lib/x86_64-linux-gnu/libnccl.so.2 (2.26.5) shadows the pip wheel's
# 2.30.4 via ld.so. DeepEP V2 requires >=2.30.4 (Gin backend API).
RUN set -eux; \
    apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true; \
    apt-mark hold libnccl2 libnccl-dev 2>/dev/null || true; \
    rm -f /lib/x86_64-linux-gnu/libnccl.so* /usr/lib/x86_64-linux-gnu/libnccl.so*; \
    ldconfig

RUN set -eux; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      "nvidia-nccl-cu13>=2.30.4"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    echo "NCCL_LIB=${NCCL_LIB}"; \
    test -n "${NCCL_LIB}"; \
    ls -la "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    echo "SO2=${SO2}"; \
    test -n "${SO2}"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ls -la "${NCCL_LIB}/libnccl.so" "${NCCL_LIB}/libnccl.so.2" || true; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    ldconfig; \
    ldconfig -p | grep 'libnccl.so' | head -3

# NVSHMEM pip wheel for the legacy link path. DeepEP V2's setup.py still links
# against libnvshmem_host.so even when the primary runtime is NCCL Gin
# (upstream setup.py has a TODO: "make NVSHMEM and legacy optional").
# Required >=3.3.9 per DeepEP docs/nvshmem.md on the V2 branch.
# The wheel only ships libnvshmem_host.so.X (versioned), but DeepEP's link
# line uses `-l:libnvshmem_host.so` (unversioned). Create the symlink.
RUN pip install --no-cache-dir --break-system-packages \
      "nvidia-nvshmem-cu12>=3.3.9" \
 && NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)" \
 && test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}" \
 && ls "${NVSHMEM_LIB}" \
 && HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)" \
 && ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so" \
 && echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf \
 && ldconfig

# -----------------------------------------------------------------------------
# aws-ofi-nccl built from master, pinned to commit 6e504db3403931cde (2026-04-24):
# "gin: Size active_put_signal to full sequence number space". Official upstream
# fix for the 128-slot GIN ring overflow that caused DeepEP V2 dispatch to hang
# with "received count: 0" on AWS EFA once num_allocated_qps >= 5. Supersedes
# our earlier out-of-tree ring-size patch.
#
# Must be built AFTER nvidia-nccl-cu13 pip install so --with-nccl finds
# matching 2.30.4 headers. The plugin's ncclGinPlugin_v11/v12 symbols must
# match the NCCL it is loaded alongside, or DeepEP's railedGinType check
# (see csrc/kernels/backend/nccl.cu in DeepEP V2) fails.
#
# Installed alongside the installer-bundled NGC plugin at /opt/amazon/ofi-nccl.
# The runtime NCCL_NET_PLUGIN env points at /opt/amazon (NGC path) because
# it is built --disable-nccl-net-library and therefore NCCL-version-agnostic;
# /opt/aws-ofi-nccl is retained for overlays that need --with-nccl behaviour.
# -----------------------------------------------------------------------------
ARG AWS_OFI_NCCL_SHA=6e504db3403931cde43a2335adcc73fbc69cccac
RUN set -eux; \
    git clone https://github.com/aws/aws-ofi-nccl.git /var/build-scratch/aws-ofi-nccl; \
    cd /var/build-scratch/aws-ofi-nccl; \
    git checkout "${AWS_OFI_NCCL_SHA}"; \
    ./autogen.sh; \
    NCCL_INCLUDE_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/include' -type d 2>/dev/null | head -1)"; \
    NCCL_LIB_DIR_OFI="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_INCLUDE_DIR}" && test -n "${NCCL_LIB_DIR_OFI}"; \
    NCCL_ROOT="$(dirname "${NCCL_INCLUDE_DIR}")"; \
    CPPFLAGS="-I/opt/amazon/efa/include -I/opt/amazon/efa/include/rdma -I/usr/local/cuda/include -I${NCCL_INCLUDE_DIR}" \
    LDFLAGS="-L/opt/amazon/efa/lib -L/opt/amazon/efa/lib64 -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L${NCCL_LIB_DIR_OFI}" \
    ./configure \
      --prefix=/opt/aws-ofi-nccl \
      --with-libfabric=/opt/amazon/efa \
      --with-cuda=/usr/local/cuda \
      --with-nccl="${NCCL_ROOT}" \
      --enable-platform-aws \
      --disable-tests; \
    make -j"$(nproc)"; \
    make install; \
    nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin; \
    echo /opt/aws-ofi-nccl/lib > /etc/ld.so.conf.d/zz-aws-ofi-nccl.conf; \
    ldconfig; \
    rm -rf /var/build-scratch/aws-ofi-nccl

# -----------------------------------------------------------------------------
# DeepEP V2 with PR #612 (auto-QP cap + get_rdma_gbs EFA fast path + dispatch
# kScaleoutUpdateInterval 3 -> 16). See patches/README.md for the three-patch
# rationale and upstream PR link.
#
# We clone the pre-patched branch rather than `git am` the three patches from
# ./patches/ for two reasons: (1) the branch is the exact tree tested in the
# reference 2-node p5en.48xlarge H200 EFA run that produced 740us D+C p50,
# which the patches are extracted from; (2) it avoids any risk of `git am`
# conflicts against a moving upstream base (the patches were generated
# against b306af0 and are byte-for-byte identical with the branch tip).
# When PR #612 merges upstream, flip DEEPEP_FORK/DEEPEP_BRANCH to vanilla and
# drop the ./patches/ directory.
# -----------------------------------------------------------------------------
ARG DEEPEP_FORK=https://github.com/dmvevents/DeepEP-1.git
ARG DEEPEP_BRANCH=aws-efa-auto-qp-cap-v2
ARG DEEPEP_SHA=c84dcac613c8df743a6487a312bbc966c745c600

RUN set -eux; \
    git clone --recurse-submodules "${DEEPEP_FORK}" /opt/DeepEP; \
    cd /opt/DeepEP; \
    git checkout "${DEEPEP_BRANCH}"; \
    # Verify the pinned SHA is at the branch tip for reproducibility.
    test "$(git rev-parse HEAD)" = "${DEEPEP_SHA}" \
      || (echo "FATAL: DeepEP HEAD $(git rev-parse HEAD) != pinned ${DEEPEP_SHA}" && exit 1); \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    # DeepEP V2 setup.py adds -Wl,-rpath for nccl but not a matching -L, so the
    # linker cannot resolve `-l:libnccl.so`. Inject the path via LIBRARY_PATH.
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"

# Runtime env matches the reference bench_elastic_ep.sh from the V2 recipe.
# NCCL_NET_PLUGIN points at the --build-ngc install path (installer-bundled,
# NCCL-version-agnostic). The PR #612 ring fix in aws-ofi-nccl 6e504db
# removes the need for the old OFI_NCCL_GIN_MAX_REQUESTS knob but it is
# retained as defence in depth for any older plugin loaded in override.
ENV NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_CUMEM_HOST_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_IGNORE_DISABLED_P2P=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    FI_EFA_FORK_SAFE=1 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

WORKDIR /opt

LABEL org.opencontainers.image.title="deepep-v2-efa-base" \
      org.opencontainers.image.description="DeepEP V2 (NCCL Gin) on AWS EFA, SM 9.0a base image for H100/H200" \
      org.opencontainers.image.source="https://github.com/antonai-work/deepep-v2-efa-base" \
      org.opencontainers.image.licenses="Apache-2.0" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="c84dcac613c8df743a6487a312bbc966c745c600" \
      aws-ofi-nccl.sha="6e504db3403931cde43a2335adcc73fbc69cccac" \
      efa.installer="1.48.0"

# =============================================================================
# Stage 2: release
#   Adds the NeMo-RL framework layer (Megatron-LM + NeMo-RL + patches) on top
#   of whichever base stage was selected. Single-sourced so both paths yield
#   identical framework content.
# =============================================================================
FROM base-${BUILD_MODE} AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# Megatron-LM with patches 0004-0006 (Shape Y: DeepEP V2 ElasticBuffer support)
# Layout matches PR #4632: 3 commits on top of NVIDIA/Megatron-LM@23dd639c.
# Pinned to the SHA NeMo-RL 46be4e8 expects via its submodule.
# -----------------------------------------------------------------------------
ARG MEGATRON_FORK=https://github.com/dmvevents/Megatron-LM.git
ARG MEGATRON_BRANCH=deepep-v2-elasticbuffer-support
ARG MEGATRON_SHA=1aa3a7ec76f255ce23cbcc94dbd8a7d49dcbb49d

RUN set -eux; \
    git clone "${MEGATRON_FORK}" /opt/Megatron-LM; \
    cd /opt/Megatron-LM; \
    git checkout "${MEGATRON_BRANCH}"; \
    test "$(git rev-parse HEAD)" = "${MEGATRON_SHA}" \
      || (echo "Megatron SHA drift: expected ${MEGATRON_SHA}, got $(git rev-parse HEAD)" && exit 1); \
    pip install --no-cache-dir --break-system-packages --no-deps -e .; \
    pip install --no-cache-dir --break-system-packages packaging; \
    python3 -c "from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2; \
                assert HAVE_DEEP_EP_V2, 'HAVE_DEEP_EP_V2 probe failed'; \
                print('Megatron Shape Y OK: HAVE_DEEP_EP_V2 =', HAVE_DEEP_EP_V2)"

# -----------------------------------------------------------------------------
# NeMo-RL @ 46be4e8 with patch 0007 (LD_LIBRARY_PATH fix, NeMo-RL PR #2410)
# -----------------------------------------------------------------------------
ARG NEMORL_REPO_URL=https://github.com/NVIDIA-NeMo/RL.git
ARG NEMORL_GIT_REF=46be4e8

RUN set -eux; \
    git clone "${NEMORL_REPO_URL}" /opt/NeMo-RL; \
    cd /opt/NeMo-RL; \
    git fetch origin "${NEMORL_GIT_REF}" || true; \
    git checkout "${NEMORL_GIT_REF}"; \
    git log -1 --oneline

COPY patches/0007-nemo-rl-ld-library-path.patch /tmp/nemo-rl-patch.patch
RUN set -eux; \
    cd /opt/NeMo-RL; \
    git apply --verbose /tmp/nemo-rl-patch.patch; \
    ls -l examples/configs/recipes/llm/aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml; \
    rm /tmp/nemo-rl-patch.patch

# NeMo-RL pure-Python deps.
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
# NO-SHIM INVARIANT: this image explicitly has no /opt/api-shim and no
# sitecustomize.py from a shim. The preflight step verifies this.
# -----------------------------------------------------------------------------
RUN set -eux; \
    test ! -e /opt/api-shim; \
    find / -name "sitecustomize.py" -path "*/api_shim/*" -exec rm -v {} \; 2>/dev/null || true; \
    find / -name "sitecustomize.py" -path "*/api-shim/*" -exec rm -v {} \; 2>/dev/null || true

ENV PYTHONPATH=/opt/Megatron-LM:/opt/NeMo-RL

# LD_LIBRARY_PATH: make both OFI plugin locations discoverable so either
# the source-built plugin (/opt/aws-ofi-nccl, selected by NCCL_NET_PLUGIN
# below) or the installer-bundled NGC plugin (/opt/amazon/ofi-nccl)
# resolves its own deps. /opt/amazon/aws-ofi-nccl does NOT exist in the
# base image; the prior path was a typo (god-mode F2, fixed 2026-05-05).
ENV LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Release-image runtime env: DEEP_EP_USE_V2_SHIM=0 (no shim, native V2), and
# the baseline DeepEP MoE envs in case a consumer only inherits the release
# image without re-sourcing the base env. Base image values are preserved
# where unchanged.
ENV DEEP_EP_USE_V2_SHIM=0 \
    NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

# Bake tests and preflight into the image for one-shot validation.
COPY docker/preflight.sh /opt/docker/preflight.sh
COPY tests/train_qwen3_moe.py /opt/tests/train_qwen3_moe.py
COPY tests/verify_efa_traffic.sh /opt/tests/verify_efa_traffic.sh
RUN chmod +x /opt/docker/preflight.sh /opt/tests/verify_efa_traffic.sh

WORKDIR /opt

LABEL org.opencontainers.image.title="nemo-rl-deepep-v2-efa" \
      org.opencontainers.image.description="All-PRs-applied E2E image: DeepEP V2 PR #612 + Megatron-LM PR #4632 + NeMo-RL PR #2410. Vanilla upstream + patches, no shim." \
      org.opencontainers.image.source="https://github.com/antonai-work/nemo-rl-deepep-v2-efa" \
      org.opencontainers.image.licenses="Apache-2.0" \
      megatron.branch="${MEGATRON_BRANCH}" \
      megatron.sha="${MEGATRON_SHA}" \
      nemo_rl.git_ref="${NEMORL_GIT_REF}" \
      shim.included="false"
