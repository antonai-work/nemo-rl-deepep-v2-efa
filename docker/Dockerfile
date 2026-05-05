# syntax=docker/dockerfile:1.10
# =============================================================================
# nemo-rl-deepep-v2-efa - all-PRs-applied, vanilla-upstream, self-contained
# =============================================================================
#
# One self-contained Dockerfile that layers all three of our upstream PRs on
# top of pure-upstream substrate. Builds from the vanilla public NVIDIA CUDA
# base image; no pre-built base image required.
#
# Pinned layers:
#   Base OS + CUDA       nvidia/cuda:12.9.0-devel-ubuntu24.04   vanilla
#   EFA userspace        aws-efa-installer 1.48.0               vanilla (--build-ngc)
#   aws-ofi-nccl         6e504db3403931cde43a2335adcc73fbc69cccac  vanilla
#                        ("gin: Size active_put_signal to full sequence
#                         number space", upstream fix 2026-04-24 that
#                         superseded our closed PR #1206)
#   NCCL                 nvidia-nccl-cu13 >= 2.30.4             vanilla
#   GDRCopy              v2.5.1                                 vanilla
#   DeepEP V2            dmvevents/DeepEP-1 branch              OUR FORK
#                        aws-efa-auto-qp-cap-v2 @ c84dcac       = vanilla V2 +
#                        PR #612's 3 commits (auto-QP cap, EFA gbs fast path,
#                        kScaleoutUpdateInterval tuning)
#   Megatron-LM          dmvevents/Megatron-LM branch           OUR FORK
#                        deepep-v2-elasticbuffer-support         = vanilla
#                        Megatron (23dd639c) + Shape Y patch (cbacb0bbf,
#                        3 commits: ElasticBuffer support, EventOverlap
#                        fallback, num_experts-explicit backward dispatch)
#   NeMo-RL              NVIDIA-NeMo/RL@46be4e8 + patch 0007     OUR PATCH
#                        patches/0007-nemo-rl-ld-library-path.patch = vanilla
#                        + LD_LIBRARY_PATH fix for EFA+OFI discovery
#                        (NeMo-RL PR #2410), closes issue #1973
#   api-shim             NOT INCLUDED                            by design
#                        DEEP_EP_USE_V2_SHIM=0 at runtime; Megatron calls
#                        deep_ep.ElasticBuffer natively via patches 0004-0006.
#
# Build FROM REPO ROOT:
#   docker build -f docker/Dockerfile -t <your-ecr>/<repo>:<tag> .
# or equivalently:
#   docker/build.sh <your-ecr>/<repo>:<tag>
#
# Preflight gate (inside running container):
#   docker run --rm <image> bash /opt/docker/preflight.sh
#   -> expected: "5/5 checks PASS"
#
# Runtime gate (2-node EFA cluster):
#   torchrun --nproc-per-node=8 --nnodes=2 ... /opt/tests/train_qwen3_moe.py
#   -> expected: monotonic loss decrease + EFA TX >= 1 GB
# =============================================================================

FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# System deps (build toolchain + openmpi + numa/libnl for rdma-core)
# Intentionally NOT installing rdma-core/libibverbs-dev from apt -- the EFA
# installer below lands matched v61 .debs; apt v50 would block them.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget tar xz-utils file \
    autoconf automake libtool pkg-config cmake ninja-build patchelf \
    python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev openmpi-bin \
    libnuma-dev libnl-3-dev libnl-route-3-dev \
    kmod pciutils \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# EFA userspace (aws-efa-installer 1.48.0, --build-ngc path)
# Manually dpkg -i --auto-deconfigure to upgrade ibverbs-providers v50 -> v61
# since libfabric1-aws 2.4.0amzn3.0 declares Depends: ibverbs-providers >= 59.
# Runtime ABI is stable across these versions; it's a packaging-metadata gap.
# -----------------------------------------------------------------------------
ARG EFA_INSTALLER_VERSION=1.48.0
RUN set -eux; \
    cd /tmp; \
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
    rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer.tar.gz

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64

# -----------------------------------------------------------------------------
# GDRCopy 2.5.1 userspace (host ships the kernel module)
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --depth=1 --branch v2.5.1 \
      https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy; \
    cd /tmp/gdrcopy; \
    CUDA=/usr/local/cuda make lib_install; \
    ldconfig; \
    rm -rf /tmp/gdrcopy

# -----------------------------------------------------------------------------
# PyTorch + NCCL 2.30.4 via pip (DeepEP V2 JIT finds NCCL headers here)
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu129 \
 || pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128

# numpy<2 required by torch.distributed.all_gather_object (used by V2 init).
RUN pip install --no-cache-dir --break-system-packages "numpy<2"

# Purge apt-installed libnccl2 2.26.x that would shadow the 2.30.4 pip wheel.
RUN set -eux; \
    apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true; \
    apt-mark hold libnccl2 libnccl-dev 2>/dev/null || true; \
    rm -f /lib/x86_64-linux-gnu/libnccl.so* /usr/lib/x86_64-linux-gnu/libnccl.so*; \
    ldconfig

RUN set -eux; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      "nvidia-nccl-cu13>=2.30.4"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    ldconfig

# NVSHMEM wheel — DeepEP V2's setup.py still links against libnvshmem_host.so
# even when NCCL Gin is the primary path (upstream TODO).
RUN pip install --no-cache-dir --break-system-packages \
      "nvidia-nvshmem-cu12>=3.3.9" \
 && NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)" \
 && test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}" \
 && HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)" \
 && ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so" \
 && echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf \
 && ldconfig

# -----------------------------------------------------------------------------
# aws-ofi-nccl built from source at 6e504db (2026-04-24):
#   "gin: Size active_put_signal to full sequence number space"
# Upstream fix for the 128-slot Gin ring overflow. Must build AFTER
# nvidia-nccl-cu13 so --with-nccl finds matching 2.30.4 headers.
# -----------------------------------------------------------------------------
ARG AWS_OFI_NCCL_SHA=6e504db3403931cde43a2335adcc73fbc69cccac
RUN set -eux; \
    git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl; \
    cd /tmp/aws-ofi-nccl; \
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
    rm -rf /tmp/aws-ofi-nccl

# -----------------------------------------------------------------------------
# DeepEP V2 with patches 0001-0003 (PR #612)
# Layout matches the PR: 3 commits on top of deepseek-ai/DeepEP@main
#   fe20874  aws-efa: cap auto-QP at 2
#   0b78333  aws-efa: EFA fast path in get_rdma_gbs
#   c84dcac  aws-efa: raise kScaleoutUpdateInterval 3->16
# When PR #612 merges upstream, flip to deepseek-ai/DeepEP.git@main and drop
# patches 0001-0003.
# -----------------------------------------------------------------------------
ARG DEEPEP_FORK=https://github.com/dmvevents/DeepEP-1.git
ARG DEEPEP_BRANCH=aws-efa-auto-qp-cap-v2
ARG DEEPEP_SHA=c84dcac613c8df743a6487a312bbc966c745c600

RUN set -eux; \
    git clone --recurse-submodules "${DEEPEP_FORK}" /opt/DeepEP; \
    cd /opt/DeepEP; \
    git checkout "${DEEPEP_BRANCH}"; \
    test "$(git rev-parse HEAD)" = "${DEEPEP_SHA}" \
      || (echo "DeepEP SHA drift: expected ${DEEPEP_SHA}, got $(git rev-parse HEAD)" && exit 1); \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"

# -----------------------------------------------------------------------------
# Megatron-LM with patches 0004-0006 (Shape Y: DeepEP V2 ElasticBuffer support)
# Layout matches PR #4632: 3 commits on top of NVIDIA/Megatron-LM@23dd639c.
# Pinned to the SHA NeMo-RL 46be4e8 expects via its submodule.
# -----------------------------------------------------------------------------
ARG MEGATRON_FORK=https://github.com/dmvevents/Megatron-LM.git
ARG MEGATRON_BRANCH=deepep-v2-elasticbuffer-support
ARG MEGATRON_SHA=cbacb0bbf2a73ed28ffcfcff61beddd31ed6ea76

RUN set -eux; \
    git clone "${MEGATRON_FORK}" /opt/Megatron-LM; \
    cd /opt/Megatron-LM; \
    git checkout "${MEGATRON_BRANCH}"; \
    test "$(git rev-parse HEAD)" = "${MEGATRON_SHA}" \
      || (echo "Megatron SHA drift: expected ${MEGATRON_SHA}, got $(git rev-parse HEAD)" && exit 1); \
    pip install --no-cache-dir --break-system-packages --no-deps -e .; \
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

# LD_LIBRARY_PATH: patch 0007's exact ENV line (EFA + OFI discoverable).
ENV LD_LIBRARY_PATH="/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Runtime env (validates all three PRs + matches Shape Y training recipe).
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
      deepep.branch="${DEEPEP_BRANCH}" \
      deepep.sha="${DEEPEP_SHA}" \
      megatron.branch="${MEGATRON_BRANCH}" \
      megatron.sha="${MEGATRON_SHA}" \
      nemo_rl.git_ref="${NEMORL_GIT_REF}" \
      efa.installer="${EFA_INSTALLER_VERSION}" \
      aws_ofi_nccl.sha="${AWS_OFI_NCCL_SHA}" \
      shim.included="false"
