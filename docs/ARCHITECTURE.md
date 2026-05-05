# Architecture

## The build chain

One multi-stage Dockerfile produces the final image. Each stage
layers a specific upstream dependency with any patches applied.

```
┌─────────────────────────────────────────────────────┐
│ FROM nvidia/cuda:12.9.0-devel-ubuntu24.04           │   Public NVIDIA mirror on Docker Hub.
│ Public CUDA 12.9 developer base                     │   Byte-identical to nvcr.io/nvidia/cuda.
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ EFA stack                                           │
│  aws-efa-installer 1.48.0 (--build-ngc flag)        │
│   + rdma-core 61.0 libraries                        │
│   + libfabric 2.4.0amzn (EFA-patched)               │
│   + aws-ofi-nccl 1.19.0 (installer-bundled NGC)     │
│  NCCL 2.30.4 via `pip install nvidia-nccl-cu13`     │
│  GDRCopy 2.5.1                                      │
│  NVSHMEM wheel (link-only; not used at runtime)     │
│  aws-ofi-nccl source-built at 6e504db (2026-04-24)  │
│   — upstream fix for GIN ring overflow,             │
│     strictly better than our closed PR #1206        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ DeepEP V2 (patched)                                 │
│  Clone: deepseek-ai/DeepEP@b306af0 (main)           │
│   or dmvevents/DeepEP-1@aws-efa-auto-qp-cap          │
│   (same content after rebase)                       │
│  Patches 0001, 0002, 0003 from PR #612 applied.     │
│  pip install -e . against the NCCL 2.30.4 above.    │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Megatron-LM (patched)                               │
│  Clone: NVIDIA/Megatron-LM@23dd639c                 │
│   (the SHA NeMo-RL 46be4e8 pins via submodule)      │
│  Patches 0004, 0005, 0006 applied.                  │
│  Installs megatron.core in-place.                   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ NeMo-RL (patched)                                   │
│  Clone: NVIDIA-NeMo/RL@46be4e8                      │
│  Patch 0007 applied (LD_LIBRARY_PATH fix).          │
│  Installs NeMo-RL package.                          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
          Final image (~15 GB compressed)
```

## The data flow at runtime

```
NeMo-RL training driver (train_qwen3_moe.py)
      │
      │ invokes Megatron-Core training loop
      ▼
Megatron-LM _DeepepManager
  (patched: probes HAVE_DEEP_EP_V2, calls ElasticBuffer instead of Buffer)
      │
      │ buffer.dispatch(x, topk_idx, topk_weights, num_experts=N, ...)
      ▼
DeepEP V2 ElasticBuffer
  (patched: num_allocated_qps=2 on EFA, EFA fast path in get_rdma_gbs,
   kScaleoutUpdateInterval=16)
      │
      │ NCCL GIN collective API (ncclTeamTagRail etc.)
      ▼
NCCL 2.30.4 (pip cu13)
      │
      │ net plugin dispatch
      ▼
aws-ofi-nccl git-6e504db
  (source-built; contains active_put_signal bitset fix)
      │
      │ libfabric EFA provider
      ▼
AWS EFA hardware (rdmap*s0 NICs, SRD transport)
      │
      ▼
Cross-node GPU-to-GPU RDMA over EFA
```

## Key design decisions

### Why no V1 → V2 shim?

Earlier iterations of this work used a Python-level
V1-Buffer-to-V2-ElasticBuffer shim (see
`integrations/api-shim/buffer_v1_compat.py` in the private
development repo). The Megatron-LM patches (0004-0006) make the
shim unnecessary by teaching Megatron to call V2 natively. The
public repo does NOT ship a shim — Dockerfile explicitly asserts
`DEEP_EP_USE_V2_SHIM=0` at runtime, and preflight verifies no
`/opt/api-shim` directory exists.

### Why `b306af0` for DeepEP?

That's the merge commit of PR #605 (DeepEP V2 public release,
2026-04-29). Any `main` commit at or after this point contains V2.
Our fork branch `aws-efa-auto-qp-cap` is rebased on top of
`b306af0` so reviewers can see the delta as exactly 3 commits.

### Why `23dd639c` for Megatron-LM?

That's the SHA NeMo-RL 46be4e8 pins via git submodule
(`.gitmodules`). We don't bump Megatron beyond that because NeMo-RL
hasn't tested newer Megatron pins. Stable target.

### Why `46be4e8` for NeMo-RL?

That's the last-known-good NeMo-RL SHA that pulls our exact
Megatron pin. Picked because:
- The config surface for `moe_enable_deepep` /
  `moe_token_dispatcher_type=flex` already exists at this SHA (PR
  #1794 landed 2026-01-20)
- The submodule pin matches our Megatron-LM target
- The LD_LIBRARY_PATH issue is still present (ours to fix)

### Why NCCL 2.30.4, not newer?

DeepEP V2's `csrc/kernels/backend/nccl.cu` uses the
`ncclTeamTagRail` API which requires ≥2.30.4. Newer NCCL works;
older does not. The pip wheel `nvidia-nccl-cu13>=2.30.4` gets the
right version.

### Why build aws-ofi-nccl from source AND install the NGC plugin?

Two plugins coexist:
- `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` — installer's NGC
  plugin (aws-ofi-nccl 1.19.0, pre-built for NGC)
- `/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` — source-built at
  6e504db (includes the GIN ring fix)

The K8s manifest's `NCCL_NET_PLUGIN` env points at the NGC plugin
by default. For DeepEP V2, you want the source build (6e504db).
Both are present; the manifest's env var selects which one.

### Why FSX for weights?

Qwen3-30B-A3B is ~30 GB. Downloading from HuggingFace on every
pod restart is ~15 minutes. FSX-backed PVC caches the weights
once across the cluster. Optional (you can use a different
storage backend); required path in the reference manifest.

## Host requirements

- NVIDIA GPUs with CUDA 9.0+ compute capability (H100 or H200
  recommended; A100 should work but untested)
- EFA-enabled instance type (p4d, p5, p5en)
- Linux kernel ≥ 5.15 with EFA 2.x driver
- `nvidia-device-plugin` and EFA K8s device plugin on EKS

## What's NOT in this repo

- Pre-built Docker images (you build from source)
- Model weights (you stage Qwen3-30B-A3B separately; see
  `tests/k8s/multi-node-training-h100.yaml` for the expected
  FSX layout)
- Cluster provisioning (you bring your own EKS + p5.48xlarge
  nodegroup; reference manifests assume HyperPod EKS but work on
  vanilla EKS with EFA plugin)
- Performance tuning beyond the three DeepEP patches (no NCCL
  topology tuning, no NUMA-affinity scripts, etc. — those are
  workload-specific)
