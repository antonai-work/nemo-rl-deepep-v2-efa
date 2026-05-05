# nemo-rl-deepep-v2-efa

**Reproducible DeepEP V2 MoE training on AWS EFA, built from vanilla
upstream sources + three open upstream PRs.**

This repo packages a complete multi-stage Docker build chain and a
Kubernetes manifest that produces a working Qwen3-30B-A3B MoE
training stack on 2× p5.48xlarge (or p5en.48xlarge) H100/H200 nodes
over AWS EFA — end-to-end validated 2026-05-05.

## What's inside

- **`patches/`** — seven standalone `.patch` files extracted from
  three upstream PRs we authored. Each patch has a header pointing
  at its open upstream PR.
- **`docker/Dockerfile`** — single multi-stage build from
  `nvidia/cuda:12.9.0-devel-ubuntu24.04` through EFA + aws-ofi-nccl
  + NCCL + GDRCopy + DeepEP V2 (patched) + Megatron-LM (patched) +
  NeMo-RL (patched).
- **`docker/build.sh`** — one-command build script.
- **`docker/preflight.sh`** — five in-container validation gates
  that prove the stack is assembled correctly before you spend
  cluster time running it.
- **`tests/train_qwen3_moe.py`** — the exact training driver that
  produced the validation evidence below (Qwen3-30B-A3B config,
  128 experts top-8, 2 MoE blocks, EP=16, seq_len=512).
- **`tests/k8s/multi-node-training-h100.yaml`** — 2-pod StatefulSet
  for EKS + EFA-enabled HyperPod (or vanilla EKS with EFA plugin).
- **`tests/verify_efa_traffic.sh`** — EFA counter snapshot + delta
  check that proves MoE traffic went over EFA, not NVLink.
- **`ci/`** — AWS CodeBuild spec for distribution-review-grade CI.
- **`docs/`** — deeper explanations of architecture, validation,
  and upstream status.

## Upstream PRs

Three PRs, one per upstream repo:

| Upstream repo | PR | Status | Patches |
|---|---|---|---|
| [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN, 3 commits | `patches/0001-0003` |
| [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) | _to be filed_ (see `docs/UPSTREAM-STATUS.md`) | READY | `patches/0004-0006` |
| [`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL) | _to be filed_ (resurrection of closed PR #2359, closes #1973) | READY | `patches/0007` |

All three are independent and safe on non-EFA fabrics. See
`docs/UPSTREAM-STATUS.md` for live tracking of merge state.

## Quick start

### Prerequisites
- Linux host with Docker and NVIDIA Container Toolkit
- 2× p5.48xlarge (H100) or p5en.48xlarge (H200) on AWS with EFA
  enabled and the EKS device plugin installed
- EKS cluster with kubectl access
- FSX for Lustre volume mounted at `/mnt/fsx` in both pods
  (required for Qwen3-30B-A3B weights; ~30 GB)
- An ECR repository for the built image

### Build

```bash
git clone https://github.com/antonai-work/nemo-rl-deepep-v2-efa
cd nemo-rl-deepep-v2-efa
docker/build.sh                          # ~45 min cold build
# Tag appears in build.sh output
```

The build script applies all 7 patches before compiling so you know
you're getting vanilla upstream + exactly the three PRs linked above.

### Preflight (in-container)

Before deploying, confirm the image assembled correctly:

```bash
docker run --rm <your-tag> bash docker/preflight.sh
```

Expected output: `5/5 checks PASS`.

### Deploy + run training

```bash
kubectl apply -f tests/k8s/multi-node-training-h100.yaml
kubectl -n nemo-rl-fullstack wait --for=condition=Ready pod --all --timeout=5m
kubectl -n nemo-rl-fullstack exec -it fullstack-0 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
  --master-addr=fullstack-0.fullstack-hs --master-port=29500 \
  /opt/tests/train_qwen3_moe.py
```

(Analogous command on the other pod with `--node-rank=1`.)

### Validate

Expected output includes:

```
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NCCL INFO NET/OFI Using transport protocol RDMA
HAVE_DEEP_EP_V2 = True
Active buffer class: ElasticBuffer
WARMUP  loss=28.5571  grad_norm=35.2123
STEP 1  loss=26.4074  grad_norm=30.6430
STEP 2  loss=25.0856  grad_norm=28.1979
STEP 3  loss=24.6252  grad_norm=27.0909
=== all-PRs-applied stack E2E training PASS ===
```

EFA traffic check:

```bash
kubectl -n nemo-rl-fullstack exec fullstack-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/before
# ... rerun the training command ...
kubectl -n nemo-rl-fullstack exec fullstack-0 -- \
  bash /opt/tests/verify_efa_traffic.sh verify /tmp/before /tmp/after 1
```

Expected: `EFA TX delta >= 1 GB (PASS)`.

## Reference validation (2026-05-05)

Measured on 2× p5.48xlarge H100 EFA with full multi-stage image:

| Gate | Observed |
|---|---|
| `HAVE_DEEP_EP_V2=True` | PASS |
| `Active buffer class: ElasticBuffer` (V2 native, no shim) | PASS |
| Loss monotonic decrease | 28.5571 → 24.6252 |
| `grad_norm` real, finite | 35.2 → 27.1 |
| EFA TX per pod | 1.096 GB (≥ 1 GB gate) |
| NCCL transport | `efa-direct` with 32 NICs |

Full per-step log lines are quoted in the three upstream PR bodies
so reviewers without AWS access can verify the traceability.

## Why a separate public repo?

The three PRs in the upstream repos above are independent — each
can be merged on its own schedule. But getting them to work
together requires all three applied simultaneously to matched
versions of their dependencies. This repo is the single source of
truth for "what version of everything" and "how do I assemble it
into a working image."

When all three PRs merge upstream, this repo's build chain reduces
to vanilla clones (no patches needed). Until then, the patches in
`patches/` let anyone reproduce the validated stack today.

## License

Apache 2.0. Patches under `patches/` inherit the license of their
upstream repositories (Apache 2.0 for all three).

## Related repos and references

- Upstream DeepEP V2: https://github.com/deepseek-ai/DeepEP
- Upstream Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Upstream NeMo-RL: https://github.com/NVIDIA-NeMo/RL
- AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- NVIDIA Megatron-LM issue #2647 (EFA feature request):
  https://github.com/NVIDIA/Megatron-LM/issues/2647
- NVIDIA Megatron-LM issue #3999 (HybridEP `max_num_tokens_per_rank`
  bug, closed by PR Shape Y on the V2 branch):
  https://github.com/NVIDIA/Megatron-LM/issues/3999
- NVIDIA-NeMo/RL issue #1973 (LD_LIBRARY_PATH for EFA OFI discovery):
  https://github.com/NVIDIA-NeMo/RL/issues/1973
