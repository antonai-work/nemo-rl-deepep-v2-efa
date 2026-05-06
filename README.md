# nemo-rl-deepep-v2-efa

**Reproducible DeepEP V2 MoE training on AWS EFA, built from vanilla
upstream sources + three open upstream PRs.**

**Status:** Dual-path Dockerfile verified locally 2026-05-06 (fast via GHCR base + vanilla inline); CodeBuild CI configured; 3-step training loss 21.57→20.05 validated on 2× p5.48xlarge H100. Last validated 2026-05-06 Wave 13.

This repo packages a complete multi-stage Docker build chain and a
Kubernetes manifest that produces a working Qwen3-30B-A3B MoE
training stack on 2× p5.48xlarge (or p5en.48xlarge) H100/H200 nodes
over AWS EFA — end-to-end validated 2026-05-06.

## Sibling repos (reproducibility triad)

| Repo | Purpose | Status |
|---|---|---|
| [deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base) | Base substrate (EFA + NCCL + DeepEP V2) | v0.2.1-sm90a released |
| [nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) | Training stack (this repo) | Dual-path build verified 2026-05-06 |
| [vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa) | Inference stack (vLLM + TRT-LLM) | Dual-path build verified 2026-05-06 |

Together, these three repos provide end-to-end DeepEP V2 MoE reproducibility on AWS EFA, from base substrate through training and inference.

## What's inside

- **`patches/`** — seven standalone `.patch` files extracted from
  three upstream PRs we authored. Each patch has a header pointing
  at its open upstream PR.
- **`docker/Dockerfile`** — dual-path multi-stage build. Default
  (fast) path pulls the pre-built base image from GHCR and adds the
  NeMo-RL framework layer in ~5-10 minutes. Opt-in vanilla path
  inlines the full 240-line base stack from
  `nvidia/cuda:12.9.0-devel-ubuntu24.04` through EFA + aws-ofi-nccl
  + NCCL + GDRCopy + DeepEP V2 (patched) before the framework layer.
  See [Fast vs From-vanilla build](#fast-vs-from-vanilla-build) below.
- **`docker/build.sh`** — one-command build script with
  `--mode fast|vanilla` flag.
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
  and upstream status. See
  [`docs/EFA-TRAFFIC-EVIDENCE.md`](docs/EFA-TRAFFIC-EVIDENCE.md)
  for the aggregated cross-node EFA hardware-counter proof across
  all frameworks.

## Validation

Wave 13 cu13-unified stack validation: [`docs/VALIDATION-WAVE13-CU13-TRAINING-PASS.md`](docs/VALIDATION-WAVE13-CU13-TRAINING-PASS.md). Captures Shape Y 3-step training success with monotonic loss, NVSHMEM ABI resolution, and bash quoting bug fix. Image digest: `sha256:ae4855b9a8b173a48171f8b3da672c688d8c1e820c0301a0cb39777cccaea9f8`.

## Upstream PRs

Five PRs filed 2026-04-28 through 2026-05-05, covering the full training + inference stack. All five are independent, EFA-specific, and safe on non-EFA fabrics:

| Upstream repo | PR | Status (2026-05-06) | Applies to |
|---|---|---|---|
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN, mergeable | Base substrate (all frameworks) |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [#4632](https://github.com/NVIDIA/Megatron-LM/pull/4632) | OPEN, mergeable | Training (this repo) |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410) | OPEN, mergeable | Training (this repo) |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [#2411](https://github.com/NVIDIA-NeMo/RL/pull/2411) | OPEN, mergeable | Training (this repo) |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | [#24443](https://github.com/sgl-project/sglang/pull/24443) | OPEN, mergeable | Inference (sibling repo) |

Plus: [vllm-project/vllm#41183](https://github.com/vllm-project/vllm/pull/41183) augmented with EFA traffic evidence via comment (OPEN, actively reviewed, inference sibling repo).

This repo consumes `patches/0001-0007` (DeepEP PR #612 + Megatron-LM #4632 + NeMo-RL #2410/#2411). See `docs/UPSTREAM-STATUS.md` for live tracking of merge state and detailed commit lists.

This repo consumes the pre-built base image published from the sibling repo [`antonai-work/deepep-v2-efa-base`](https://github.com/antonai-work/deepep-v2-efa-base) for the fast build path. That repo carries the same `patches/0001-0003` for DeepEP PR #612; the fast path simply skips re-compiling them.

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

# Default (fast) path -- FROM ghcr.io/antonai-work/deepep-v2-efa-base:
docker/build.sh --mode fast nemo-rl-deepep-v2-efa:latest       # ~5-10 min

# From-vanilla path -- inline 240-line base stack, fully reproducible:
docker/build.sh --mode vanilla nemo-rl-deepep-v2-efa:vanilla   # ~45 min
```

Both modes produce byte-identical framework content (Megatron-LM +
NeMo-RL + patches) on top of byte-identical base stacks. The vanilla
mode inlines the full base recipe so GHCR is not required at build
time. See [Fast vs From-vanilla build](#fast-vs-from-vanilla-build)
for the full trade-off.

The build script applies all 7 patches before compiling (vanilla mode)
or relies on the base image's pre-applied DeepEP `patches/0001-0003`
(fast mode), then adds `patches/0004-0007`. Either way you get vanilla
upstream + exactly the five PRs linked above.

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

## Fast vs From-vanilla build

`docker/Dockerfile` supports two build modes. Both select their base
stack via Docker BuildKit's `BUILD_MODE` ARG + multi-stage `FROM
base-${BUILD_MODE}` pattern, then add an identical framework layer on
top. Pick based on your network posture and build-time budget.

| Aspect | `--mode fast` (default) | `--mode vanilla` |
|---|---|---|
| Base image | `FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a` | Inline 240-line stack from `nvidia/cuda:12.9.0-devel-ubuntu24.04` |
| Build time | ~5-10 min (framework layer only) | ~45 min cold (compiles aws-ofi-nccl + DeepEP + Megatron + NeMo-RL) |
| Network needs | GHCR pull (PAT with `read:packages` until the package flips public) | Docker Hub + efa-installer.amazonaws.com + GitHub clones only |
| Reproducibility | Trusts the base image's SHA-digest | Fully reproducible from Dockerfile + repo contents |
| Patches applied | `patches/0001-0003` baked into base; `patches/0004-0007` added here | All `patches/0001-0007` applied in this Dockerfile |
| Use case | Iteration, CI, fast cluster deploy | Air-gapped builds, security review, first-time reproduction |

Both paths pass the same `docker/preflight.sh` gate (5/5 checks). The
AWS CodeBuild pipeline in `ci/buildspec.yml` exercises both modes on
every commit so a drift between them is caught by CI.

The base image digest pinned by the fast path is:

```
ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
  @ sha256:5083af841d926f63ff1eb98bdded6e3e23854330feabb53c9d910fff4899587c
```

The base repo is at
[`antonai-work/deepep-v2-efa-base`](https://github.com/antonai-work/deepep-v2-efa-base).

## Continuous integration

An AWS CodeBuild pipeline lives in `ci/`:

- `ci/buildspec.yml` — runs the fast-path build, runs the vanilla-path
  build as a drift check, runs `docker/preflight.sh` inside both
  images (must emit `5 PASS, 0 FAIL`), then pushes the fast tag to
  ECR. Uses `BUILD_GENERAL1_2XLARGE` with `privilegedMode=true` on
  `aws/codebuild/amazonlinux2-x86_64-standard:5.0`.
- [`ci/CODEBUILD-SETUP.md`](ci/CODEBUILD-SETUP.md) — step-by-step
  provisioning: ECR repo creation, IAM role trust + inline policy,
  Secrets Manager entry for the GHCR PAT, `aws codebuild
  create-project` invocation, and the webhook command to auto-build
  on pushes to `main`. All account IDs, region, and repo names are
  templated -- no hardcoded values.

To trigger a build once the project is provisioned:

```bash
aws codebuild start-build --project-name <PROJECT_NAME>
```

## Benchmarking

If you want to isolate DeepEP D+C performance from end-to-end training,
run the upstream DeepEP microbenchmarks inside the same image. The full
dispatch+combine harness (`tests/elastic/test_ep.py`) gives you the
~740 us p50 baseline on 2-node p5.48xlarge, and the low-latency harness
(`tests/legacy/test_low_latency.py`) exercises the decoding-phase
kernel. Both are covered in [`docs/DEEPEP-BENCHMARKS.md`](docs/DEEPEP-BENCHMARKS.md),
including 2-node `kubectl exec` invocations, the flags that actually
matter, and how to read the output against the `EP_EFA_MAX_QPS=2` /
`EP_EFA_RDMA_GBS=25.0` envs baked into the image.

## Why a separate public repo?

The five upstream PRs linked above are independent — each can be
merged on its own schedule. But getting the training-side three
(#612, #4632, #2410) to work together requires all three applied
simultaneously to matched versions of their dependencies. This repo
is the single source of truth for "what version of everything" and
"how do I assemble it into a working image."

When the training PRs merge upstream, this repo's build chain reduces
to vanilla clones (no patches needed). Until then, the patches in
`patches/` let anyone reproduce the validated stack today. The
inference-side PRs (#2411, sglang#24443) ship independently through
the sibling `vllm-deepep-v2-efa` repo.

## Validation

Cross-framework evidence (2-node EFA traffic, NCCL init markers, DeepEP dispatch+combine latencies, loss curves) is documented across the three repos:

| Document | Coverage |
|---|---|
| [VALIDATION-EVIDENCE.md](docs/VALIDATION-EVIDENCE.md) | Per-framework E2E proofs (3-step training loss 28.5→24.6, grad norms, etc.) |
| [EFA-TRAFFIC-EVIDENCE.md](docs/EFA-TRAFFIC-EVIDENCE.md) | Hardware-counter proof of MoE traffic over EFA (not NVLink), aggregated across all frameworks |
| [DEEPEP-BENCHMARKS.md](docs/DEEPEP-BENCHMARKS.md) | Microbenchmark guide (D+C latency, low-latency kernel, output interpretation) |

Inference-side evidence (vLLM + TRT-LLM) is in the sibling repo [`antonai-work/vllm-deepep-v2-efa`](https://github.com/antonai-work/vllm-deepep-v2-efa) under the same filenames.

## License

Apache 2.0. Patches under `patches/` inherit the license of their
upstream repositories (Apache 2.0 for all of DeepEP, Megatron-LM,
NeMo-RL, and SGLang).

## CI/CD

Two build pipelines available:

- **GitHub Actions** (if configured): `.github/workflows/` for public GHCR push
- **AWS CodeBuild**: `ci/buildspec.yml` + setup guide in [`ci/CODEBUILD-SETUP.md`](ci/CODEBUILD-SETUP.md)

Both pipelines build fast + vanilla modes, run `docker/preflight.sh` (5/5 checks), and push the fast tag to registry.

## Related repos and references

- Sibling base repo: https://github.com/antonai-work/deepep-v2-efa-base
- Sibling inference repo: https://github.com/antonai-work/vllm-deepep-v2-efa
- Upstream DeepEP V2: https://github.com/deepseek-ai/DeepEP
- Upstream Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Upstream NeMo-RL: https://github.com/NVIDIA-NeMo/RL
- Upstream SGLang: https://github.com/sgl-project/sglang
- AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
