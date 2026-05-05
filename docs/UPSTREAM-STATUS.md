# Upstream PR status

Live tracking of the three upstream PRs this repo depends on. When
all three merge, this repo's `docker/Dockerfile` can drop the
patch-apply step and clone vanilla upstream instead.

Last updated: 2026-05-05.

## Summary

| Repo | PR | State | Estimate |
|---|---|---|---|
| [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN | After 2026-04-28 + 7 days. Rebased + pinged 2026-05-05. |
| [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) | [#4632](https://github.com/NVIDIA/Megatron-LM/pull/4632) | DRAFT | Filed 2026-05-05. |
| [`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL) | [#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410) | DRAFT | Filed 2026-05-05. |

## PR #612: DeepEP AWS EFA optimizations

- **URL**: https://github.com/deepseek-ai/DeepEP/pull/612
- **Head branch**: `dmvevents/DeepEP-1:aws-efa-auto-qp-cap`
  (3 commits on top of `deepseek-ai/DeepEP@main`)
- **Filed**: 2026-04-28
- **Last activity**: 2026-05-05 (rebase + courtesy ping)
- **State**: OPEN, 0 reviews, 1 comment (maintainer pending)

### Commits

| SHA | Message |
|---|---|
| `fe20874` | `aws-efa: cap auto-QP at 2 on EFA to avoid 128-slot GIN ring overflow` |
| `0b78333` | `aws-efa: add EFA fast path in get_rdma_gbs to fix SM auto-sizing` |
| `c84dcac` | `aws-efa: raise dispatch kScaleoutUpdateInterval from 3 to 16` |

### What happens when this merges

- Drop `DeepEP_REPO_URL=dmvevents/DeepEP-1` in
  `docker/Dockerfile`, restore to `deepseek-ai/DeepEP`.
- Drop `patches/0001-*.patch`, `patches/0002-*.patch`,
  `patches/0003-*.patch` from the apply list.
- Remove this section from `UPSTREAM-STATUS.md`.
- This repo's `docker/build.sh` becomes a pure-vanilla recipe.

## Megatron-LM PR (Shape Y)

- **URL**: https://github.com/NVIDIA/Megatron-LM/pull/4632
- **Head branch**: `dmvevents/Megatron-LM:deepep-v2-elasticbuffer-support`
  (3 commits on top of `NVIDIA/Megatron-LM@main`)
- **Filed**: 2026-05-05
- **State**: DRAFT (validated standalone 2026-04-29 + validated as
  integrated stack 2026-05-05)

### Commits

| SHA | Message |
|---|---|
| `d6b6e138d` | `moe: add DeepEP V2 ElasticBuffer support to _DeepepManager` |
| `cf18f7268` | `moe: graceful fallback for EventOverlap import under DeepEP V2` |
| `cbacb0bbf` | `moe: pass num_experts explicitly to V2 backward dispatch` |

### Scope

Teaches Megatron-LM's `_DeepepManager` (the flex dispatcher) to call
`deep_ep.ElasticBuffer` when V2 is installed, falling back to the
legacy V1 `deep_ep.Buffer` otherwise. Zero-change for existing V1
users. Mirrors the existing `HAVE_HYBRIDEP` probe pattern at
`megatron/core/transformer/moe/fused_a2a.py:270-275` — reviewers
will recognize the idiom.

Bundled bug fix: resolves upstream issue #3999
(`max_num_of_tokens_per_rank` QP assertion) for the V2 branch.

### What happens when this merges

- Drop `MEGATRON_REPO_URL=dmvevents/Megatron-LM` in
  `docker/Dockerfile`, restore to `NVIDIA/Megatron-LM`.
- Drop `patches/0004-*.patch`, `patches/0005-*.patch`,
  `patches/0006-*.patch` from the apply list.

## NeMo-RL PR (LD_LIBRARY_PATH)

- **URL**: https://github.com/NVIDIA-NeMo/RL/pull/2410
- **Head branch**: `dmvevents/RL:aws-efa-deepep-support`
  (1 commit on top of `NVIDIA-NeMo/RL@46be4e8`)
- **Filed**: 2026-05-05
- **State**: DRAFT (validated as part of integrated stack
  2026-05-05)
- **Related**: resurrects closed PR
  [#2359](https://github.com/NVIDIA-NeMo/RL/pull/2359), closes
  issue [#1973](https://github.com/NVIDIA-NeMo/RL/issues/1973)

### Commit

| SHA | Message |
|---|---|
| `7f0f21a7` | `deps: re-export LD_LIBRARY_PATH for AWS EFA OFI discovery` |

### Scope

Prepends `/opt/amazon/aws-ofi-nccl/lib` and
`/opt/amazon/efa/lib` to `LD_LIBRARY_PATH` in the release stage
of `docker/Dockerfile` + `docker/Dockerfile.ngc_pytorch`. Without
this, `libnccl-net-ofi.so` is not discoverable at runtime and
NCCL silently falls back to the Socket transport on AWS EFA
instances. Three lines added per Dockerfile, zero removed.

Also adds one example recipe YAML under
`examples/configs/recipes/llm/` demonstrating DeepEP + EFA
configuration.

### What happens when this merges

- Drop `NEMORL_REPO_URL=dmvevents/RL`, restore to
  `NVIDIA-NeMo/RL`.
- Drop `patches/0007-*.patch`.

## Related issues (for reviewers unfamiliar with the context)

- **DeepEP V2 release**: https://github.com/deepseek-ai/DeepEP/pull/605
  (merged 2026-04-29 as `b306af0`)
- **aws-ofi-nccl GIN ring fix**: https://github.com/aws/aws-ofi-nccl/commit/6e504db
  (merged 2026-04-24; supersedes our closed PR
  [#1206](https://github.com/aws/aws-ofi-nccl/pull/1206))
- **Megatron-LM issue #3999** (HybridEP QP assertion bug):
  https://github.com/NVIDIA/Megatron-LM/issues/3999
- **Megatron-LM issue #2647** (EFA feature request, NCCL team
  engaged): https://github.com/NVIDIA/Megatron-LM/issues/2647
- **NeMo-RL issue #1973** (LD_LIBRARY_PATH for EFA OFI
  discovery): https://github.com/NVIDIA-NeMo/RL/issues/1973
