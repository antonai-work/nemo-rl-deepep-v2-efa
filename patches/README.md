# Patches — standalone extracts of three upstream PRs

Seven `.patch` files, extracted from three open upstream PRs we
authored. Each patch is a standard `git format-patch` output; apply
with `git am <file.patch>` inside a fresh checkout of the target
upstream repo.

## Upstream PRs

| Patch files | Upstream repo | PR | Target base |
|---|---|---|---|
| `0001`, `0002`, `0003` | [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612 "AWS EFA optimizations for V2"](https://github.com/deepseek-ai/DeepEP/pull/612) | `main@b306af0` |
| `0004`, `0005`, `0006` | [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) | _to be filed_ — `dmvevents:deepep-v2-elasticbuffer-support` | `main@23dd639c` |
| `0007` | [`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL) | _to be filed_ — resurrects closed [#2359](https://github.com/NVIDIA-NeMo/RL/pull/2359), closes [#1973](https://github.com/NVIDIA-NeMo/RL/issues/1973) | `46be4e8` |

## Applying patches

Each target repo has different clone + apply instructions. The
`docker/Dockerfile` at the repo root does all three automatically;
these instructions are for reviewers who want to inspect manually.

### DeepEP (patches 0001–0003)

```bash
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
git checkout b306af0  # main at time of PR rebase
git am /path/to/patches/0001-*.patch /path/to/patches/0002-*.patch /path/to/patches/0003-*.patch
```

### Megatron-LM (patches 0004–0006)

```bash
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM
git checkout 23dd639c  # the pin used by NeMo-RL 46be4e8
git am /path/to/patches/0004-*.patch /path/to/patches/0005-*.patch /path/to/patches/0006-*.patch
```

### NeMo-RL (patch 0007)

```bash
git clone https://github.com/NVIDIA-NeMo/RL
cd RL
git checkout 46be4e8
git am /path/to/patches/0007-*.patch
```

## Patch-by-patch summary

### 0001 — DeepEP: cap auto-QP at 2 on EFA

`deep_ep/buffers/elastic.py` — when EFA is detected, override the
default of 129 QPs down to 2. Prevents wasteful QP allocation (EFA
has a tighter budget than Mellanox). Previously also prevented a
crash, but aws-ofi-nccl commit `6e504db` (2026-04-24) fixed the
upstream ring-overflow issue. Now perf-only on the crash axis,
still the right default for EFA resource efficiency.

### 0002 — DeepEP: EFA fast path in `get_rdma_gbs()`

`deep_ep/utils/envs.py` — `check_fast_rdma_atomic_support()` calls
`ibstat`, which is not installed on EFA instances. Without this
patch, SM auto-sizing gets a garbage NIC-bandwidth estimate leading
to 2× perf regression. This commit adds an `FI_PROVIDER=efa` check
and returns 25 GB/s directly.

### 0003 — DeepEP: raise `kScaleoutUpdateInterval` 3→16

`deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` — tunes how
often the hybrid_dispatch kernel flushes scaleout state updates.
3 is aggressive for high-bandwidth fabrics; 16 is more appropriate
for EFA's CPU-proxy Gin backend. Measured 22% D+C latency
improvement on 2-node p5.48xlarge bench. Pure DeepEP kernel tuning.

### 0004 — Megatron: add DeepEP V2 ElasticBuffer support to `_DeepepManager`

`megatron/core/transformer/moe/fused_a2a.py` — the main Shape Y
patch. Adds a `HAVE_DEEP_EP_V2` probe (mirrors the existing
`HAVE_HYBRIDEP` pattern at `fused_a2a.py:270-275`). When V2 is
present, `_DeepepManager`'s `get_buffer()` constructs
`ElasticBuffer` instead of V1 `Buffer`. When V2 is absent, falls
back to the original V1 code path. Zero breaking change to
existing V1 users.

Also bundled: fix for the `num_max_tokens_per_rank` pinning bug
reported in upstream issue #3999 (Megatron passed per-batch token
count, which varies across ranks, breaking V2's JIT template match
— we pin it to a ceiling at first dispatch).

### 0005 — Megatron: graceful fallback for `EventOverlap` import

`fused_a2a.py` — under DeepEP V2, `EventOverlap` moved from the
top-level namespace into `deep_ep.utils`. This patch adds a
try/except that handles both layouts.

### 0006 — Megatron: pass `num_experts` explicitly to V2 backward dispatch

`fused_a2a.py` — the V2 `ElasticBuffer.dispatch()` signature
requires `num_experts` as a mandatory kwarg when `handle=None`.
The V1 path derived it from the handle. This patch threads
`num_experts` through the backward dispatch explicitly.

### 0007 — NeMo-RL: re-export `LD_LIBRARY_PATH` for AWS EFA OFI discovery

`docker/Dockerfile` + `docker/Dockerfile.ngc_pytorch` — prepends
`/opt/amazon/aws-ofi-nccl/lib:/opt/amazon/efa/lib` to
`LD_LIBRARY_PATH` in the release stage. Without this, NCCL can't
find `libnccl-net-ofi.so` at runtime and falls back to the Socket
transport silently. Resurrects closed PR #2359 and closes issue
#1973.

## Notes on provenance

These `.patch` files were generated with `git format-patch`
against the upstream base commits listed above. SHAs in the patch
headers are from our working fork (`dmvevents/...`). The diff
content is identical to what was submitted to the upstream PR
branches.

If you'd prefer to work from the upstream PR branches directly:

- DeepEP: [`dmvevents/DeepEP-1:aws-efa-auto-qp-cap`](https://github.com/dmvevents/DeepEP-1/tree/aws-efa-auto-qp-cap)
- Megatron-LM: [`dmvevents/Megatron-LM:deepep-v2-elasticbuffer-support`](https://github.com/dmvevents/Megatron-LM/tree/deepep-v2-elasticbuffer-support)
- NeMo-RL: filed once the public repo goes live; until then, apply
  `patches/0007-*.patch` manually.
