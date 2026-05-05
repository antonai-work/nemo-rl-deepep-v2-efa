# DeepEP benchmarks: normal and low-latency kernels

This guide walks a reviewer through running the two upstream DeepEP
microbenchmarks inside the image built by `docker/Dockerfile` — the
full dispatch + combine harness (normal kernel) and the decoding-path
low-latency harness. Both run inside the same container image; no
separate build is needed.

End-to-end training expectations (loss, grad_norm, EFA TX) are in
[`VALIDATION.md`](./VALIDATION.md). This doc is just for the two
isolated DeepEP benchmarks.

## Where the tests live inside the image

The image clones DeepEP to `/opt/DeepEP` (see `Dockerfile` `WORKDIR
/opt`). Inside that checkout, at the pinned SHA used by this image:

| Kernel path  | Script                              |
|--------------|-------------------------------------|
| Normal D+C   | `/opt/DeepEP/tests/elastic/test_ep.py` |
| Low-latency  | `/opt/DeepEP/tests/legacy/test_low_latency.py` |

The low-latency script lives under `tests/legacy/` on the pinned
`aws-efa-auto-qp-cap-v2` branch — that's the correct location for V2
(it retains the V1-style `deep_ep.Buffer` API for the low-latency
kernels). If a future rebase moves it (e.g. upstream promotes it to
`tests/elastic/test_low_latency.py` once V2 covers the low-latency
path), update this doc.

## 1. Normal dispatch + combine (`test_ep.py`)

The full D+C harness. Use this for:
- Training-phase MoE sizing (large batches, many tokens).
- Baseline D+C latency and bandwidth numbers.
- Correctness regression across dispatch/combine modes — the harness
  internally sweeps FP8 dispatch on/off, bias variants, async
  scheduling, NVLink allow/disallow, etc. (see `enumerate_ep_modes`
  in the source).

### Flags you will actually touch

| Flag              | Typical  | Notes                                          |
|-------------------|----------|------------------------------------------------|
| `--num-tokens`    | 128      | Per-rank max token count for the harness.      |
| `--hidden`        | 7168     | Hidden dim (matches DeepSeek V2 / Qwen3-A3B).  |
| `--num-topk`      | 8        | Experts per token.                             |
| `--num-experts`   | 288      | Total experts across the EP group.             |
| `--num-processes` | 8        | Ranks per node. Keep at 8 for p5.48xlarge.     |

FP8 dispatch is not a CLI flag on this script — the harness iterates
through `use_fp8_dispatch` internally as part of its mode sweep. The
per-run header line prints the active mode so the reviewer can see
which iteration a given latency number came from.

### 2-node p5.48xlarge invocation (matches our validated config)

The config below matches what we validated in this repo
(`num_tokens=128`, `hidden=7168`, `num_topk=8`, `num_experts=288`).
Expected: D+C p50 ≈ 740 us.

```
# On the rank-0 pod (pod-0)
kubectl -n <ns> exec <pod-0> -- bash -c '
  cd /opt/DeepEP &&
  python3 -m torch.distributed.run \
    --nproc-per-node=8 --nnodes=2 \
    --node_rank=0 \
    --master_addr=<pod0-ip> --master_port=29500 \
    tests/elastic/test_ep.py \
      --num-tokens 128 --hidden 7168 \
      --num-topk 8 --num-experts 288
'

# On the rank-1 pod (pod-1) — same command, --node_rank=1
kubectl -n <ns> exec <pod-1> -- bash -c '
  cd /opt/DeepEP &&
  python3 -m torch.distributed.run \
    --nproc-per-node=8 --nnodes=2 \
    --node_rank=1 \
    --master_addr=<pod0-ip> --master_port=29500 \
    tests/elastic/test_ep.py \
      --num-tokens 128 --hidden 7168 \
      --num-topk 8 --num-experts 288
'
```

Fill in `<ns>`, `<pod-0>`, `<pod-1>`, and `<pod0-ip>` for your
deployment.

## 2. Low-latency dispatch + combine (`test_low_latency.py`)

The low-latency harness. Use this for:
- Decoding-phase latency (small token counts, tight SLAs).
- Inference workloads where step time is dominated by a few tokens
  per rank, not bulk training throughput.
- Sanity-checking that the low-latency kernel path actually improves
  over the normal path for your decoding batch size.

On Mellanox InfiniBand the low-latency kernel typically reports
dispatch + combine under 100 us for the default 128-token /
hidden-7168 shape. EFA numbers on p5.48xlarge are not yet published
upstream — report what the reviewer's run produces and compare
against the normal-kernel 740 us baseline to confirm the expected
latency reduction.

### Flags you will actually touch

| Flag              | Typical  | Notes                                               |
|-------------------|----------|-----------------------------------------------------|
| `--num-tokens`    | 128      | Per-rank token count. Stay small — this is decode.  |
| `--hidden`        | 7168     | Hidden dim.                                         |
| `--num-topk`      | 8        | Experts per token.                                  |
| `--num-experts`   | 288      | Total experts.                                      |
| `--num-processes` | 8        | Ranks per node.                                     |
| `--disable-nvlink`| (flag)   | Force the cross-node path; useful to confirm EFA    |
|                   |          | is actually carrying traffic, not NVLink.           |
| `--use-logfmt`    | (flag)   | Exercise the LogFMT combine path.                   |

### 2-node p5.48xlarge invocation

```
# Pod 0
kubectl -n <ns> exec <pod-0> -- bash -c '
  cd /opt/DeepEP &&
  python3 -m torch.distributed.run \
    --nproc-per-node=8 --nnodes=2 \
    --node_rank=0 \
    --master_addr=<pod0-ip> --master_port=29500 \
    tests/legacy/test_low_latency.py \
      --num-tokens 128 --hidden 7168 \
      --num-topk 8 --num-experts 288
'

# Pod 1 — same command, --node_rank=1
kubectl -n <ns> exec <pod-1> -- bash -c '
  cd /opt/DeepEP &&
  python3 -m torch.distributed.run \
    --nproc-per-node=8 --nnodes=2 \
    --node_rank=1 \
    --master_addr=<pod0-ip> --master_port=29500 \
    tests/legacy/test_low_latency.py \
      --num-tokens 128 --hidden 7168 \
      --num-topk 8 --num-experts 288
'
```

## Interpreting the output

Both scripts print per-configuration headers followed by
`Dispatch + combine bandwidth/latency` lines. The key things to look
for, top-down:

- `Config:` header — confirms `#QPs: N/M` matches your expectation.
  On this image the allocated QP count `M` should be 2 for normal D+C
  because `EP_EFA_MAX_QPS=2` is baked into the image (see the env
  block in `docker/Dockerfile`). A value above 2 means the env did
  not take effect and you will exceed the p5.48xlarge EFA QP budget.
- `Dispatch + combine bandwidth:` — aggregate GB/s across all NICs.
  On 2-node p5.48xlarge with 32 EFA NICs total, expect a number
  consistent with `EP_EFA_RDMA_GBS=25.0` per rail (this is the
  upstream PR #612 fast-path tag; it tells DeepEP V2 that EFA SRD
  delivers ~25 Gbps/rail once QPs are warm).
- `Dispatch + combine latency:` / `p50`, `p95`, `max` — the numbers
  you cite when reporting.
- `RDMA bw:` — per-rail bandwidth. Divide by the number of EFA
  NICs the run actually used to sanity-check against the 25 Gbps
  per-rail ceiling. Below that is fine; above would be impossible
  and points at a measurement bug.

If the normal-kernel p50 comes in meaningfully above 740 us, or the
low-latency-kernel p50 comes in above ~100 us (InfiniBand reference),
the fast path probably did not activate — check that:

- `EP_EFA_MAX_QPS=2` is still present inside the pod (`env | grep
  EP_EFA`).
- `EP_EFA_RDMA_GBS=25.0` is still present.
- `FI_PROVIDER=efa` is active (`env | grep FI_PROVIDER`).
- `DEEP_EP_BACKEND=nccl` — this image uses the NCCL-GIN backend
  for V2; silently flipping it removes EFA acceleration.
- NCCL init logs show `NET/OFI Selected provider is efa, fabric is
  efa-direct (found 32 nics)`, not `NET/Socket`. The `NET/Socket`
  anti-signal is called out in
  [`VALIDATION.md`](./VALIDATION.md#failure-modes-and-what-they-mean).

## Cross-link

- End-to-end training expectations (loss, grad_norm, EFA TX
  byte-counter gate): [`VALIDATION.md`](./VALIDATION.md).
- Upstream PR tracker (which patches activate which benchmark
  behavior): [`UPSTREAM-STATUS.md`](./UPSTREAM-STATUS.md).
