# Validation reference

The expected-output contract for the training driver at
`tests/train_qwen3_moe.py`, against the exact
`docker/Dockerfile` in this repo.

Measured 2026-05-05 on 2× p5.48xlarge H100 EFA. Every line below
is reproducible by a reviewer with the right hardware.

## Preflight (before cluster deploy)

```bash
docker run --rm <your-tag> bash docker/preflight.sh
```

Expected `stdout` (exact match in spirit, versions may vary by
day-0 build):

```
Check 1/5: libnccl-net-ofi.so discoverable via ldconfig
  PASS: /opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so (libc6,x86-64) => /opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so

Check 2/5: HAVE_DEEP_EP_V2 probe is True
  PASS: HAVE_DEEP_EP_V2 = True

Check 3/5: deep_ep.Buffer resolves to the vanilla class (no shim)
  PASS: <class 'deep_ep.buffers.legacy.Buffer'>
  (rejected: 'api_shim.buffer_v1_compat.CompatBuffer' would fail this test)

Check 4/5: /opt/api-shim directory does NOT exist
  PASS: no shim installed

Check 5/5: DEEP_EP_USE_V2_SHIM=0 in container env
  PASS: DEEP_EP_USE_V2_SHIM=0

=== 5/5 checks PASS ===
```

Any FAIL here means the image was not built with all 7 patches
applied. Re-run `docker/build.sh`.

## NCCL initialization log

When the training driver starts, rank 0 prints NCCL initialization
lines via `NCCL_DEBUG=INFO`. The key lines to grep for:

```
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NCCL INFO NET/OFI Using transport protocol RDMA
NCCL INFO Init COMPLETE
```

The `efa-direct` provider + `RDMA` protocol + 32 NICs confirms the
aws-ofi-nccl plugin is loaded and EFA is the active transport.
`git-6e504db` confirms the source-built plugin (with the
`active_put_signal` bitset fix) is in use, not the installer-bundled
NGC plugin.

Anti-signal to watch for: `NCCL INFO NET/Socket` — means NCCL
fell back to TCP because it couldn't find the OFI plugin. That's
usually an `LD_LIBRARY_PATH` issue (patch 0007 should fix this
on NeMo-RL images).

## DeepEP V2 activation log

From `tests/train_qwen3_moe.py`'s first print statements:

```
HAVE_DEEP_EP_V2 = True
Active buffer class: ElasticBuffer
num_allocated_qps = 2   (0001: capped for EFA)
EP_EFA_RDMA_GBS = 25.0  (0002: EFA fast path hit)
```

Anti-signal: `Active buffer class: Buffer` means Megatron's V1
code path is running, not the V2 patch. Check patch 0004 applied.

## Training step output

3 training steps + 1 warmup step. Config: Qwen3-30B-A3B (hidden
2048, ffn_hidden 1024, 128 experts top-8, 2 MoE blocks, seq_len
512, micro_bs 2). EP=16 across 2× 8 GPUs = 16 ranks.

```
WARMUP  loss=28.5571  grad_norm=35.2123  step_ms=24766.8
STEP 1  loss=26.4074  grad_norm=30.6430  step_ms=315.9
STEP 2  loss=25.0856  grad_norm=28.1979  step_ms=42.6
STEP 3  loss=24.6252  grad_norm=27.0909  step_ms=43.4

=== all-PRs-applied stack E2E training PASS ===
```

### What this proves

- **Loss monotonic decrease** (28.56 → 24.63): real training
  signal, not a randomly-initialized forward pass.
- **grad_norm present and non-zero** (35.2 → 27.1): backward
  pass ran, gradients were computed and AllReduce'd.
- **Step latency stabilizes after warmup** (316 ms → 43 ms): the
  warmup step absorbs CUDA graph construction, kernel JIT compile,
  and EFA RDMA connection setup; steady-state is ~43 ms.

### Variance expected day-over-day

Step-1 latency can vary 200-500 ms depending on JIT cache state.
Steady-state should be within ±5 ms of 43. Loss values are
deterministic with fixed seed (the driver sets `torch.manual_seed`
and `torch.cuda.manual_seed_all`).

## EFA counter deltas

The `tests/verify_efa_traffic.sh` script snapshots
`/sys/class/infiniband/*/ports/1/hw_counters/tx_bytes` +
`rx_bytes` across all EFA NICs, before and after the training
run, and computes the delta.

Expected:

```
Pod 0 tx_bytes delta across 16 NICs: 1.096 GB
Pod 1 tx_bytes delta across 16 NICs: 1.096 GB
```

≥ 1 GB per pod confirms:
- MoE dispatch/combine traffic actually went over EFA (not NVLink
  shortcut, not TCP socket fallback)
- Cross-node communication was balanced (both pods saw the same
  TX volume; no skewed expert placement)

### Why only 1 GB and not more?

The workload is intentionally small (micro_bs=2, 3 training steps).
Realistic production training runs would see 100s of GB over the
same duration. 1 GB is our evidence floor — proof that EFA is
active, not a performance target.

For real-world tuning, benchmark with:

```bash
# Inside the pod, D+C-only microbenchmark
cd /opt/DeepEP/tests/elastic
python3 test_ep.py --num-tokens 128 --hidden 7168 --num-topk 8
```

Expected on 2-node p5.48xlarge: D+C p50 ≈ 740 µs. Matches the
performance ceiling reported by the upstream DeepSeek V2 test
harness on Mellanox InfiniBand.

## Failure modes and what they mean

| Symptom | Likely cause |
|---|---|
| `NCCL INFO NET/Socket` instead of `NET/OFI` | Patch 0007 not applied to NeMo-RL Dockerfile; `libnccl-net-ofi.so` not in `LD_LIBRARY_PATH`. |
| Training hangs at first dispatch with `num_allocated_qps = 129` | Patch 0001 not applied; DeepEP trying to allocate more QPs than EFA provides. If aws-ofi-nccl is < 6e504db, also hits the ring-overflow bug. |
| `Active buffer class: Buffer` (V1) | Patches 0004-0006 not applied to Megatron; flex dispatcher is calling the V1 code path. |
| Loss NaN or stuck | Real training bug; check Megatron config matches the driver's expectations. Not a stack issue. |
| EFA TX delta ≈ 0 | Either ranks are colocated (single-node run, no cross-node traffic) or `FI_PROVIDER!=efa`. Check K8s manifest env vars. |
| Step 1 latency > 60s | First-time kernel JIT compile; Step 2+ should be fast. Use FSx to cache JIT output across restarts. |
