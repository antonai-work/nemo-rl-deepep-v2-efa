# Wave 14 2-Pod Cross-Node EFA Training Validation — BLOCKED (node infrastructure)

**Date**: 2026-05-06
**Cluster**: HyperPod EKS 2x p5.48xlarge (H100)
**Image**: `058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-deepep-v2-efa:allprs-4ccc605`
**Digest**: `sha256:ae4855b9a8b1...`
**GHCR base**: `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a`
**Namespace**: `deepep-v2-w14-train-2026-05-06`
**Result**: **BLOCKED — structural containerd/sandbox-pause fault on node `hyperpod-i-01aee349f9991c414`**

## Verdict

Wave 14's target ship-gate (50-step 2-pod Shape Y training with
`ElasticBuffer`, loss convergence, and >=1 GB EFA TX delta per pod) could
NOT be validated. The root cause is infrastructure, not software.

Node `hyperpod-i-01aee349f9991c414` reports `Status: Ready` with
`containerd://2.1.5` at the kubelet level, but creating any NEW pod sandbox
fails with:

```
Failed to create pod sandbox: rpc error: code = Unknown desc =
failed to start sandbox "<sha>": failed to get sandbox image
"localhost/kubernetes/pause": ... dial tcp 127.0.0.1:443: connect:
connection refused
```

The fault is reproducible across workloads — this validation deployed an
A/B control after vLLM failed identically:

| Workload | Node `0a3eb` | Node `01aee` |
|---|---|---|
| vllm-deepep-v2 (Wave 14 serve) | Running 1/1 at 11m | ContainerCreating (same error, 11m+) |
| nemo-rl-fullstack (Wave 14 train) | Running 1/1 at 3m57s | ContainerCreating (same error, 3m57s+) |

This confirms the fault is node-level (containerd sandbox-image path),
independent of image content or resource request.

## What was attempted

| Step | Result |
|---|---|
| H100 cluster lock claimed | OK |
| `nvshmem-efa/deepep-nvshmem` scaled to 0 | OK |
| Apply `deepep-v2-w14-train-2026-05-06` manifest (2 replicas, `vpc.amazonaws.com/efa: 32`) | OK |
| Pod-1 on `0a3eb` | Running 1/1 at 3m57s, 8 GPUs visible |
| Pod-0 on `01aee` | ContainerCreating — same `FailedCreatePodSandBox` |

## Failure signature

- `FailedCreatePodSandBox ... dial tcp 127.0.0.1:443: connect: connection refused`
- Repeats on a 10-15s cadence.
- **NOT** one of the guarded Wave 14 crash signatures (`c10::cuda::SetDevice`,
  `nvshmem_selected_device_transport`). Cu13 ABI hypothesis remains intact.

## Image sanity (collateral evidence from `nemo-rl-fullstack-1`)

Pod-1 on the healthy node confirmed the Wave-13 cu13 stack remains live:

```
$ python3 -c "import deep_ep; print(deep_ep.__file__)"
/opt/DeepEP/deep_ep/__init__.py

$ python3 -c "import deep_ep; print('ElasticBuffer' in dir(deep_ep))"
True

$ nvidia-smi -L | wc -l
8
```

The Shape Y training script exists and is intact:
`/opt/tests/train_step_shapeY.py` (11184 bytes, ctime 2026-05-06T11:36Z).

## Per-pod EFA TX delta

N/A — could not execute the 50-step 2-node torchrun. The ship-gate requires
`torchrun --nnodes=2 --nproc-per-node=8`, which requires both pods Running
concurrently. Pod-0 never started.

## Last loss value after 50 training steps

N/A — training not started. Wave 13 single-node run delivered 3-step loss
trajectory 21.57 -> 20.67 -> 20.05 (see
[VALIDATION-WAVE13-CU13-TRAINING-PASS.md](VALIDATION-WAVE13-CU13-TRAINING-PASS.md)).

## Recommended remediation (outside Wave 14 scope)

Per AWS HyperPod runbook, remedy order:

1. `ctr -n k8s.io images ls | grep pause` on `01aee` to verify pause image
   presence. If absent, preload via `ctr images import` or re-run the
   HyperPod bootstrap script.
2. Restart `kubelet` + `containerd` services on the node.
3. If (2) does not clear it, cordon + drain + replace the node through
   HyperPod SageMaker console.

## Relationship to prior waves

| Wave | Scope | Result |
|---|---|---|
| Wave 8 | cu12 base + cu13 torch training | FAIL (ABI mismatch) |
| Wave 11 | cu12 realign diagnostic | FAIL (wheels are cu13) |
| Wave 13 | Single-node Shape Y cu13 training | PASS (3-step loss convergence, 62 KB EFA) |
| Wave 14 | 2-node cross-node sustained training (50 steps) | **BLOCKED** (node-01aee containerd fault) |

Wave 13 evidence holds: cu13 stack trains correctly and routes MoE all2all
via `ElasticBuffer` over EFA. Wave 14 ship gate deferred pending node repair.

## Evidence files

Captured on operator host at `/tmp/wave14/evidence/`:

- `nemo-pod0-describe-final.txt` — full `kubectl describe` of stuck
  `nemo-rl-fullstack-0` showing identical sandbox-pause fault as the vllm
  pod that preceded it.
- `01aee-all-pods-final.txt` — 35+ pods on node, 3+ stuck in
  ContainerCreating (`nemo-rl-fullstack-0`, `checkpoint-backup-...`, and a
  6-hour-old `node-debugger`), confirming this is a pre-existing node-level
  fault unrelated to Wave 14.
- `node-01aee-describe.txt` — node state (Ready=True, 32 EFA, 8 GPUs).

## Verdict summary

Ship gate **not met** due to infrastructure. Training image remains PASS
from Wave 13. Rerun Wave 14 after node-01aee containerd state is
repaired.
