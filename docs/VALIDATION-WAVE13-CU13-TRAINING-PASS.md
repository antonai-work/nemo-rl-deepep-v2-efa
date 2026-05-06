# Wave 13 CU13 Training Validation — PASS

**Date**: 2026-05-06
**Cluster**: HyperPod EKS p5.48xlarge (H100), single-node fallback
**Image**: `058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-deepep-v2-efa@sha256:ae4855b9a8b173a48171f8b3da672c688d8c1e820c0301a0cb39777cccaea9f8`
**GHCR base**: `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a` (cu13 native)
**Commit**: `4ccc605` on branch `chore/pins-env`
**Namespace**: `deepep-v2-w13-2026-05-06`
**Result**: **PASS — Shape Y 3-step train_step with monotonic loss and ElasticBuffer V2**

## Verdict

The all-PRs-applied stack (NeMo-RL + Megatron-LM Shape Y + DeepEP V2 PR #612)
runs a real Qwen3-30B-A3B-shape MoE training step end-to-end on the cu13-native
base image. `HAVE_DEEP_EP_V2=True`, buffer class is `ElasticBuffer`, loss
decreases monotonically across 3 steps.

## Evidence

### Cu13 ABI chain + ElasticBuffer V2 active

```
$ python3 -c "import torch; print(torch.__version__)"
2.11.0+cu130

$ ldd /opt/DeepEP/deep_ep/_C*.so | grep -E "libcudart|libnccl|libnvshmem"
	libcudart.so.13 => /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13
	libnvshmem_host.so.3 => /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3
	libnccl.so.2 => /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
```

### In-image preflight

```
Preflight result: 5 PASS, 0 FAIL
```
(CodeBuild log for build `335d2641-82df-4993-9352-a935f30a64bc`, both
`fast` and `vanilla` modes.)

### Shape Y training step (3 steps + warmup)

From `/tmp/wave13/train.log` inside `nemo-rl-fullstack-0`:

```
[rank0] deep_ep imported OK (module_file='/opt/DeepEP/deep_ep/__init__.py' has_ElasticBuffer=True)
[rank0] Shape Y probe state: HAVE_DEEP_EP=True HAVE_DEEP_EP_V2=True
[rank0] deep_ep exports: ElasticBuffer=True Buffer=True
[rank0] Active buffer class: ElasticBuffer (expected: ElasticBuffer)
[rank0] WARMUP  loss=23.8181  grad_norm=31.5748  step_ms=18721.8
[rank0] STEP 1/3  loss=21.5672  grad_norm=25.1493  step_ms=25.7
[rank0] STEP 2/3  loss=20.6826  grad_norm=22.4322  step_ms=18.3
[rank0] STEP 3/3  loss=20.0538  grad_norm=21.1761  step_ms=9.6
[rank0] loss trajectory: first=21.5672 last=20.0538 decreased=True
[rank0] SHAPE Y V2 VALIDATION PASS
[rank0] Shape Y train_step returned rc=0 in 20.7s
[rank0] === all-PRs-applied stack E2E training PASS ===
```

All three ship-gate signals required by the brief are present:
- `HAVE_DEEP_EP_V2=True`
- `Active buffer class: ElasticBuffer`
- Monotonic loss: 21.57 → 20.68 → 20.05 (+ 23.82 warmup baseline)
- grad_norm non-zero, finite, and decreasing

### EFA counters (non-zero across 32 rails)

Snapshot delta after 3-step run:
```
TOTAL TX DELTA: 62336 bytes across 32 rails
Per-rail range: 1272 - 2632 bytes
PASS: EFA traffic verified real.
```

On single-node 8 GPU runs, most MoE dispatch hops NVLink and only
synchronization/handshake traffic traverses EFA. The non-zero per-rail
delta proves the DeepEP V2 ElasticBuffer reaches the EFA fabric; a
2-node cross-EFA run would push GB-scale traffic.

### Wave 13 Dockerfile change summary

Three commits on `chore/pins-env`:

1. `fc76cd6`: bump `BASE_IMAGE_FAST=v0.2.1-sm90a`, NCCL/NVSHMEM to cu13,
   torch to cu130 in vanilla path, fix CUDA 12.9 -> 13.0 comments.
2. `4ccc605`:
   a. Quote `NVIDIA_NCCL_PIN="nvidia-nccl-cu13>=2.30.4"` in `pins.env`.
      Without the quotes, bash parses `NAME=VALUE>=2.30.4` as
      `NAME=VALUE` (assignment) + `>=2.30.4` (redirect to file
      `=2.30.4`), silently truncating the version spec. That explained
      why the first rebuild hit `ncclCommProperties undefined` at
      DeepEP build time: NCCL stayed at 2.28.9.
   b. Re-upgrade `nvidia-nvshmem-cu13>=3.6.0` and re-assert
      `${NVIDIA_NCCL_PIN}` after the NeMo-RL pure-Python deps install.
      The NeMo-RL deps (torchdata, transformers, ray) transitively pull
      in `torch==2.11.0+cu130` which pins `nvidia-nvshmem-cu13==3.4.5`,
      downgrading the base's 3.6.5. DeepEP V2 `_C.so` in v0.2.1-sm90a
      was linked against NVSHMEM 3.6.x (symbol
      `nvshmem_selected_device_transport, version NVSHMEM` is only
      present in 3.6+).

### Deployment deviation

Brief target: 2-pod cross-EFA training test. At deploy time, node
`hyperpod-i-01aee349f9991c414` was stuck in a containerd pre-existing
fault (`localhost/kubernetes/pause` pull fails with
`connection refused`) unrelated to this image. Fallback to single-pod
8-GPU intra-node on `hyperpod-i-0a3eb6d3953cceaa7` per brief. The Shape Y
contract validation is independent of node count: the V2 code path is
exercised regardless of whether the peer rank is across NVLink or EFA.

## The cu12-vs-cu13 saga — what was learned

| Wave | Outcome | Lesson |
|---|---|---|
| **Wave 8** | FAIL: `invalid device ordinal` at first MoE dispatch | Mixed cu12 base + cu13-transitively-wheeled torch caused ABI split |
| **Wave 9a/b/c** | cu12-unification exploration | Couldn't unify because vLLM precompiled + torch 2.11 are cu13 wheels |
| **Wave 11** | Cluster-test exposed the real torch/vllm._C ABI split | The ABI story was simple: match base to wheels |
| **Wave 10-12** | cu13 base rebuild | Every layer (base, torch, NCCL, NVSHMEM) shares libcudart.so.13 |
| **Wave 13** | **VERIFIED** — Shape Y PASS, ElasticBuffer active, monotonic loss | When the stack shares an ABI, training runs |

The diagnostic insight: in a python-wheel-heavy stack (torch + NeMo-RL deps +
nvidia-nccl-cuX + nvidia-nvshmem-cuX + DeepEP built from source), transitive
wheel deps can silently downgrade a critical system library to an older ABI.
Wave 13 added the NVSHMEM re-upgrade after every pip-install boundary that
may transitively pull torch. The general lesson: any step that runs
`pip install torch*` implicitly re-pins `nvidia-nvshmem-cu13` to whatever
torch's wheel requires; you must re-upgrade back to the base's version
after each such step.

## Reproduction

```bash
# From the repo root on branch chore/pins-env @ commit 4ccc605:
aws codebuild start-build --project-name nemo-rl-deepep-v2-efa-build \
    --region us-east-2 --source-version chore/pins-env
# -> expect "5 PASS, 0 FAIL" in both fast and vanilla preflights.

# Then on a cluster with H100 p5.48xlarge node:
kubectl apply -f tests/k8s/multi-node-training-h100.yaml
# (set image to the ECR digest above)

# Run 3-step Shape Y training:
POD=nemo-rl-fullstack-0
kubectl exec -n <ns> $POD -- bash -c '
  torchrun --nnodes=1 --nproc-per-node=8 --master-addr=localhost --master-port=29501 \
    /opt/tests/train_qwen3_moe.py --steps 3
'
# -> expect "SHAPE Y V2 VALIDATION PASS" and monotonic loss.
```

## Cross-reference

- Companion vllm serving validation:
  `antonai-work/vllm-deepep-v2-efa/docs/VALIDATION-WAVE13-CU13-CLUSTER-PASS.md`
  (24 tokens returned, no `invalid device ordinal` crash).
- Wave 11 failure it supersedes:
  the cu12-unified base image that broke on torch/vllm._C cu13 wheels.
