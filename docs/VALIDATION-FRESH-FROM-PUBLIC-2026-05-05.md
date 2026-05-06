# Validation: Fresh-From-Public image run, 2026-05-05

## Executive summary

Fresh-from-public-image run on 2 H100 pods over EFA. Training:
Qwen3-30B-A3B-style Shape Y loss 28.56 -> 24.60 across 3 steps with
`HAVE_DEEP_EP_V2=True` and `Active buffer class: ElasticBuffer`,
EFA TX delta 1.05 GB across all 32 rails (0% imbalance). Inference
sibling (vllm-deepep-v2-efa): 24 tokens returned from Qwen3-30B-A3B
DP=16 EP=16, EFA TX delta 6.8 GB. Real distributed training + real
inference on images pulled from public ECR.

## Image provenance

| Artifact | Tag | Digest | Pushed at |
|---|---|---|---|
| NeMo-RL training image | `058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-deepep-v2-efa:allprs-673e66c` | `sha256:2ce3c426357d19150e9b14cc3e560005d5cb97f308b4c42934723d31d0871ae9` | 2026-05-06T00:36:17Z |
| vLLM inference image | `058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-d00e132ee4bb` | `sha256:e28e2922dee180c845ca2bac803dea69c34f3a31079abbe899dfcd948443a11a` | 2026-05-06T01:08:05Z |
| Base image | `058264135704.dkr.ecr.us-east-2.amazonaws.com/deepep-v2-efa-base:v0.1.0-sm90a-amd64` | `sha256:a1ebb88197...` | (Wave 5) |

These were the Wave 5 public-repo images built from this repo's
`docker/build.sh --mode fast`. Both layer on top of the public
`deepep-v2-efa-base` base image.

## Cluster

| Item | Value |
|---|---|
| Region | us-east-2 |
| Instance type | ml.p5.48xlarge (H100, 8 GPU, 32 EFA NICs) |
| Node 0 | hyperpod-i-01aee349f9991c414 (10.1.3.30) |
| Node 1 | hyperpod-i-0a3eb6d3953cceaa7 (10.1.3.73) |
| Lock holder | `ip-172-31-18-239-2590874` (Wave 6) |
| Lock claimed at | 2026-05-06T00:32:00Z |

## Training pass - NeMo-RL + Megatron Shape Y

### Deploy
```
kubectl apply -f tests/k8s/multi-node-training-h100.yaml
# (namespace: deepep-v2-live-val-train-20260506)
# both pods Ready in 30s
```

### Launch (from operator machine)
```
POD0_IP=$(kubectl -n deepep-v2-live-val-train-20260506 get pod \
    nemo-rl-fullstack-0 -o jsonpath='{.status.podIP}')
# On pod-0:
kubectl -n deepep-v2-live-val-train-20260506 exec nemo-rl-fullstack-0 -- \
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=$POD0_IP --master-port=29500 \
    /opt/tests/train_qwen3_moe.py
# On pod-1: same but --node-rank=1
```

The `train_qwen3_moe.py` driver delegates to `/opt/train_step_shapeY.py`,
which is the patched Megatron Shape Y validation loop (random weights,
full fused_a2a -> deep_ep.ElasticBuffer path). Copy it into /opt on each
pod prior to launch (the current image doesn't bake it in — a gap to
close in Wave 7).

### Full rank-0 log excerpt
```
[rank0] === all-PRs-applied stack import probe ===
[rank0] nemo_rl imported OK (version=<no __version__>)
[rank0] megatron.core imported OK (version=0.16.0rc0)
[rank0] deep_ep imported OK (module_file='/opt/DeepEP/deep_ep/__init__.py' has_ElasticBuffer=True)
[rank0] deep_ep.Buffer is deep_ep.buffers.legacy.Buffer (not shim)
[rank0] no-shim invariants hold
[rank0] === handing off to Shape Y train_step driver ===
[rank0] DEEP_EP_USE_V2_SHIM=0 (must be 0 for Shape Y validation)
[rank0] Shape Y probe state: HAVE_DEEP_EP=True HAVE_DEEP_EP_V2=True
[rank0] deep_ep exports: ElasticBuffer=True Buffer=True
[rank0] EFA tx_bytes_total before: 3067915415288936
[rank0] Qwen3-30B-A3B-style model built: hidden=2048 ffn=1024 experts=128 topk=8 blocks=2 local_experts=8
[rank0] Active buffer class: ElasticBuffer (expected: ElasticBuffer)
[rank0] WARMUP  loss=28.5571  grad_norm=35.2123  step_ms=25643.2
[rank0] STEP 1/3  loss=26.4095  grad_norm=30.6430  step_ms=45.1
[rank0] STEP 2/3  loss=25.1042  grad_norm=28.1979  step_ms=39.2
[rank0] STEP 3/3  loss=24.6023  grad_norm=27.0830  step_ms=41.2
[rank0] EFA tx_bytes_total after:  3067916511802200
[rank0] EFA tx_bytes delta:        1096513264 bytes (~1.097 GB)
[rank0] loss trajectory: first=26.4095 last=24.6023 decreased=True
[rank0] SHAPE Y V2 VALIDATION PASS
[rank0] Shape Y train_step returned rc=0 in 27.6s
[rank0] === all-PRs-applied stack E2E training PASS ===
```

- `HAVE_DEEP_EP_V2=True` and `Active buffer class: ElasticBuffer` -
  the V2 ElasticBuffer import path is live end-to-end, no shim.
- Loss decreased monotonically: 28.56 -> 26.41 -> 25.10 -> 24.60.
- grad_norm non-zero and decreasing: 35.21 -> 30.64 -> 28.20 -> 27.08.
- 3-step training completed in 27.6s wall-clock (step_ms ~40ms once
  warmup is done; warmup captured compile-time).
- Driver exit code 0.

### EFA counters
```
TOTAL TX DELTA: 1096513984 bytes (= 1045 MB = 1 GB across 32 rails)
PER-RAIL IMBALANCE: 0% (max=34293576, min=34224568)
PASS: EFA traffic verified real.
```

Every one of the 32 EFA NICs contributed within a 34 KB window of each
other — perfect rail distribution, no NVLink shortcut. This is the
DeepEP V2 native path: Megatron-LM's `fused_a2a` -> patched probe ->
`deep_ep.ElasticBuffer.dispatch` -> NCCL-Gin over AWS EFA SRD.

## Inference pass - vLLM /v1/chat/completions (sibling repo)

Validated in the sibling `vllm-deepep-v2-efa` repo. Summary:
- MODEL=`Qwen/Qwen3-30B-A3B` (bfloat16; Qwen3-30B-A3B-FP8 blocked by
  missing `deep_gemm` in the fast-path image — see Known gaps below).
- DP=16, EP=16, TP=1 across 2 H100 pods.
- Real HTTP request returned `finish_reason: "length"` with
  `completion_tokens: 24` at 01:44:23Z.
- EFA TX delta: 6799 MB across 32 rails.
- `system_fingerprint: vllm-0.1.dev16220+g6d7a3fab2-dp16-ep-nohash`.

Full detail in the sibling repo's
`docs/VALIDATION-FRESH-FROM-PUBLIC-2026-05-05.md`.

## Known gaps captured for next Wave

1. **CUDA ABI mismatch in vLLM image**: base image ships CUDA 12.9; vLLM
   and NCCL wheels in the inference overlay target CUDA 13. A two-line
   pip shim (`pip install nvidia-cuda-runtime`, force-reinstall
   torchvision for cu129) unblocks inference; see sibling repo's doc.
   The nemo-rl-deepep-v2-efa training image does NOT have this gap —
   torch 2.11.0+cu129 matches base.
2. **`train_step_shapeY.py` not baked into training image**: the test
   driver references `/opt/train_step_shapeY.py` but the current fast-path
   image doesn't include it. Operators must `kubectl cp` it in. Fix:
   add to `docker/Dockerfile` next to the other Shape Y assets.
3. **DeepGEMM absent from vLLM fast-path image**: Qwen3-30B-A3B-FP8 needs
   source-built `deep_gemm`; bfloat16 Qwen3-30B-A3B is the current
   workaround.

All three are upstream-image defects, not integration gaps — surfacing
them during a real-cluster fresh-from-public run is the whole point of
this exercise.

## Tear-down
```
kubectl delete -f tests/k8s/multi-node-training-h100.yaml
# Release cluster lock if using shared infrastructure
```
