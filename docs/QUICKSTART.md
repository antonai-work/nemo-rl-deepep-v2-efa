# Quickstart — 20 minutes from clone to running training

Assumes you already have:
- A Linux build host with Docker and NVIDIA Container Toolkit
- An EKS cluster with 2× p5.48xlarge H100 nodes (or p5en, H200)
- EFA device plugin installed on the nodes
- An ECR repository you can push to
- AWS CLI authenticated for push

If you're missing any of those, see `docs/ARCHITECTURE.md` for
prerequisites.

## 1. Clone + build (45 min)

```bash
git clone https://github.com/antonai-work/nemo-rl-deepep-v2-efa
cd nemo-rl-deepep-v2-efa

# Set your ECR target
export ECR_REPO=<your-account>.dkr.ecr.<region>.amazonaws.com/nemo-rl-deepep-v2-efa
export TAG=allprs-$(git rev-parse --short HEAD)

docker/build.sh "$ECR_REPO:$TAG"
```

The `build.sh` script:
1. Applies patches `0001`–`0007` to their target upstream repos
   as sub-stages of the Docker build
2. Compiles DeepEP V2 against NCCL 2.30.4
3. Installs Megatron-LM + NeMo-RL as Python packages
4. Produces a final image with all env vars preset

Expected output at the end:
```
Successfully tagged <ECR_REPO>:allprs-abc1234
```

## 2. Preflight (2 min)

Before deploying to the cluster, verify the image assembled
correctly:

```bash
docker run --rm "$ECR_REPO:$TAG" bash /opt/docker/preflight.sh
```

Expected: `5/5 checks PASS`. See `docs/VALIDATION.md` for the
exact checks.

## 3. Push to ECR (5 min)

```bash
aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin $ECR_REPO
docker push "$ECR_REPO:$TAG"
```

## 4. Deploy the K8s manifest (2 min)

Edit `tests/k8s/multi-node-training-h100.yaml`:
- Replace `<ECR_REPO>:<TAG>` with your pushed image
- Set `claimName` on the FSX volume to your existing PVC (or
  create a new one — see the manifest comments)
- Adjust namespace if needed (default: `nemo-rl-fullstack`)

```bash
kubectl apply -f tests/k8s/multi-node-training-h100.yaml
kubectl -n nemo-rl-fullstack rollout status statefulset/fullstack --timeout=10m
```

## 5. Run the training (3 min for 3 steps)

Open two terminals, one per pod:

### Terminal 1: pod 0 (node rank 0)

```bash
POD0_IP=$(kubectl -n nemo-rl-fullstack get pod fullstack-0 -o jsonpath='{.status.podIP}')
kubectl -n nemo-rl-fullstack exec -it fullstack-0 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=0 \
  --master-addr=$POD0_IP --master-port=29500 \
  /opt/tests/train_qwen3_moe.py
```

### Terminal 2: pod 1 (node rank 1)

```bash
POD0_IP=$(kubectl -n nemo-rl-fullstack get pod fullstack-0 -o jsonpath='{.status.podIP}')
kubectl -n nemo-rl-fullstack exec -it fullstack-1 -- \
  torchrun --nproc-per-node=8 --nnodes=2 --node-rank=1 \
  --master-addr=$POD0_IP --master-port=29500 \
  /opt/tests/train_qwen3_moe.py
```

Expected pod-0 output:

```
HAVE_DEEP_EP_V2 = True
Active buffer class: ElasticBuffer
NCCL INFO NET/OFI Initializing aws-ofi-nccl git-6e504db
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NCCL INFO NET/OFI Using transport protocol RDMA

WARMUP  loss=28.5571  grad_norm=35.2123
STEP 1  loss=26.4074  grad_norm=30.6430
STEP 2  loss=25.0856  grad_norm=28.1979
STEP 3  loss=24.6252  grad_norm=27.0909

=== all-PRs-applied stack E2E training PASS ===
```

## 6. Verify EFA traffic (1 min)

Before starting training, snapshot:

```bash
kubectl -n nemo-rl-fullstack exec fullstack-0 -- \
  bash /opt/tests/verify_efa_traffic.sh snapshot /tmp/before
```

After training completes:

```bash
kubectl -n nemo-rl-fullstack exec fullstack-0 -- \
  bash /opt/tests/verify_efa_traffic.sh verify /tmp/before /tmp/after 1
```

Expected: `EFA TX delta 1.096 GB >= 1 GB threshold (PASS)`.

## 7. Teardown

```bash
kubectl delete -f tests/k8s/multi-node-training-h100.yaml
```

## Troubleshooting

**Build fails at `patch 0001` apply step.** Upstream DeepEP may
have moved past `b306af0`. Check `patches/README.md` for the
target base; re-sync with `git am --abort && git rebase upstream/main`
before re-trying.

**`NCCL INFO NET/Socket` instead of `NET/OFI`.** LD_LIBRARY_PATH
does not include `/opt/amazon/aws-ofi-nccl/lib`. Check patch 0007
was applied. Verify with `docker run --rm <image> bash -c 'echo $LD_LIBRARY_PATH'`.

**Training hangs at first dispatch.** Almost always
`num_allocated_qps > EFA budget`. Check patch 0001 applied with
`docker run --rm <image> grep 'num_allocated_qps = 2' /opt/DeepEP/deep_ep/buffers/elastic.py`.

**`Active buffer class: Buffer` (V1).** Patches 0004–0006 not
applied to Megatron. Check with
`docker run --rm <image> grep 'HAVE_DEEP_EP_V2' /opt/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`.

**EFA TX delta = 0.** Either ranks colocated (single-node run) or
`FI_PROVIDER` != `efa`. Check pod env with
`kubectl -n nemo-rl-fullstack exec fullstack-0 -- env | grep FI_`.

More troubleshooting: `docs/VALIDATION.md` has a full
failure-mode table.
