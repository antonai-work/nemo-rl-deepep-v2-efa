# CodeBuild setup - nemo-rl-fullstack-v2

CodeBuild project that reproduces the "all-PRs-applied" `nemo-rl-fullstack`
image produced by the local-build agent. Mirrors the
`dynamo-inference-public` pattern from the awsi distribution-review
playbook (gist `e03bab9e96d0f8933f6dc15d02894b91`).

**Status:** spec only. Not fired yet. Wait for local-build evidence
(`docs/megatron-shapeY-v2-validation-*.md` equivalent + 2-pod p5 10-cycle
stress) to land green before creating the project.

---

## What this project builds

One image:

```
058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-fullstack:allprs-<shortsha>
058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-fullstack:latest
```

- FROM `deepep-base-v2:latest` (pulled from ECR at pre_build time).
- Layers: DeepEP V2 fork @ `aws-efa-auto-qp-cap-v2` + Megatron-LM fork @
  `deepep-v2-elasticbuffer-support` + NeMo-RL @ 46be4e8 + local
  LD_LIBRARY_PATH patch.
- No api-shim. `DEEP_EP_USE_V2_SHIM=0` baked into the image.

See `integrations/nemo-rl-fullstack/Dockerfile` for pinned SHAs.

---

## Prerequisites

1. AWS account 058264135704, region `us-east-2`.
2. The base image `058264135704.dkr.ecr.us-east-2.amazonaws.com/deepep-base-v2:latest`
   must already exist in ECR. Build it via a sibling CodeBuild project
   (not documented here; the base image is stable and is not part of the
   all-PRs test scope) or push manually one time:
   ```bash
   docker build -t deepep-base-v2:latest base/deepep-base-v2/
   aws ecr get-login-password --region us-east-2 | docker login \
     --username AWS --password-stdin 058264135704.dkr.ecr.us-east-2.amazonaws.com
   docker tag deepep-base-v2:latest \
     058264135704.dkr.ecr.us-east-2.amazonaws.com/deepep-base-v2:latest
   docker push \
     058264135704.dkr.ecr.us-east-2.amazonaws.com/deepep-base-v2:latest
   ```
3. GitHub connection to `antonai-work/deepep-v2-integration` authorized
   in CodeStar Connections (or swap to personal-access-token source).

---

## One-time bootstrap

### 1. Create ECR repos

```bash
REGION=us-east-2
for repo in nemo-rl-fullstack deepep-base-v2; do
  aws ecr create-repository --repository-name "${repo}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    || echo "  ${repo} already exists"
done
```

Apply a lifecycle policy (keep last 10 SHA-tagged + `latest`, expire
untagged after 1 day):

```bash
cat > /tmp/ecr-lifecycle.json <<'EOF'
{
  "rules": [
    { "rulePriority": 1, "description": "keep mutable tags",
      "selection": { "tagStatus": "tagged", "tagPatternList": ["latest","allprs-*"], "countType": "imageCountMoreThan", "countNumber": 10 },
      "action": { "type": "expire" } },
    { "rulePriority": 2, "description": "keep last 10 SHA-tagged",
      "selection": { "tagStatus": "tagged", "tagPatternList": ["*"], "countType": "imageCountMoreThan", "countNumber": 10 },
      "action": { "type": "expire" } },
    { "rulePriority": 3, "description": "expire untagged",
      "selection": { "tagStatus": "untagged", "countType": "sinceImagePushed", "countUnit": "days", "countNumber": 1 },
      "action": { "type": "expire" } }
  ]
}
EOF

for repo in nemo-rl-fullstack; do
  aws ecr put-lifecycle-policy --repository-name "${repo}" \
    --region "${REGION}" --lifecycle-policy-text file:///tmp/ecr-lifecycle.json
done
```

### 2. Create the CodeBuild service role

Trust policy - `codebuild.amazonaws.com` only:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "codebuild.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

Save as `trust.json`, then:

```bash
aws iam create-role \
  --role-name CodeBuildNemoRLFullstackRole \
  --assume-role-policy-document file://trust.json
```

Inline policy (minimum required; tighten `Resource` in prod):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/codebuild/*"
    },
    {
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": [
        "arn:aws:ecr:us-east-2:058264135704:repository/nemo-rl-fullstack",
        "arn:aws:ecr:us-east-2:058264135704:repository/deepep-base-v2"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject","s3:GetObject"],
      "Resource": "arn:aws:s3:::<your-sbom-bucket>/*"
    }
  ]
}
```

Save as `inline.json`, then:

```bash
aws iam put-role-policy \
  --role-name CodeBuildNemoRLFullstackRole \
  --policy-name NemoRLFullstackBuildInline \
  --policy-document file://inline.json
```

If you skip the S3 SBOM upload, drop the last statement.

### 3. Create the CodeBuild project

```bash
aws codebuild create-project \
  --name nemo-rl-fullstack-v2 \
  --region us-east-2 \
  --source type=GITHUB,location=https://github.com/antonai-work/deepep-v2-integration.git,buildspec=integrations/nemo-rl-fullstack/buildspec.yml \
  --artifacts type=NO_ARTIFACTS \
  --environment "type=LINUX_CONTAINER,computeType=BUILD_GENERAL1_2XLARGE,image=aws/codebuild/amazonlinux2-x86_64-standard:5.0,privilegedMode=true,environmentVariables=[
    {name=AWS_ACCOUNT_ID,value=058264135704},
    {name=AWS_DEFAULT_REGION,value=us-east-2},
    {name=ECR_REPO,value=nemo-rl-fullstack},
    {name=BASE_ECR_REPO,value=deepep-base-v2},
    {name=CVE_ALLOW_CRITICAL,value=0}
  ]" \
  --service-role arn:aws:iam::058264135704:role/CodeBuildNemoRLFullstackRole \
  --timeout-in-minutes 120
```

To add S3 SBOM upload later, patch with:

```bash
aws codebuild update-project --name nemo-rl-fullstack-v2 \
  --environment "...,environmentVariables=[
    ...existing...,
    {name=S3_SBOM_BUCKET,value=s3://your-sbom-bucket/nemo-rl-fullstack}
  ]"
```

**Compute-type notes:**
- `BUILD_GENERAL1_2XLARGE` (72 vCPU / 145 GB / 824 GB SSD) - required.
  The fullstack image is ~35 GB; `LARGE` (200 GB scratch) overflows
  during `exporting layers` — same failure mode the dynamo-inference
  combined build hit (playbook section 5.6).
- `privilegedMode: true` - required for Docker-in-Docker.
- `timeout: 120 min` - cold build ~45 min (base pull ~2 min, Megatron +
  NeMo-RL + ray + transformers layer ~35 min, validation + push + trivy
  ~8 min).

### 4. Trigger the first build

**(a)** Console - "Start build" on the `nemo-rl-fullstack-v2` project.

**(b)** CLI - pin to a specific commit so the image tag is reproducible:

```bash
aws codebuild start-build \
  --project-name nemo-rl-fullstack-v2 \
  --region us-east-2 \
  --source-version main \
  --query 'build.id' --output text
```

Monitor:

```bash
BUILD_ID=<from-above>
aws codebuild batch-get-builds --ids "${BUILD_ID}" --region us-east-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}' --output json

# CloudWatch logs:
aws logs tail /aws/codebuild/nemo-rl-fullstack-v2 \
  --log-stream-name-prefix "${BUILD_ID#*:}" --region us-east-2 --since 10m
```

### 5. Verify the artifacts

```bash
# ECR image:
aws ecr describe-images --repository-name nemo-rl-fullstack --region us-east-2 \
  --query 'reverse(sort_by(imageDetails, &imagePushedAt))[:3]' --output table

# Pull + smoke-check locally:
aws ecr get-login-password --region us-east-2 | docker login \
  --username AWS --password-stdin 058264135704.dkr.ecr.us-east-2.amazonaws.com
docker pull 058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-fullstack:latest
docker run --rm --entrypoint "" \
  058264135704.dkr.ecr.us-east-2.amazonaws.com/nemo-rl-fullstack:latest \
  python3 -c "from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2; assert HAVE_DEEP_EP_V2; print('OK')"
```

---

## Validation contract

The post_build phase enforces the same in-image contract the local-build
agent validates, minus the cluster-level checks (EFA fabric is not
available on CodeBuild workers):

| Gate | How |
|------|-----|
| `HAVE_DEEP_EP_V2 == True` | `python3 -c "from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2; assert HAVE_DEEP_EP_V2"` |
| `deep_ep.Buffer` is native ElasticBuffer (not shim CompatBuffer) | `python3 -c "import deep_ep; ..."` |
| `/opt/api-shim` absent | `test ! -e /opt/api-shim` |
| `ldconfig -p` knows `libnccl-net` | `ldconfig -p \| grep libnccl-net` |
| `LD_LIBRARY_PATH` has `/opt/amazon/aws-ofi-nccl` | `bash -c "echo \$LD_LIBRARY_PATH \| grep -q /opt/amazon/aws-ofi-nccl"` |

The cluster-level check — `NCCL INFO NET/OFI Using Amazon EFA` must
appear at NCCL init — stays on the 2-pod p5 validation path (see
`docs/10-cycle-stress-passed.md` style evidence). CodeBuild cannot see
EFA and does not attempt that check.

---

## Troubleshooting

### `manifest unknown` on `docker pull deepep-base-v2:latest`

The base image hasn't been pushed yet. See Prerequisites section 2.

### `no space left on device` during combined export

You're on `BUILD_GENERAL1_LARGE`. Upgrade to `BUILD_GENERAL1_2XLARGE`.

### `Cannot connect to the Docker daemon`

Set `privilegedMode=true` on the project.

### `network: dial tcp: lookup github.com: i/o timeout` during clone

CodeBuild is behind a VPC without a NAT gateway. Either run the project
in CodeBuild-managed network (no VPC config) or add a NAT + VPC
endpoints for S3 / ECR / CloudWatch / GitHub.

### CVE gate fails on a known false-positive

Set `CVE_ALLOW_CRITICAL=1` on the project for a temporary override.

### Build takes longer than expected

Cold build should be ~45 min. Verify BuildKit `--cache-from` pulled the
previous `:latest`. Log should show `CACHED` on early layers for warm
builds. If not, ECR auth probably failed in `pre_build` — re-check the
IAM role policy.

---

## Referenced files (paths relative to integrations/nemo-rl-fullstack/)

- `Dockerfile` - the all-PRs-applied image definition (local-build agent
  authored; we use verbatim).
- `build.sh` - docker build driver used by both local dev and CodeBuild.
- `buildspec.yml` - CodeBuild pipeline entry.
- `ci/codebuild-post-build.sh` - in-image validation gates + push + trivy
  + SBOM + optional S3 upload.
- `nemo-rl-pr-prep.final.patch` - NeMo-RL LD_LIBRARY_PATH + recipe patch;
  COPY'd into the image at build time. Must be committed to the repo for
  CodeBuild to see it.

---

## Relationship to the local-build agent

The local-build agent and this CodeBuild project build the SAME image
from the SAME Dockerfile. The split is:

| Stage | Local agent | CodeBuild |
|-------|-------------|-----------|
| `docker build` | yes | yes (identical) |
| In-image import gates | yes | yes (post_build.sh) |
| EFA fabric smoke | yes (2-pod p5 + `NCCL INFO NET/OFI Using Amazon EFA`) | no (no EFA on workers) |
| 10-cycle D+C stress | yes (2-pod p5 cluster) | no (no cluster) |
| External trivy scan | optional | yes (CVE gate) |
| SBOM upload to S3 | no | yes (if `S3_SBOM_BUCKET` set) |

Ship gate: both the local 10-cycle cluster evidence AND the CodeBuild
green must be present before the image is considered distribution-ready.
