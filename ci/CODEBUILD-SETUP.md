# AWS CodeBuild setup for `nemo-rl-deepep-v2-efa`

This doc walks a cluster operator through provisioning the AWS infrastructure
that `ci/buildspec.yml` requires. Every AWS ARN, account ID, and repo name
is templated -- replace the placeholders with your own values.

## What this pipeline does

On every commit, CodeBuild:

1. Builds the **fast-path** image (`FROM ghcr.io/antonai-work/deepep-v2-efa-base`)
   and tags it `:allprs-<short-sha>` + `:latest`.
2. Builds the **from-vanilla** image (full 240-line inline base stack) as a
   build-only validation that the two paths stay in sync.
3. Runs `/opt/docker/preflight.sh` inside **both** images. Each must emit
   `5 PASS, 0 FAIL` or CI fails.
4. Pushes only the fast tag to ECR (vanilla is a build-time check).

Runtime: `BUILD_GENERAL1_2XLARGE` (8 vCPU, 16 GB) with `privilegedMode=true`
(needed for Docker-in-Docker) on `aws/codebuild/amazonlinux2-x86_64-standard:5.0`.
A cold vanilla build takes ~45 minutes; fast-path is ~10 minutes.

## Placeholder conventions

| Placeholder           | Example                                         | Meaning                              |
|-----------------------|-------------------------------------------------|--------------------------------------|
| `<AWS_ACCOUNT_ID>`    | `123456789012`                                  | Your 12-digit AWS account ID.        |
| `<AWS_REGION>`        | `us-east-2`                                     | Region for ECR repo + Secrets Manager.|
| `<ECR_REPO>`          | `nemo-rl-deepep-v2-efa`                         | ECR repo name (created below).       |
| `<PROJECT_NAME>`      | `nemo-rl-deepep-v2-efa-ci`                      | CodeBuild project name.              |
| `<ROLE_NAME>`         | `nemo-rl-deepep-v2-efa-codebuild-role`          | IAM role name for the build.         |
| `<SECRET_NAME>`       | `nemo-rl-deepep-v2-efa/ghcr-read-token`         | Secrets Manager secret name.         |
| `<GITHUB_OWNER>`      | `antonai-work`                                  | GitHub owner of this repo.           |

## Step 1: Create the ECR repo

```bash
aws ecr create-repository \
    --region <AWS_REGION> \
    --repository-name <ECR_REPO> \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability IMMUTABLE
```

Keeping tags immutable is recommended so a pushed `:allprs-<sha>` cannot be
silently overwritten. `:latest` is the only mutable reference, which is fine
because it always points to the most recent successful build.

## Step 2: Store the GHCR PAT in Secrets Manager

The fast-path build needs to pull `ghcr.io/antonai-work/deepep-v2-efa-base`.
GHCR packages are private by default on first publish. Until the org flips
the package to public, CodeBuild needs a GitHub Personal Access Token with
`read:packages` scope.

1. Generate a PAT at https://github.com/settings/tokens/new with scopes:
   - `read:packages`
   (Classic PATs with this single scope are sufficient. Fine-grained tokens
   also work; scope to "Read access to packages".)

2. Store in Secrets Manager:

```bash
aws secretsmanager create-secret \
    --region <AWS_REGION> \
    --name <SECRET_NAME> \
    --description "GHCR PAT for pulling deepep-v2-efa-base in CodeBuild" \
    --secret-string "ghp_..."
```

Note the returned ARN; you will pass it as `GHCR_READ_TOKEN_SECRET_ARN`
in Step 5. Once the package is flipped to public, this secret + the
`GHCR_READ_TOKEN_SECRET_ARN` env var can be removed and the `pre_build`
step becomes a no-op.

## Step 3: Create the IAM role

Trust policy (`trust.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "codebuild.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Inline policy (`policy.json`) -- replace `<AWS_ACCOUNT_ID>`, `<AWS_REGION>`,
`<ECR_REPO>`, `<SECRET_NAME>`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:<AWS_REGION>:<AWS_ACCOUNT_ID>:log-group:/aws/codebuild/*"
    },
    {
      "Sid": "EcrAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "EcrPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:GetDownloadUrlForLayer",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/<ECR_REPO>"
    },
    {
      "Sid": "SecretsManagerRead",
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:<AWS_REGION>:<AWS_ACCOUNT_ID>:secret:<SECRET_NAME>-*"
    }
  ]
}
```

Create the role and attach the inline policy:

```bash
aws iam create-role \
    --role-name <ROLE_NAME> \
    --assume-role-policy-document file://trust.json

aws iam put-role-policy \
    --role-name <ROLE_NAME> \
    --policy-name inline \
    --policy-document file://policy.json
```

## Step 4: (Optional) Connect your GitHub source

CodeBuild can read from `antonai-work/nemo-rl-deepep-v2-efa` via either
(a) OAuth (simplest: authorize in the console once), or (b) a GitHub App
connection. Either is fine; the snippet below assumes OAuth.

## Step 5: Create the CodeBuild project

```bash
aws codebuild create-project \
    --region <AWS_REGION> \
    --name <PROJECT_NAME> \
    --source '{
      "type": "GITHUB",
      "location": "https://github.com/<GITHUB_OWNER>/nemo-rl-deepep-v2-efa.git",
      "buildspec": "ci/buildspec.yml",
      "gitCloneDepth": 1
    }' \
    --artifacts '{ "type": "NO_ARTIFACTS" }' \
    --environment '{
      "type": "LINUX_CONTAINER",
      "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
      "computeType": "BUILD_GENERAL1_2XLARGE",
      "privilegedMode": true,
      "environmentVariables": [
        { "name": "AWS_ACCOUNT_ID", "value": "<AWS_ACCOUNT_ID>" },
        { "name": "AWS_REGION",     "value": "<AWS_REGION>" },
        { "name": "ECR_REPO",       "value": "<ECR_REPO>" },
        { "name": "GHCR_READ_TOKEN_SECRET_ARN",
          "value": "arn:aws:secretsmanager:<AWS_REGION>:<AWS_ACCOUNT_ID>:secret:<SECRET_NAME>" }
      ]
    }' \
    --service-role "arn:aws:iam::<AWS_ACCOUNT_ID>:role/<ROLE_NAME>" \
    --timeout-in-minutes 120
```

If the GHCR package is already public, omit the `GHCR_READ_TOKEN_SECRET_ARN`
entry; `ci/buildspec.yml` skips the GHCR login when the variable is unset.

## Step 6: Trigger a build

Ad-hoc from the CLI:

```bash
aws codebuild start-build \
    --region <AWS_REGION> \
    --project-name <PROJECT_NAME>
```

Hook up to GitHub push events (optional) via:

```bash
aws codebuild create-webhook \
    --region <AWS_REGION> \
    --project-name <PROJECT_NAME> \
    --filter-groups '[[
      { "type": "EVENT", "pattern": "PUSH" },
      { "type": "HEAD_REF", "pattern": "refs/heads/main" }
    ]]'
```

## Step 7: Watch a run

```bash
aws codebuild list-builds-for-project \
    --region <AWS_REGION> \
    --project-name <PROJECT_NAME>

aws codebuild batch-get-builds \
    --region <AWS_REGION> \
    --ids <BUILD_ID>
```

Or use the CodeBuild console. The `post_build` phase streams the preflight
output; a successful build ends with:

```
[post_build] Preflight 5/5 PASS
[post_build] Vanilla preflight 5/5 PASS
[post_build] Pushing fast tag to ECR (vanilla is build-only, not pushed)
[post_build] Done. Pushed <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPO>:allprs-<sha>
```

## Troubleshooting

- **`unauthorized: authentication required` from GHCR** -- the PAT expired
  or was created without `read:packages`. Regenerate and update the
  Secrets Manager entry.
- **`denied: requested access to the resource is denied` from ECR** -- the
  IAM role is missing the `EcrPush` policy statement or the `ECR_REPO`
  value does not match the actual repo name.
- **`no space left on device`** during the vanilla build -- the
  `BUILD_GENERAL1_2XLARGE` compute type ships with ~50 GB of ephemeral
  storage. The vanilla image with intermediate layers fits, but if you add
  additional steps, move to `BUILD_GENERAL1_LARGE_AARCH64` with more space
  or prune intermediate stages.
- **`ERROR: Preflight did not emit '5 PASS, 0 FAIL'`** -- inspect the
  preflight output printed just above the failure line. The most common
  cause is an LD_LIBRARY_PATH regression or a Megatron/NeMo-RL SHA drift.
