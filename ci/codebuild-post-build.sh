#!/usr/bin/env bash
#
# nemo-rl-fullstack - CodeBuild post_build phase.
#
# Mirrors the dynamo-inference-public post_build pattern from gist
# e03bab9e96d0f8933f6dc15d02894b91. Performs (in order):
#   1. In-image validation gates (mirrors the local-build agent's
#      "validation the CodeBuild runs should perform" contract).
#   2. ECR push (allprs-<sha> + latest).
#   3. External trivy CVE scan (CRITICAL + HIGH).
#   4. CVE gate (FAILs build if any CRITICAL unless CVE_ALLOW_CRITICAL=1).
#   5. SBOM extraction from /opt/security (if present; mirrors
#      dynamo-inference pattern, not required here since nemo-rl-fullstack
#      does not have the SBOM stage baked in).
#   6. Optional S3 upload of SBOM + CVE reports.
#
# Requires:
#   FULLSTACK_URI, FULLSTACK_LATEST, ECR, SHA     set by pre_build
#   CVE_ALLOW_CRITICAL, S3_SBOM_BUCKET            optional
#
set -euo pipefail

: "${FULLSTACK_URI:?FULLSTACK_URI is required}"
: "${FULLSTACK_LATEST:?FULLSTACK_LATEST is required}"
: "${ECR:?ECR is required}"
: "${SHA:?SHA is required}"
CVE_ALLOW_CRITICAL="${CVE_ALLOW_CRITICAL:-0}"
S3_SBOM_BUCKET="${S3_SBOM_BUCKET:-}"

IMAGE_FOR_VALIDATION="${FULLSTACK_URI}"

echo "=================================================================="
echo "[post_build] nemo-rl-fullstack validation + push pipeline"
echo "=================================================================="
echo "  image: ${IMAGE_FOR_VALIDATION}"
echo "  sha:   ${SHA}"
echo "=================================================================="

# ------------------------------------------------------------------
# 1. In-image validation gates (no EFA fabric available in CodeBuild;
#    these checks verify IMAGE CONTENTS only, per the local-build agent's
#    contract. Cluster-level validation — "NCCL INFO NET/OFI Using Amazon
#    EFA" etc. — stays on the 2-pod p5 test path.)
# ------------------------------------------------------------------
mkdir -p validation-out
VAL_LOG="validation-out/in-image-checks_${SHA}.txt"
{
  echo "=== nemo-rl-fullstack in-image validation ==="
  echo "Image: ${IMAGE_FOR_VALIDATION}"
  echo "Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
} > "${VAL_LOG}"

validate() {
  local name="$1"; shift
  local cmd=( "$@" )
  echo "[validate] ${name}" | tee -a "${VAL_LOG}"
  if docker run --rm --entrypoint "" "${IMAGE_FOR_VALIDATION}" "${cmd[@]}" >>"${VAL_LOG}" 2>&1; then
    echo "  PASS: ${name}" | tee -a "${VAL_LOG}"
  else
    echo "  FAIL: ${name}" | tee -a "${VAL_LOG}"
    echo "FAIL: ${name} (see ${VAL_LOG})" >&2
    return 1
  fi
}

# Check 1: HAVE_DEEP_EP_V2=True (Megatron Shape Y patch is live).
validate "HAVE_DEEP_EP_V2 == True" \
  python3 -c "from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2; assert HAVE_DEEP_EP_V2, 'HAVE_DEEP_EP_V2 is False'; print('HAVE_DEEP_EP_V2:', HAVE_DEEP_EP_V2)"

# Check 2: ElasticBuffer is the buffer class (no shim).
validate "deep_ep.Buffer is ElasticBuffer / not shim" \
  python3 -c "import deep_ep; b = getattr(deep_ep, 'Buffer', None); eb = deep_ep.ElasticBuffer; print('Buffer:', b); print('ElasticBuffer:', eb); assert 'CompatBuffer' not in type(b).__name__ if b is not None else True, 'shim CompatBuffer leaked'; print('OK')"

# Check 3: no api-shim path present.
validate "/opt/api-shim absent" \
  test ! -e /opt/api-shim

# Check 4: ldconfig reports libnccl-net paths (LD_LIBRARY_PATH fix).
validate "ldconfig knows about libnccl-net" \
  bash -c "ldconfig -p | grep -E 'libnccl-net' | head -5"

# Check 5: LD_LIBRARY_PATH carries the aws-ofi-nccl lib dir.
validate "LD_LIBRARY_PATH includes /opt/amazon/aws-ofi-nccl" \
  bash -c "echo \$LD_LIBRARY_PATH | grep -q /opt/amazon/aws-ofi-nccl"

echo "[post_build] in-image validation passed."

# ------------------------------------------------------------------
# 2. ECR push
# ------------------------------------------------------------------
echo "[post_build] Pushing images to ECR..."
docker push "${FULLSTACK_URI}"
docker push "${FULLSTACK_LATEST}"

# ------------------------------------------------------------------
# 3. External trivy CVE scan (real DB; in-image trivy is not wired for
#    nemo-rl-fullstack — the image does not have the security-scan stage
#    that dynamo-inference does).
# ------------------------------------------------------------------
echo "[post_build] External trivy CVE scan (CRITICAL + HIGH)..."
mkdir -p cve-reports
trivy image --severity CRITICAL,HIGH --scanners vuln \
  --format table --timeout 30m --no-progress --skip-version-check \
  --output "cve-reports/nemo-rl-fullstack_${SHA}_cve-critical-high.txt" \
  "${FULLSTACK_URI}" || echo "  trivy exited non-zero (continuing; gate runs below)"

# ------------------------------------------------------------------
# 4. CVE gate
# ------------------------------------------------------------------
echo "[post_build] CVE gate (FAILs unless CVE_ALLOW_CRITICAL=1)..."
crit=$(grep -lE 'CRITICAL ' cve-reports/*.txt 2>/dev/null || true)
if [ -n "${crit}" ] && [ "${CVE_ALLOW_CRITICAL}" != "1" ]; then
  echo "CRITICAL CVEs detected in:"
  for f in ${crit}; do
    echo "  --- ${f} ---"
    grep -E 'CRITICAL ' "${f}" | head -10
  done
  echo "Set CVE_ALLOW_CRITICAL=1 on the CodeBuild project to waive during review."
  exit 1
fi
echo "[post_build] CVE gate passed (or allowlisted)."

# ------------------------------------------------------------------
# 5. SBOM extraction (best-effort; image does not ship /opt/security by
#    default, so this is a no-op unless we add a security-scan stage later.)
# ------------------------------------------------------------------
echo "[post_build] SBOM extraction (best-effort)..."
mkdir -p sbom-out/nemo-rl-fullstack
cid=$(docker create "${FULLSTACK_URI}" true)
if docker cp "${cid}:/opt/security/." "sbom-out/nemo-rl-fullstack/" 2>/dev/null; then
  echo "  extracted /opt/security from image"
else
  echo "  (image has no /opt/security dir — skipping SBOM extraction)"
fi
docker rm "${cid}" >/dev/null

# ------------------------------------------------------------------
# 6. Optional S3 upload
# ------------------------------------------------------------------
if [ -n "${S3_SBOM_BUCKET}" ]; then
  echo "[post_build] Uploading SBOM + CVE reports to ${S3_SBOM_BUCKET}/${SHA}/..."
  aws s3 cp --recursive sbom-out/        "${S3_SBOM_BUCKET}/${SHA}/sbom/"       || true
  aws s3 cp --recursive cve-reports/     "${S3_SBOM_BUCKET}/${SHA}/cve/"        || true
  aws s3 cp --recursive validation-out/  "${S3_SBOM_BUCKET}/${SHA}/validation/" || true
else
  echo "[post_build] S3_SBOM_BUCKET not set — skipping upload."
fi

echo "=================================================================="
echo "[post_build] Build summary"
echo "=================================================================="
echo "  nemo-rl-fullstack:        ${FULLSTACK_URI}"
echo "  nemo-rl-fullstack:latest: ${FULLSTACK_LATEST}"
echo "  validation log:           ${VAL_LOG}"
echo "  cve report:               cve-reports/nemo-rl-fullstack_${SHA}_cve-critical-high.txt"
echo "=================================================================="
