#!/bin/bash
# Preflight for nemo-rl-fullstack: three checks that must ALL pass before
# declaring the all-PRs-applied image ready for training.
#
# Exit 0 = all three pass. Exit non-zero = fail fast, which check failed.
#
# Check 1: NeMo-RL LD fix is live
#   -> ldconfig -p lists the aws-ofi-nccl plugin so NCCL can load it.
# Check 2: Shape Y probe is active
#   -> Megatron's patched fused_a2a.py has HAVE_DEEP_EP_V2 = True.
# Check 3: No shim
#   -> `deep_ep.Buffer` is the real V2 class, not CompatBuffer.
#   -> /opt/api-shim does not exist.
#   -> DEEP_EP_USE_V2_SHIM env is "0".

set -uo pipefail

HEADER="============================================================"

fail_count=0
pass_count=0

report() {
    local status="$1"
    local label="$2"
    local detail="$3"
    if [[ "${status}" == "PASS" ]]; then
        echo "  [PASS] ${label}"
        [[ -n "${detail}" ]] && echo "         ${detail}"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] ${label}"
        [[ -n "${detail}" ]] && echo "         ${detail}"
        fail_count=$((fail_count + 1))
    fi
}

echo "${HEADER}"
echo "nemo-rl-fullstack preflight  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "${HEADER}"

# Check 1: LD fix
echo
echo "Check 1: NeMo-RL LD_LIBRARY_PATH fix (PR #2359 resurrection)"
echo "  Env LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
ldconfig_out="$(ldconfig -p 2>/dev/null | grep -E 'libnccl-net-ofi\.so|libnccl-net\.so' || true)"
echo "  ldconfig -p | grep libnccl-net:"
if [[ -z "${ldconfig_out}" ]]; then
    report "FAIL" "libnccl-net plugin NOT on ld path" \
        "ldconfig sees no libnccl-net-ofi.so -- NCCL will fall back to sockets"
else
    echo "${ldconfig_out}" | sed 's/^/    /'
    report "PASS" "libnccl-net plugin on ld path" \
        "NCCL will discover OFI plugin on AWS EFA"
fi

# Check 2: Shape Y probe
echo
echo "Check 2: Shape Y V2 probe (Megatron PR)"
probe_out="$(python3 -c "
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP, HAVE_DEEP_EP_V2
print('HAVE_DEEP_EP=%s' % HAVE_DEEP_EP)
print('HAVE_DEEP_EP_V2=%s' % HAVE_DEEP_EP_V2)
print('STATUS=%s' % ('pass' if HAVE_DEEP_EP_V2 else 'fail'))
" 2>&1)"
echo "${probe_out}" | sed 's/^/    /'
if echo "${probe_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "HAVE_DEEP_EP_V2=True" \
        "Megatron's patched fused_a2a.py calls deep_ep.ElasticBuffer natively"
else
    report "FAIL" "HAVE_DEEP_EP_V2 is not True" \
        "Either Shape Y patch missing or deep_ep.ElasticBuffer is not importable"
fi

# Check 3a: deep_ep.Buffer is real V2 class
echo
echo "Check 3a: deep_ep.Buffer class is V2 (not CompatBuffer)"
cls_out="$(python3 -c "
import deep_ep
cls = deep_ep.Buffer
modname = getattr(cls, '__module__', '?')
clsname = getattr(cls, '__name__', '?')
is_compat = 'compat' in modname.lower() or 'compat' in clsname.lower()
print('deep_ep.Buffer module=%s name=%s' % (modname, clsname))
print('STATUS=%s' % ('fail' if is_compat else 'pass'))
print('Also: ElasticBuffer in dir(deep_ep)=%s' % ('ElasticBuffer' in dir(deep_ep)))
" 2>&1)"
echo "${cls_out}" | sed 's/^/    /'
if echo "${cls_out}" | grep -q 'STATUS=pass'; then
    report "PASS" "deep_ep.Buffer is real V2, no shim interception" ""
else
    report "FAIL" "deep_ep.Buffer appears to be a CompatBuffer (shim is active)" ""
fi

# Check 3b: /opt/api-shim not present
echo
echo "Check 3b: /opt/api-shim must not exist"
if [[ -e /opt/api-shim ]]; then
    report "FAIL" "/opt/api-shim exists" "$(ls -la /opt/api-shim 2>&1 | head -3)"
else
    report "PASS" "/opt/api-shim absent" "NO_SHIM_OK"
fi

# Check 3c: DEEP_EP_USE_V2_SHIM=0
echo
echo "Check 3c: DEEP_EP_USE_V2_SHIM env var must be 0"
shim_env="${DEEP_EP_USE_V2_SHIM:-<unset>}"
if [[ "${shim_env}" == "0" ]]; then
    report "PASS" "DEEP_EP_USE_V2_SHIM=0" "shim disabled by env"
else
    report "FAIL" "DEEP_EP_USE_V2_SHIM=${shim_env}" "must be 0 to prove native V2"
fi

echo
echo "${HEADER}"
echo "Preflight result: ${pass_count} PASS, ${fail_count} FAIL"
echo "${HEADER}"

exit "${fail_count}"
