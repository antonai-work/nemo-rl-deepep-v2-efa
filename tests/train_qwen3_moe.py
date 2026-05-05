#!/usr/bin/env python3
"""Training driver for the all-PRs-applied stack.

Goal: exercise the NeMo-RL -> Megatron-Core -> DeepEP V2 import chain end
to end on the Qwen3-30B-A3B MoE shape, then run the proven Shape Y
training step.

Why wrap Shape Y rather than running full NeMo-RL GRPO:
  - GRPO needs a reward model + rollout loop + vLLM generation server. That
    is 10+ min of setup for a test whose point is "does the integrated stack
    import and step through DeepEP V2, not does GRPO converge".
  - The task spec accepts "bare Megatron training loop invoked *inside*
    NeMo-RL's policy container" when full GRPO is too heavyweight.
  - Shape Y already has proven evidence (loss decreases, grad_norm non-zero,
    EFA TX > 1 GB, HAVE_DEEP_EP_V2 True). We re-run it with NeMo-RL imports
    active to validate they share one process without LD/import conflicts.

Evidence captured:
  - nemo_rl top-level import works (i.e. every pure-Python dep we pinned
    resolves and the LD_LIBRARY_PATH doesn't break Ray/transformers).
  - nemo_rl version + megatron.core version + deep_ep module path.
  - Shape Y banner: HAVE_DEEP_EP_V2, buffer class name, loss curve.
"""
from __future__ import annotations

import os
import sys
import time


def log0(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def main() -> int:
    # 1. Prove NeMo-RL imports (this is the heaviest dep surface in the stack
    #    and exercises Ray / transformers / tokenizers / hydra all in one shot).
    log0("[rank0] === all-PRs-applied stack import probe ===")
    try:
        import nemo_rl  # noqa: F401
        # Try to get a version marker.
        try:
            import nemo_rl as _nr
            ver = getattr(_nr, "__version__", "<no __version__>")
        except Exception:
            ver = "<unavailable>"
        log0(f"[rank0] nemo_rl imported OK (version={ver})")
    except Exception as exc:
        log0(f"[rank0] FATAL: nemo_rl import failed: {exc!r}")
        return 10

    try:
        import megatron.core
        mc_ver = getattr(megatron.core, "__version__", "<no __version__>")
        log0(f"[rank0] megatron.core imported OK (version={mc_ver})")
    except Exception as exc:
        log0(f"[rank0] FATAL: megatron.core import failed: {exc!r}")
        return 11

    try:
        import deep_ep
        log0(
            "[rank0] deep_ep imported OK "
            f"(module_file={deep_ep.__file__!r} "
            f"has_ElasticBuffer={'ElasticBuffer' in dir(deep_ep)})"
        )
    except Exception as exc:
        log0(f"[rank0] FATAL: deep_ep import failed: {exc!r}")
        return 12

    # 2. Invariant: NO SHIM. Confirm that deep_ep.Buffer is the real V2
    #    class and not CompatBuffer.
    buf_cls = deep_ep.Buffer
    if "compat" in buf_cls.__module__.lower() or "compat" in buf_cls.__name__.lower():
        log0(
            "[rank0] FATAL: deep_ep.Buffer appears to be a CompatBuffer "
            f"({buf_cls.__module__}.{buf_cls.__name__}). "
            "This image must be shim-free."
        )
        return 13
    log0(
        f"[rank0] deep_ep.Buffer is {buf_cls.__module__}.{buf_cls.__name__} (not shim)"
    )

    if os.environ.get("DEEP_EP_USE_V2_SHIM", "0") != "0":
        log0(
            "[rank0] FATAL: DEEP_EP_USE_V2_SHIM="
            f"{os.environ.get('DEEP_EP_USE_V2_SHIM')} must be 0"
        )
        return 14

    if os.path.exists("/opt/api-shim"):
        log0("[rank0] FATAL: /opt/api-shim exists on disk; must be absent")
        return 15

    log0("[rank0] no-shim invariants hold")

    # 3. Hand off to the Shape Y training driver. It will init
    #    torch.distributed, build a Qwen3 MoE model, and run real training
    #    steps through Megatron's patched fused_a2a -> deep_ep.ElasticBuffer.
    log0("[rank0] === handing off to Shape Y train_step driver ===")
    t0 = time.perf_counter()

    # Import and invoke the Shape Y main() in-process so any NeMo-RL /
    # Ray side-effects at import time are actually exercised together
    # with the Shape Y code path.
    sys.path.insert(0, "/opt")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_step_shapeY", "/opt/train_step_shapeY.py"
    )
    if spec is None or spec.loader is None:
        log0("[rank0] FATAL: cannot load /opt/train_step_shapeY.py")
        return 20
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rc = mod.main()
    dt = time.perf_counter() - t0
    log0(f"[rank0] Shape Y train_step returned rc={rc} in {dt:.1f}s")

    if rc != 0:
        log0(f"[rank0] FATAL: Shape Y training step failed (rc={rc})")
        return rc

    log0("[rank0] === all-PRs-applied stack E2E training PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
