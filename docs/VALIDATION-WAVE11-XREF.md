# Wave 11 cross-reference (vllm cu12 NCCL test outcome)

Wave 11 was an inference-side test in the sibling `vllm-deepep-v2-efa`
repo; it is not a training-side test and did not exercise this
(NeMo-RL) image. It is linked here because it definitively answers
the open question "does Wave 9b's post-build ctypes claim that cu12
NCCL fixes the Wave 8 crash hold up on live hardware?" and that
answer informs the cu12-vs-cu13 decision for the training image
track too.

## Outcome in one line

**FAIL, same mode as Wave 8.** The Wave 9b image
(`058264135704.dkr.ecr.us-east-2.amazonaws.com/vllm-deepep-v2-efa:fast-7ef31a3acc2b`,
digest `sha256:37155d5449ab2c97b61a91ae71ab11e928fe3bf2a862656615e8ebc9ebe88bab`)
crashed on the first MoE dispatch with
`c10::Error: device id must be non-negative!-112` out of
`c10::cuda::SetDevice(signed char, bool)`, bubbling up as
`RuntimeError: Event device index` from
`deep_ep/buffers/elastic.py:811 -> self.runtime.dispatch(...)`.

## Why this matters for NeMo-RL

The Wave 8 and Wave 11 crashes both live at the
`c10::cuda::SetDevice(signed char, bool)` ABI boundary between
cu12 torch and cu13 vllm._C. Any NeMo-RL training image that mixes
cu12 torch with cu13-built C++ extensions (vllm, or any DeepEP-based
extension built against a different runtime than its host torch)
risks the same failure. Wave 10's cu13-across-the-board base is the
fix; keep an eye on what major-version the NeMo-RL image's torch
wheel resolves to when rebuilding on top of the next base bump.

## Full evidence

See the full Wave 11 doc in the sibling repo:
<https://github.com/antonai-work/vllm-deepep-v2-efa/blob/main/docs/VALIDATION-WAVE11-CU12-NCCL-TEST.md>

And the Wave 8 predecessor that first identified this mixed-runtime
failure mode:
<https://github.com/antonai-work/vllm-deepep-v2-efa/blob/main/docs/VALIDATION-WAVE8-FRESH-2026-05-06.md>
