"""v0.72.5 — the #331 repair: keep streamed NF4 weights out of ``MatMul4Bit``.

WHY THIS EXISTS

``bitsandbytes.autograd._functions.MatMul4Bit.forward`` stashes the packed weight
and the ``quant_state`` on ``ctx`` as PLAIN ATTRIBUTES::

    ctx.state = quant_state
    ctx.tensors = (None, B)

They never go through ``save_for_backward``, so ``torch.utils.checkpoint`` cannot
discard and recompute them. The reference is taken in the forward, it ALIASES the
streaming buffer pool, and it is read in the backward after that slot has already
been refilled with a different layer.

Symptom, measured on 8xH100 against a resident NF4 reference (see
``benchmarks/gate-h100-validation.md``): the forward stays bit-exact, the loss curve
looks healthy, and the gradients are wrong on every layer except the last
``stream_buffers``. It bites NF4 above roughly 165 MiB per layer, so 32B and 72B, and
never bf16 — which goes through ``MmBackward0`` with a normal ``save_for_backward``.

Both de-aliasing repairs were measured and rejected: because bnb holds the reference
across the whole forward-to-backward span, ANY de-aliasing keeps one copy of every
layer alive for that span and costs O(model), not O(window). On real 32B that was
peak VRAM 4 220 -> 19 720 MiB, which deletes the feature's premise.

THE REPAIR: do not send a streamed NF4 weight through ``MatMul4Bit`` at all.
Dequantise inside the checkpointed region and use a native matmul. ``F.linear``
saves the dequantised weight properly, so checkpointing DOES discard and recompute
it, and the transient lives only inside the recomputed block — O(window) by
construction.

WHY THIS IS NOT A NUMERICS CHANGE AT TRAINING SHAPES (STEP 13 of the record)

``bitsandbytes::gemm_4bit`` dispatches on M (tokens)::

    _gemm_4bit_custom_max_m = 1536      # CUDA
    if M > _gemm_4bit_custom_max_m: -> _dequant_linear_fallback

and on real projection shapes it takes that fallback at every M measured from 8 to
2048. So at 8B/32B shapes bitsandbytes is ALREADY doing what this repair does; the
repair makes it explicit and moves it inside the checkpoint. Measured over 423 rows,
the gradient is bit-exact in every one of them, worst ``max_abs`` exactly 0.0 — by
construction, since bnb's own backward is already dequantise-then-matmul.

The forward differs only where the fused kernel genuinely runs (small M), and then by
one bf16 ulp: worst 3.95e-3 relative to scale against 2^-8 = 3.9e-3.
"""

import os
import sys

import pytest

# The NF4 streamed/resident pair builders live in test_v07202 and are deliberately
# NOT duplicated here: two copies of a fixture drift, and this file's whole subject
# is a numerical comparison between the two models they build.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from test_v07202 import _nf4_stream, _resident_nf4  # noqa: E402


def _count_matmul_4bit(monkeypatch):
    """Count ``bnb.matmul_4bit`` calls.

    ``bitsandbytes/nn/modules.py`` does ``import bitsandbytes as bnb`` and then calls
    ``bnb.matmul_4bit(...)``, i.e. the attribute is looked up on the module object at
    CALL time, so patching the module attribute intercepts it.
    """
    import bitsandbytes as bnb

    calls = {"n": 0}
    real = bnb.matmul_4bit

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(bnb, "matmul_4bit", counting)
    return calls


class TestStreamedNF4AvoidsMatMul4Bit:
    """The repair, and the control that makes it mean something.

    ``0 calls`` on its own is equally consistent with "the counter never
    intercepted anything" — which is exactly how an earlier path control in this
    investigation was fooled. The resident control must COUNT, in the same test
    session, or the streamed assertion proves nothing.
    """

    def test_resident_nf4_does_reach_matmul_4bit(self, tmp_path, monkeypatch):
        """CONTROL. Without this, the assertion below is unfalsifiable."""
        _, _, weights, _, _ = _nf4_stream(tmp_path)
        resident = _resident_nf4(weights)
        calls = _count_matmul_4bit(monkeypatch)
        with torch.no_grad():
            resident(input_ids=torch.randint(0, 64, (1, 8)))
        assert calls["n"] > 0, (
            "the counter did not intercept bnb.matmul_4bit at all, so it cannot "
            "detect the streamed path avoiding it either"
        )

    def test_streamed_nf4_forward_does_not_reach_matmul_4bit(self, tmp_path, monkeypatch):
        model, _, _, _, _ = _nf4_stream(tmp_path)
        calls = _count_matmul_4bit(monkeypatch)
        with torch.no_grad():
            model(input_ids=torch.randint(0, 64, (1, 8)))
        assert calls["n"] == 0, (
            f"streamed NF4 still routed {calls['n']} call(s) through MatMul4Bit, which "
            "captures the packed weight outside save_for_backward and therefore aliases "
            "the buffer pool across the checkpoint boundary (#331)"
        )


# ==========================================================================
# #328 — HF gradient checkpointing must stay OFF on a streamed preference run
# ==========================================================================
PREFERENCE_TASKS = ("dpo", "orpo", "simpo", "kto")


class TestStreamedPreferenceLossesDisableHfGradientCheckpointing:
    """``StreamedDecoderLayer`` already wraps every layer in
    ``checkpoint(use_reentrant=False)``. ``should_enable_hf_gradient_checkpointing``
    exists so the HF Trainer does not ALSO checkpoint a streamed model — but only
    ``sft.py`` ever consulted it. The four preference wrappers never did, and TRL's
    ``DPOConfig`` / ``ORPOConfig`` / ``CPOConfig`` / ``KTOConfig`` all default
    ``gradient_checkpointing=True``, so it arrived switched on through the default
    rather than through anything the user wrote.

    On torch 2.5.1 that is the documented ~1.5x silent double-recompute. On torch
    2.13 + CUDA it is fatal: HF's checkpoint wraps the INNER ``LlamaDecoderLayer``,
    whose recompute runs long after ``functional_call``'s reparametrisation context
    has exited and restored the ``meta`` placeholders, so the recompute multiplies a
    ``meta`` ``input_layernorm.weight`` by a real ``cuda`` activation::

        RuntimeError: Tensor on device cuda:0 is not on the expected device meta!

    Measured on 8xH100: all four preference losses fail, SFT passes, and forcing the
    flag back off makes the DPO failure disappear with the wrapper's own checkpoint
    still active — which is what makes HF's checkpointing the cause rather than a
    correlate.
    """

    @pytest.mark.parametrize("task", PREFERENCE_TASKS)
    def test_streaming_turns_hf_gradient_checkpointing_off(self, task, tmp_path, monkeypatch):
        from test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task=task, device="cpu"
        )
        assert wrapper.trainer.args.gradient_checkpointing is False, (
            f"{task}: HF gradient checkpointing reached the Trainer switched ON for a "
            "streamed run, so HF checkpoints the inner decoder layer as well as the "
            "streaming wrapper. On torch 2.13 CUDA that recompute sees the restored "
            "meta placeholders and dies with 'expected device meta' (#328)"
        )

    @pytest.mark.parametrize("asked", [True, False])
    def test_a_non_streaming_run_gets_what_the_config_asked_for(
        self, asked, tmp_path, monkeypatch
    ):
        """CONTROL, and the second half of the same defect.

        Two things have to be true at once, and only asserting one of them would
        hide the other:

        * the repair must switch HF checkpointing off ONLY where streaming already
          checkpoints — pinning the flag to a constant ``False`` would satisfy the
          streamed test above while silently removing checkpointing from every
          ordinary DPO run;
        * the wrapper must not inherit TRL's default either way.

        Stating it as "the config value survives to the Trainer" is deliberately
        version-independent. TRL's own default for these configs is NOT stable —
        measured ``False`` on trl 0.19.1 and ``True`` on trl 0.26.2 — so asserting
        the default directly would be red on one supported stack and green on the
        other while the actual bug (the wrapper never setting it) went unnoticed on
        both. On the older stack a user's explicit ``gradient_checkpointing: true``
        was being silently dropped; on the newer one TRL's ``True`` arrived
        uninvited. One cause, two opposite symptoms.
        """
        from test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path,
            monkeypatch,
            task="dpo",
            device="cpu",
            stream_layers=False,
            gradient_checkpointing=asked,
        )
        assert wrapper.trainer.args.gradient_checkpointing is asked
