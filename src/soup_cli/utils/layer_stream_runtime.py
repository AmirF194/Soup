"""soup train --stream-layers — streaming runtime (v0.72.0 BETA).

The torch half: pre-allocated VRAM buffer pool, the CPU-RAM weight source, the
prefetch scheduler, the layer wrapper, and the meta-device model build.

Data flow per step (plan 5.2)::

    FORWARD   layer i: wait(i) -> prefetch(i+1) -> checkpoint(body_i)
    BACKWARD  layer i: wait(i) -> prefetch(i-1) -> recompute + backward

Each layer is read TWICE per step and that cannot be optimised away:
``dL/dx = W^T . dL/dy``, so the backward pass needs W to reach lower layers and
their adapters. This is physics, not an implementation detail.

**No top-level torch / peft / transformers** — all lazy, so the light CLI keeps
importing without the training stack.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

_DTYPE_NAMES = ("bfloat16", "float16", "float32")


# ==========================================================================
# model-graph navigation
# ==========================================================================
def decoder_owner(model: Any) -> Any:
    """Return the module that owns ``.layers`` (LlamaModel / Qwen2Model).

    This is deliberately NOT the CausalLM wrapper. PEFT's ``LoraModel.forward``
    calls ``self.model.forward(...)`` directly, bypassing ``__call__`` and
    therefore every forward hook registered on the wrapper. transformers always
    reaches the layer container through ``__call__``, so hooks land here.
    """
    node = model
    for _ in range(8):
        if hasattr(node, "layers"):
            return node
        for attr in ("base_model", "model", "transformer"):
            child = getattr(node, attr, None)
            if child is not None and child is not node:
                node = child
                break
        else:
            break
    raise ValueError(
        "could not locate the decoder-layer container (a module with .layers) — "
        "layer streaming supports Llama/Qwen-shaped models only"
    )


def _set_module_param(root: Any, full_name: str, tensor: Any) -> None:
    import torch.nn as nn

    parts = full_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    module._parameters[parts[-1]] = nn.Parameter(tensor, requires_grad=False)


def _torch_dtype(name: str):
    import torch

    if name not in _DTYPE_NAMES:
        raise ValueError(f"unsupported dtype {name!r}; supported: {_DTYPE_NAMES}")
    return getattr(torch, name)


# ==========================================================================
# Tier 1 — the whole frozen base in CPU RAM (plan 5.5)
# ==========================================================================
class RamSource:
    """The base held in CPU RAM, allocated ONCE and filled by ``copy_``.

    The obvious ``load_file -> .to(dtype) -> .pin_memory()`` costs three
    transient copies of every layer. Measured on the dev box, that transient —
    not the store — is what pushed a 5.55 GB base past the 7.12 GB page-locked
    ceiling and made a 3B run impossible. So the store is pre-allocated at its
    final dtype and each source tensor is streamed into it one at a time.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Mapping[str, Tuple[Tuple[int, ...], str]],
        *,
        pin: bool = True,
    ):
        import torch
        from safetensors import safe_open

        from soup_cli.utils.layer_shard import layer_shard_path

        self.store: list = []
        self.nbytes = 0
        self.pinned = bool(pin)
        for idx in range(n_layers):
            held: Dict[str, Any] = {}
            with safe_open(layer_shard_path(shard_dir, idx), framework="pt") as handle:
                for name, (shape, dtype) in spec.items():
                    dst = torch.empty(
                        tuple(shape), dtype=_torch_dtype(dtype), pin_memory=self.pinned
                    )
                    src = handle.get_tensor(name)
                    dst.copy_(src)
                    del src
                    held[name] = dst
                    self.nbytes += dst.numel() * dst.element_size()
            self.store.append(held)

    @staticmethod
    def spec_from_shard(shard_dir: str, index: Any) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        """Shapes for ONE decoder layer, read from the shard header only."""
        from safetensors import safe_open

        from soup_cli.utils.layer_shard import layer_shard_path

        spec: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        with safe_open(layer_shard_path(shard_dir, 0), framework="pt") as handle:
            for name in handle.keys():
                shape = tuple(int(d) for d in handle.get_slice(name).get_shape())
                spec[name] = (shape, index.dtype)
        return spec

    def get(self, idx: int, name: str):
        return self.store[idx][name]


# ==========================================================================
# Tier 0 — pre-allocated VRAM buffers (plan 5.4)
# ==========================================================================
class LayerBufferPool:
    """N pre-allocated per-layer buffers. Never allocates inside the loop —
    that is what keeps the allocator from fragmenting (plan P7)."""

    def __init__(
        self,
        layer_spec: Mapping[str, Tuple[Tuple[int, ...], str]],
        n_buffers: int = 2,
        device: str = "cuda",
    ):
        import torch

        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.n = int(n_buffers)
        self.buffers = [
            {
                name: torch.empty(tuple(shape), dtype=_torch_dtype(dtype), device=device)
                for name, (shape, dtype) in layer_spec.items()
            }
            for _ in range(self.n)
        ]
        self.events = [torch.cuda.Event() for _ in range(self.n)] if self.is_cuda else []
        self.owner: list = [None] * self.n
        self.loads = 0
        self.nbytes = sum(
            buf.numel() * buf.element_size() for buf in self.buffers[0].values()
        ) * self.n

    def slot_for(self, idx: int) -> int:
        return idx % self.n

    def load_async(self, idx: int, source: RamSource, stream: Any = None) -> int:
        import torch

        slot = self.slot_for(idx)
        if self.is_cuda and stream is not None:
            # The slot's previous owner may still be in flight on the compute
            # stream; the prefetch must not clobber it (plan P1).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for name, dst in self.buffers[slot].items():
                    dst.copy_(source.get(idx, name), non_blocking=True)
                self.events[slot].record(stream)
        else:
            for name, dst in self.buffers[slot].items():
                dst.copy_(source.get(idx, name))
        self.owner[slot] = idx
        self.loads += 1
        return slot

    def wait(self, idx: int) -> Dict[str, Any]:
        """Block the compute stream until layer ``idx`` is resident.

        The ownership check is the plan-P1 tripwire: a buffer recycled while an
        autograd node still references it produces silently WRONG gradients, not
        a crash. Failing loudly here is the whole point.
        """
        import torch

        slot = self.slot_for(idx)
        if self.owner[slot] != idx:
            raise RuntimeError(
                f"layer-stream scheduler bug: buffer slot {slot} holds layer "
                f"{self.owner[slot]}, but layer {idx} was requested. Raise "
                f"training.stream_buffers (currently {self.n}) or report this."
            )
        if self.is_cuda:
            torch.cuda.current_stream().wait_event(self.events[slot])
        return self.buffers[slot]


class StreamPrefetcher:
    """Drives the prefetch. Forward walks 0..L-1; backward recompute walks
    L-1..0, so the direction is inferred from the call order."""

    def __init__(self, pool: Any, source: Any, n_layers: int, stream: Any = None):
        self.pool = pool
        self.source = source
        self.n_layers = int(n_layers)
        self.stream = stream
        self.prev: Optional[int] = None
        self.direction = 1
        self.primes = 0

    def prime(self) -> None:
        """Start of a forward pass: layer 0, walking upward."""
        self.prev = None
        self.direction = 1
        self.primes += 1
        self.pool.load_async(0, self.source, self.stream)

    def advance(self, idx: int) -> None:
        # Direction is explicit state, not re-derived per call. It only ever
        # flips downward, at the forward/backward turnaround, and is reset by
        # prime() at the start of the next step. Inferring it fresh each call
        # happens to work today only because the turnaround index is the last
        # layer; a future deeper lookahead would break that assumption
        # silently.
        if self.prev is not None and idx < self.prev:
            self.direction = -1
        self.prev = idx
        nxt = idx + self.direction
        if 0 <= nxt < self.n_layers and self.pool.owner[self.pool.slot_for(nxt)] != nxt:
            self.pool.load_async(nxt, self.source, self.stream)


# ==========================================================================
# the streamed layer (plan 5.6)
# ==========================================================================
def _build_streamed_layer_class():
    import torch
    import torch.nn as nn
    from torch.func import functional_call
    from torch.utils.checkpoint import checkpoint

    class StreamedDecoderLayer(nn.Module):
        def __init__(self, inner, idx, pool, prefetcher, name_map=None, use_checkpoint=True):
            super().__init__()
            self.inner = inner
            self.idx = int(idx)
            self.pool = pool
            self.prefetcher = prefetcher
            self.name_map = dict(name_map or {})
            self.use_checkpoint = bool(use_checkpoint)

        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            # `.to(device)` / `.to(dtype)` walk the module tree via _apply. The
            # wrapped layer's weights are META PLACEHOLDERS, substituted per
            # call from the buffer pool, and moving a meta tensor raises
            # NotImplementedError — which transformers' Trainer.__init__ AND
            # accelerate's prepare_model both trigger. Pass meta tensors
            # through untouched; everything real (the LoRA adapters, which live
            # inside this same subtree) still moves and casts normally.
            def _skip_meta(tensor: Any) -> Any:
                if getattr(tensor, "is_meta", False):
                    return tensor
                return fn(tensor)

            return super()._apply(_skip_meta, recurse=recurse)

        def state_dict(
            self,
            *args: Any,
            destination: Any = None,
            prefix: str = "",
            keep_vars: bool = False,
        ) -> Any:
            # v0.72.1 — serialise as though this wrapper were not in the tree.
            #
            # The wrapper holds the real layer as a child named `inner`, so
            # every adapter parameter would otherwise be written as
            # `...layers.0.inner.self_attn.q_proj.lora_A.weight`. That file
            # loads as ZERO tensors into any normal model — PEFT reports the
            # keys as missing and returns the untuned base, with no exception.
            # Every adapter artifact (the final `trainer.save_model()`, each
            # `save_steps` checkpoint, and therefore everything downstream:
            # `soup merge` / `serve` / `chat` / `adapters *` / the Registry)
            # reaches disk through this method, so delegating at OUR prefix is
            # what makes a streamed adapter indistinguishable from a normal
            # LoRA run.
            #
            # Serialisation-only, deliberately: the forward path is untouched,
            # so v0.72.0's bit-exactness gates remain valid. The cost is that
            # `named_parameters()` still shows `.inner.`, i.e. loading INTO a
            # streamed model stays unsupported (`--resume` is refused; the
            # checkpoint/resume slot is v0.72.3).
            #
            # The wrapper owns no parameters or buffers of its own — they all
            # live on `inner` — so nothing is lost by not serialising it. It
            # also means bypassing nn.Module.state_dict skips only hooks
            # registered on the WRAPPER itself, of which there are none (the
            # prefetch hook lives on the decoder container, not here).
            if args:
                # torch's legacy positional form: (destination, prefix, keep_vars)
                if destination is None:
                    destination = args[0]
                if len(args) > 1 and prefix == "":
                    prefix = args[1]
                if len(args) > 2 and keep_vars is False:
                    keep_vars = args[2]
            return self.inner.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        def __getattr__(self, name: str) -> Any:
            # transformers reads contract attributes straight off the layer
            # object (this version reads `decoder_layer.attention_type`). The
            # wrapper must be attribute-transparent or the model breaks at
            # forward time — and a wrapper returning a DEFAULT instead would
            # silently pick the wrong attention path.
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                inner = self._modules.get("inner")
                if inner is None:
                    raise
                return getattr(inner, name)

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            if self.use_checkpoint and torch.is_grad_enabled():
                return checkpoint(
                    self._body, hidden_states, *args, use_reentrant=False, **kwargs
                )
            return self._body(hidden_states, *args, **kwargs)

        def _body(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            buffers = self.pool.wait(self.idx)
            self.prefetcher.advance(self.idx)
            # Weights arrive with requires_grad=False, so autograd allocates no
            # grad buffers for them — but W STAYS IN THE GRAPH for W^T . dL/dy,
            # which is how the lower adapters receive gradient at all.
            weights = {meta: buffers[ckpt] for meta, ckpt in self.name_map.items()}
            return functional_call(
                self.inner, weights, (hidden_states, *args), kwargs
            )

    return StreamedDecoderLayer


_STREAMED_LAYER_CLASS = None


def _streamed_layer_class():
    global _STREAMED_LAYER_CLASS
    if _STREAMED_LAYER_CLASS is None:
        _STREAMED_LAYER_CLASS = _build_streamed_layer_class()
    return _STREAMED_LAYER_CLASS


class _StreamedDecoderLayerProxy:
    """Callable shim so ``StreamedDecoderLayer(...)`` works as a name."""

    def __call__(self, *args, **kwargs):
        return _streamed_layer_class()(*args, **kwargs)

    def __instancecheck__(self, instance):
        return isinstance(instance, _streamed_layer_class())


StreamedDecoderLayer = _StreamedDecoderLayerProxy()


# ==========================================================================
# allocator
# ==========================================================================
def probe_expandable_segments() -> bool:
    """Attempt ``expandable_segments:True`` (plan P7) and report whether it took.

    Windows silently ignores it — torch warns "expandable_segments not
    supported on this platform" and carries on. Never claim it is active when
    it is not.
    """
    if sys.platform.startswith("win"):
        return False
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if not torch.cuda.is_initialized():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


# ==========================================================================
# model construction — the resident load must NEVER happen (plan P14)
# ==========================================================================
def build_meta_skeleton(model_id: str, *, dtype: str, trust_remote_code: bool = False):
    """Build the model structure on ``meta``: no weight storage is allocated."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = _torch_dtype(dtype)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config.use_cache = False
    with init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(
                config, dtype=torch_dtype, trust_remote_code=trust_remote_code
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
            )
    return model


def materialize_extras(model: Any, shard_dir: str, index: Any, *, device: str, dtype: str) -> int:
    """Give real storage to everything that is NOT a decoder layer."""
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import extras_shard_path

    extras = load_file(extras_shard_path(shard_dir))
    torch_dtype = _torch_dtype(dtype)
    placed = 0
    pending_tied = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta or ".layers." in name:
            continue
        if name in extras:
            _set_module_param(model, name, extras[name].to(device=device, dtype=torch_dtype))
            placed += 1
        else:
            pending_tied.append(name)
    if pending_tied:
        # tie_word_embeddings=True -> lm_head.weight is absent from the
        # checkpoint by design and is restored from the input embeddings.
        model.tie_weights()
        still_meta = [n for n, p in model.named_parameters() if p.is_meta and ".layers." not in n]
        if still_meta:
            raise RuntimeError(
                f"non-layer weights left unmaterialised after tying: {still_meta[:4]}"
            )
    for name, buf in list(model.named_buffers()):
        if buf is not None and str(buf.device) != str(device):
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], buf.to(device))
    return placed


def materialize_meta_adapters(model: Any, *, seed: int = 0, device: str = "cuda") -> int:
    """Give real storage to LoRA adapters PEFT initialised on ``meta``.

    PEFT creates adapter weights on the base layer's device. In a streaming
    build that device is ``meta``, so the adapters have no storage: the
    optimizer happily accepts them and the run trains nothing. Re-initialise
    with PEFT's own scheme (A ~ kaiming_uniform, B = 0).
    """
    import torch
    import torch.nn as nn

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = 0
    for module_name, module in model.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            full = f"{module_name}.{pname}" if module_name else pname
            if not param.is_meta or "lora_" not in full:
                continue
            data = torch.empty(param.shape, dtype=torch.float32)
            if "lora_B" in full:
                data.zero_()
            else:
                nn.init.kaiming_uniform_(data, a=5**0.5, generator=generator)
            module._parameters[pname] = nn.Parameter(data.to(device), requires_grad=True)
            count += 1
    return count


# ==========================================================================
# installation
# ==========================================================================
@dataclass
class StreamRuntime:
    """Live streaming state for one model."""

    pool: Any
    source: Any
    prefetcher: Any
    n_layers: int
    pinned: bool
    device: str
    hook: Any = None

    def stats(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "buffers": self.pool.n,
            "buffer_bytes": self.pool.nbytes,
            "store_bytes": self.source.nbytes,
            "pinned": self.pinned,
            "layer_loads": self.pool.loads,
            "device": self.device,
        }


def _layer_name_map(layer: Any) -> Dict[str, str]:
    """meta parameter name inside one layer -> its shard key."""
    return {
        pname: pname.replace(".base_layer.", ".")
        for pname, param in layer.named_parameters()
        if param.is_meta
    }


def install_streaming(
    model: Any,
    *,
    shard_dir: str,
    index: Any,
    buffers: int = 2,
    pin: bool = True,
    device: str = "cuda",
    console: Any = None,
) -> StreamRuntime:
    """Wrap every decoder layer and wire the buffer pool + prefetch scheduler."""
    import torch

    owner = decoder_owner(model)
    layers = owner.layers
    n_layers = len(layers)
    if n_layers != index.n_layers:
        raise ValueError(
            f"model has {n_layers} decoder layers but the shard index has "
            f"{index.n_layers} — reshard the checkpoint"
        )

    name_map = _layer_name_map(layers[0])
    if not name_map:
        raise RuntimeError(
            "no meta decoder weights found — the base was materialised, which "
            "defeats layer streaming entirely"
        )
    shard_spec = RamSource.spec_from_shard(shard_dir, index)
    missing = sorted(set(name_map.values()) - set(shard_spec))
    if missing:
        raise ValueError(f"shard is missing decoder weights: {missing[:4]}")
    spec = {ckpt: shard_spec[ckpt] for ckpt in name_map.values()}

    source, pinned = _build_source(shard_dir, n_layers, spec, pin, console)
    pool = LayerBufferPool(spec, n_buffers=buffers, device=device)
    stream = torch.cuda.Stream() if str(device).startswith("cuda") else None
    prefetcher = StreamPrefetcher(pool, source, n_layers, stream)

    layer_cls = _streamed_layer_class()
    for idx in range(n_layers):
        layers[idx] = layer_cls(layers[idx], idx, pool, prefetcher, name_map)

    handle = owner.register_forward_pre_hook(lambda *_a, **_k: prefetcher.prime())

    # transformers' Trainer.__init__ calls _move_model_to_device -> model.to(),
    # and .to() on a module holding meta parameters raises NotImplementedError.
    # The decoder weights stay on meta BY DESIGN, so declare that this model
    # manages its own placement — exactly the marker a device_map-sharded model
    # carries, and the one _move_model_to_device short-circuits on.
    model.hf_device_map = {"": str(device)}

    return StreamRuntime(
        pool=pool,
        source=source,
        prefetcher=prefetcher,
        n_layers=n_layers,
        pinned=pinned,
        device=str(device),
        hook=handle,
    )


def _build_source(shard_dir, n_layers, spec, pin, console):
    """Build the RAM store, falling back to pageable and SAYING SO.

    Page-locking is bounded by the box, not by free RAM: the dev box topped out
    at 7.12 GB with 9.1 GB "available". Falling back is correct; hiding the
    ~97% -> ~79% GPU-utilisation cost is not.
    """
    if not pin:
        return RamSource(shard_dir, n_layers, spec, pin=False), False
    try:
        return RamSource(shard_dir, n_layers, spec, pin=True), True
    except (RuntimeError, MemoryError) as exc:
        message = (
            "layer streaming could not page-lock the base "
            f"({type(exc).__name__}); falling back to a PAGEABLE RAM store. "
            "Host-to-device copies become synchronous, which costs overlap — "
            "measured GPU utilisation drops from ~97% to ~79%. Free RAM or use "
            "a smaller base to keep the pinned store."
        )
        if console is not None:
            console.print(f"[yellow]{message}[/]")
        else:
            logger.warning(message)
        return RamSource(shard_dir, n_layers, spec, pin=False), False


def build_streamed_model(
    *,
    model_id: str,
    shard_dir: str,
    index: Any,
    lora_config: Any,
    device: str = "cuda",
    dtype: str = "bfloat16",
    buffers: int = 2,
    pin: bool = True,
    seed: int = 0,
    trust_remote_code: bool = False,
    console: Any = None,
) -> Tuple[Any, StreamRuntime]:
    """Meta skeleton -> extras -> LoRA -> streaming. No resident base load."""
    from peft import get_peft_model

    model = build_meta_skeleton(model_id, dtype=dtype, trust_remote_code=trust_remote_code)
    materialize_extras(model, shard_dir, index, device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, lora_config)
    materialize_meta_adapters(model, seed=seed, device=device)
    runtime = install_streaming(
        model,
        shard_dir=shard_dir,
        index=index,
        buffers=buffers,
        pin=pin,
        device=device,
        console=console,
    )
    return model, runtime
