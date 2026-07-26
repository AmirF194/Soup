"""soup train --stream-layers — checkpoint sharder (v0.72.0 BETA).

Rewrites an HF checkpoint into one ``layer_NNN.safetensors`` per decoder layer
plus a single ``extras.safetensors`` (embeddings / final norm / untied head),
so the runtime can stream exactly one layer at a time.

Built on the ``utils/spectrum_scan.iter_weight_matrices`` pattern: ``safe_open``
per source shard, one tensor materialised at a time, symlinked shards skipped,
element caps enforced. Peak RSS while sharding is ONE decoder layer, not the
model — which matters, because the whole point is that the model does not fit.

**No top-level torch / safetensors** — both are ``[train]``-extra deps and this
module sits on the light CLI's import path.
"""

import contextlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from soup_cli import __version__

logger = logging.getLogger(__name__)

#: A decoder-layer parameter key: ``model.layers.<idx>.<rest>``.
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")

_SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")

#: Refuse absurd checkpoints rather than thrash (mirrors spectrum_scan caps).
_MAX_LAYERS = 512
_MAX_TENSOR_ELEMENTS = 2**31
_MAX_SHARD_FILES = 4096
_MAX_TOTAL_TENSORS = 200_000

_INDEX_NAME = "index.json"
_EXTRAS_NAME = "extras.safetensors"


# ==========================================================================
# index
# ==========================================================================
@dataclass(frozen=True)
class ShardIndex:
    """What the sharder produced — the runtime's contract with the cache."""

    n_layers: int
    layer_keys: Tuple[str, ...]
    extra_keys: Tuple[str, ...]
    dtype: str
    total_params: int
    arch: str
    soup_version: str
    source_fingerprint: str = ""


def layer_shard_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"layer_{idx:03d}.safetensors")


def extras_shard_path(out_dir: str) -> str:
    return os.path.join(out_dir, _EXTRAS_NAME)


def read_shard_index(out_dir: str) -> ShardIndex:
    """Read ``index.json``. Raises on a missing or malformed index."""
    path = os.path.join(out_dir, _INDEX_NAME)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return ShardIndex(
        n_layers=int(payload["n_layers"]),
        layer_keys=tuple(payload["layer_keys"]),
        extra_keys=tuple(payload["extra_keys"]),
        dtype=str(payload["dtype"]),
        total_params=int(payload["total_params"]),
        arch=str(payload.get("arch", "")),
        soup_version=str(payload.get("soup_version", "")),
        source_fingerprint=str(payload.get("source_fingerprint", "")),
    )


# ==========================================================================
# cache location
# ==========================================================================
def model_slug(model: str) -> str:
    """Sanitise a model id into a traversal-safe cache directory name."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    base = model.strip().replace("\\", "/").rstrip("/").replace("/", "__")
    slug = _SLUG_RE.sub("_", base).replace("..", "_").strip("._-")
    return (slug or "model")[:128]


def default_layer_stream_cache_dir() -> str:
    """``~/.soup/layer-stream`` — the shard cache root (no side effects)."""
    return os.path.join(os.path.expanduser("~"), ".soup", "layer-stream")


def _resolve_cache_root(cache_dir: Optional[str], is_under: Any) -> str:
    """Cache ROOT only (explicit arg > env override > default). Always a str."""
    if cache_dir is not None:
        return os.path.realpath(os.path.expanduser(str(cache_dir)))
    override = os.environ.get("SOUP_LAYER_STREAM_CACHE_DIR")
    if override and not any(ord(ch) < 0x20 for ch in override):
        candidate = os.path.realpath(os.path.expanduser(override))
        bounds = [
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
            os.path.realpath(tempfile.gettempdir()),
        ]
        if any(is_under(candidate, bound) for bound in bounds):
            return candidate
    return default_layer_stream_cache_dir()


def resolve_shard_dir(model: str, cache_dir: Optional[str] = None) -> str:
    """Per-model shard dir (explicit arg > env override > default).

    ``SOUP_LAYER_STREAM_CACHE_DIR`` is rejected when it holds C0 control
    characters or escapes ``$HOME`` / ``$CWD`` / ``$TMPDIR`` — silently, because
    an env var is operator config, not API input (mirrors spectrum_scan).
    """
    from soup_cli.utils.paths import is_under

    slug = model_slug(model)
    root: str = _resolve_cache_root(cache_dir, is_under)
    chosen = os.path.join(root, slug)
    os.makedirs(chosen, exist_ok=True)
    return chosen


# ==========================================================================
# sharding
# ==========================================================================
def _discover_safetensors(weights_dir: str) -> List[str]:
    """Sorted, non-symlinked ``*.safetensors`` under ``weights_dir``."""
    root = os.path.realpath(os.path.expanduser(weights_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    found = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(root, name)
        if os.path.islink(path):
            logger.warning("layer-stream sharder: skipping symlinked shard %s", name)
            continue
        if not os.path.isfile(path):
            continue
        found.append(path)
    if not found:
        raise FileNotFoundError(
            f"no .safetensors weight files found in {weights_dir} — layer "
            f"streaming needs a safetensors checkpoint"
        )
    if len(found) > _MAX_SHARD_FILES:
        raise ValueError(f"too many shard files ({len(found)} > {_MAX_SHARD_FILES})")
    return found


def source_weight_bytes(weights_dir: str) -> int:
    """Total bytes of the source ``*.safetensors`` — a cheap size probe that
    lets a caller refuse an oversized base BEFORE spending minutes sharding."""
    return sum(os.path.getsize(path) for path in _discover_safetensors(weights_dir))


def _source_fingerprint(shards: List[str]) -> str:
    """Identity of the SOURCE checkpoint: basename + size + mtime per shard.

    The shard cache is keyed by a model slug, so without this a local
    checkpoint retrained in place (or two ids colliding onto one slug) would
    silently reuse stale shards and stream the WRONG WEIGHTS into training —
    no error, just a loss curve describing a different model.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(shards):
        stat = os.stat(path)
        digest.update(os.path.basename(path).encode("utf-8", "replace"))
        digest.update(f"|{stat.st_size}|{int(stat.st_mtime_ns)}|".encode())
    return digest.hexdigest()


def _validate_out_dir(out_dir: str) -> str:
    """Bound our OWN writes rather than trusting the caller.

    ``realpath`` (not ``abspath``) so a symlinked ANCESTOR is resolved instead
    of transparently followed, and the result is then required to sit under
    $HOME / $CWD / $TMPDIR — the same bound ``resolve_shard_dir`` applies, but
    enforced here so a direct caller cannot bypass it.
    """
    from soup_cli.utils.paths import is_under

    if any(ord(ch) < 0x20 for ch in out_dir):
        raise ValueError("shard output directory must not contain control characters")
    expanded = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.islink(expanded):
        raise ValueError(f"shard output directory must not be a symlink: {out_dir}")
    parent = os.path.dirname(expanded) or "."
    resolved = os.path.join(os.path.realpath(parent), os.path.basename(expanded))
    bounds = [
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
        os.path.realpath(tempfile.gettempdir()),
    ]
    if not any(is_under(resolved, bound) for bound in bounds):
        raise ValueError(
            f"shard output directory must be under $HOME, $CWD or $TMPDIR; "
            f"got {out_dir}"
        )
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _atomic_save(blob: Dict[str, Any], path: str) -> None:
    """safetensors write via a temp file in the same dir + os.replace."""
    from safetensors.torch import save_file

    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        save_file(blob, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_index(index: ShardIndex, out_dir: str) -> None:
    payload = asdict(index)
    payload["layer_keys"] = list(index.layer_keys)
    payload["extra_keys"] = list(index.extra_keys)
    path = os.path.join(out_dir, _INDEX_NAME)
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _cached_index(out_dir: str, dtype: str, fingerprint: str) -> Optional[ShardIndex]:
    """Return a reusable index, or None when absent / corrupt / stale.

    A dtype mismatch MUST invalidate: streaming float32 shards into a bfloat16
    pool would quietly train the wrong precision rather than fail.
    """
    try:
        index = read_shard_index(out_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if index.dtype != dtype:
        return None
    if index.source_fingerprint != fingerprint:
        return None  # source checkpoint changed under us
    if not os.path.exists(extras_shard_path(out_dir)):
        return None
    for idx in range(index.n_layers):
        if not os.path.exists(layer_shard_path(out_dir, idx)):
            return None
    return index


def shard_checkpoint(
    weights_dir: str,
    out_dir: str,
    *,
    dtype: str = "bfloat16",
    arch: str = "",
    force: bool = False,
) -> ShardIndex:
    """Rewrite an HF checkpoint into per-layer safetensors shards."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r} for layer streaming; "
            f"supported: {', '.join(_SUPPORTED_DTYPES)}"
        )
    shards = _discover_safetensors(weights_dir)
    resolved_out = _validate_out_dir(out_dir)
    fingerprint = _source_fingerprint(shards)
    if not force:
        cached = _cached_index(resolved_out, dtype, fingerprint)
        if cached is not None:
            return cached

    from safetensors import safe_open

    # pass 1 — build key -> shard without materialising a single tensor
    where: Dict[str, str] = {}
    layer_ids = set()
    for path in shards:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                where[key] = path
                if len(where) > _MAX_TOTAL_TENSORS:
                    raise ValueError(
                        f"checkpoint declares more than {_MAX_TOTAL_TENSORS} tensors"
                    )
                match = _LAYER_RE.match(key)
                if match:
                    layer_ids.add(int(match.group(1)))

    if not layer_ids:
        raise ValueError(
            f"no decoder layer weights (model.layers.N.*) found in {weights_dir} — "
            f"layer streaming has nothing to stream"
        )
    n_layers = max(layer_ids) + 1
    if n_layers > _MAX_LAYERS:
        raise ValueError(f"too many decoder layers ({n_layers} > {_MAX_LAYERS})")
    if len(layer_ids) != n_layers:
        missing = sorted(set(range(n_layers)) - layer_ids)
        raise ValueError(f"decoder layer indices are not contiguous; missing {missing[:8]}")

    total_params = 0
    layer_keys: Tuple[str, ...] = ()
    # ExitStack, not a dict comprehension of __enter__(): if shard N fails to
    # open, everything opened before it must still be closed.
    with contextlib.ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt")) for path in shards
        }
        for idx in range(n_layers):
            prefix = f"model.layers.{idx}."
            blob = {}
            for key, path in where.items():
                if not key.startswith(prefix):
                    continue
                tensor = _read_tensor(handles[path], key, dtype)
                blob[key[len(prefix):]] = tensor
                total_params += tensor.numel()
            keys = tuple(sorted(blob))
            if idx == 0:
                layer_keys = keys
            elif keys != layer_keys:
                raise ValueError(
                    f"decoder layer {idx} has a different parameter set than layer 0 "
                    f"— layer streaming needs a uniform layer shape"
                )
            _atomic_save(blob, layer_shard_path(resolved_out, idx))
            del blob

        extras = {}
        for key, path in where.items():
            if _LAYER_RE.match(key):
                continue
            tensor = _read_tensor(handles[path], key, dtype)
            extras[key] = tensor
            total_params += tensor.numel()
        extra_keys = tuple(sorted(extras))
        _atomic_save(extras, extras_shard_path(resolved_out))
        del extras

    index = ShardIndex(
        n_layers=n_layers,
        layer_keys=layer_keys,
        extra_keys=extra_keys,
        dtype=dtype,
        total_params=total_params,
        arch=arch,
        soup_version=__version__,
        source_fingerprint=fingerprint,
    )
    _atomic_write_index(index, resolved_out)
    return index


def _read_tensor(handle: Any, key: str, dtype: str) -> "Any":
    """Materialise one tensor, size-capped, converted to the target dtype."""
    import torch

    shape = handle.get_slice(key).get_shape()
    elements = math.prod(int(dim) for dim in shape)
    if elements > _MAX_TENSOR_ELEMENTS:
        raise ValueError(
            f"tensor {key} is too large for layer streaming "
            f"({elements} elements > {_MAX_TENSOR_ELEMENTS})"
        )
    target = getattr(torch, dtype)
    return handle.get_tensor(key).to(target).contiguous()
