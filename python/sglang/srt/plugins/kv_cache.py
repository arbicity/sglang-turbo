"""Plugin-facing API for registering out-of-tree KV-cache backends.

A plugin KV-cache binds together the string name accepted via
``--kv-cache-dtype``, the torch storage dtype of the underlying buffer, and a
factory ``(runner) -> token_to_kv_pool`` that builds the pool.

Usage from a downstream plugin::

    from sglang.srt.plugins.kv_cache import register
    from sglang.srt.server_args import add_kv_cache_dtype_choices

    add_kv_cache_dtype_choices(["my_kv"])
    register("my_kv", torch_dtype=torch.uint8, pool_factory=_build_my_pool)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _Entry:
    torch_dtype: torch.dtype
    pool_factory: Callable[[Any], Any]
    paired_attention_backend: str | None
    cell_size_factory: Callable[[Any, int], int] | None = None


_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    *,
    torch_dtype: torch.dtype,
    pool_factory: Callable[[Any], Any],
    paired_attention_backend: str | None = None,
    cell_size_factory: Callable[[Any, int], int] | None = None,
) -> None:
    """Register a plugin KV-cache dtype.

    Args:
        name: The string accepted via ``--kv-cache-dtype``. Must also be added
            to the argparse choices via
            :func:`sglang.srt.server_args.add_kv_cache_dtype_choices`.
        torch_dtype: The torch dtype the pool buffer is allocated as.
        pool_factory: ``(runner) -> token_to_kv_pool`` callable. Receives the
            partially-initialized :class:`ModelRunner` and returns the pool
            assigned to ``runner.token_to_kv_pool``.
        paired_attention_backend: Optional name of the
            :mod:`sglang.srt.plugins.attention` backend this dtype is paired
            with. Read back through :func:`get_paired_attention_backend` /
            :func:`find_dtype_paired_with_backend` by a plugin's resolution
            hook (``sglang.srt.arg_groups.resolution_hooks``) that defaults
            either flag from the other.
        cell_size_factory: optional ``(runner, num_layers) -> int`` returning
            the per-token KV-cache cost in bytes summed across ``num_layers``
            effective attention layers (Mamba/recurrent layers excluded).
            Supersedes the built-in estimate derived from ``torch_dtype``.
            On a hybrid sliding-window model the factory is also called as
            ``(runner, num_layers, layer_ids=[...])`` once for the
            full-attention layers and once for the sliding-window layers, so
            each sub-pool is priced from the layers it holds.

    Idempotent: re-registering the same ``name`` overrides the previous entry.
    """
    _REGISTRY[name] = _Entry(
        torch_dtype=torch_dtype,
        pool_factory=pool_factory,
        paired_attention_backend=paired_attention_backend,
        cell_size_factory=cell_size_factory,
    )


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is registered as a plugin KV-cache dtype."""
    return name in _REGISTRY


def registered_names() -> list[str]:
    """Return the currently-registered plugin KV-cache dtype names."""
    return list(_REGISTRY.keys())


def get_torch_dtype(name: str) -> torch.dtype:
    """Return the torch storage dtype registered for ``name``."""
    return _REGISTRY[name].torch_dtype


def build_pool(name: str, runner: Any) -> Any:
    """Invoke the registered ``pool_factory(runner)`` for ``name``."""
    return _REGISTRY[name].pool_factory(runner)


def get_paired_attention_backend(name: str) -> str | None:
    """Return the paired attention-backend name for ``name``, or ``None``."""
    return _REGISTRY[name].paired_attention_backend


def find_dtype_paired_with_backend(backend_name: str) -> str | None:
    """Return the first kv-cache dtype paired with ``backend_name``, or ``None``."""
    for dtype_name, entry in _REGISTRY.items():
        if entry.paired_attention_backend == backend_name:
            return dtype_name
    return None


def has_cell_size(name: str) -> bool:
    """Return ``True`` if ``name`` registered a ``cell_size_factory``."""
    entry = _REGISTRY.get(name)
    return entry is not None and entry.cell_size_factory is not None


def get_cell_size(
    name: str,
    runner: Any,
    num_layers: int,
    layer_ids: list[int] | None = None,
) -> int:
    """Return the plugin's per-token KV cost in bytes across ``num_layers``.

    ``layer_ids`` names the layers priced, for a sub-pool of a hybrid
    sliding-window model; ``None`` prices the plugin's own layer set.
    """
    factory = _REGISTRY[name].cell_size_factory
    if factory is None:
        raise TypeError(f"KV-cache plugin {name!r} registered no cell_size_factory")
    if layer_ids is None:
        return int(factory(runner, num_layers))
    return int(factory(runner, num_layers, layer_ids=list(layer_ids)))


__all__ = [
    "register",
    "is_registered",
    "registered_names",
    "get_torch_dtype",
    "build_pool",
    "get_paired_attention_backend",
    "find_dtype_paired_with_backend",
    "has_cell_size",
    "get_cell_size",
]
