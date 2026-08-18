"""Plugin-facing API for registering out-of-tree attention backends.

Usage from a downstream plugin::

    from sglang.srt.plugins.attention import register

    register("mybackend", lambda runner: MyAttnBackend(runner))

``--attention-backend mybackend`` then resolves to ``MyAttnBackend``. Extend
the argparse choices via
:func:`sglang.srt.server_args.add_attention_backend_choices`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    PLUGIN_ATTENTION_BACKENDS,
    register_attention_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


MULTI_STEP_ATTENTION_BACKENDS: dict[str, Callable[[Any, int, int], Any]] = {}


def register(
    name: str,
    factory: Callable[[Any], AttentionBackend],
    multi_step_factory: Callable[[Any, int, int], Any] | None = None,
) -> None:
    """Register an attention backend factory under ``name``.

    Args:
        name: The string accepted via ``--attention-backend``.
        factory: ``(runner) -> AttentionBackend`` callable.
        multi_step_factory: optional
            ``(draft_model_runner, topk, speculative_num_steps) ->
            multi-step draft backend`` callable enabling EAGLE/NEXTN
            speculative decoding with this backend. The returned wrapper
            must expose ``attn_backends`` plus the
            ``init_forward_metadata*`` / ``init_cuda_graph_state`` fan-out
            surface the draft CUDA graph runner drives. Omitting it keeps
            spec-decode gated off for this backend.

    Idempotent: re-registering the same ``name`` overrides the previous entry.
    """
    ATTENTION_BACKENDS[name] = factory
    PLUGIN_ATTENTION_BACKENDS.add(name)
    if multi_step_factory is not None:
        MULTI_STEP_ATTENTION_BACKENDS[name] = multi_step_factory


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is in the attention-backend registry."""
    return name in ATTENTION_BACKENDS


def get_multi_step_factory(name: str) -> Callable[[Any, int, int], Any] | None:
    """Return the registered multi-step draft-decode factory for ``name``."""
    return MULTI_STEP_ATTENTION_BACKENDS.get(name)


def registered_names() -> list[str]:
    """Return the currently-registered attention backend names."""
    return list(ATTENTION_BACKENDS.keys())


__all__ = [
    "register",
    "register_attention_backend",
    "is_registered",
    "get_multi_step_factory",
    "registered_names",
    "ATTENTION_BACKENDS",
    "MULTI_STEP_ATTENTION_BACKENDS",
]
