"""Plugin-facing API for registering custom attention backends.

Thin re-export of the existing
:mod:`sglang.srt.layers.attention.attention_registry` so out-of-tree
plugins have a single ``sglang.srt.plugins.*`` namespace to import from
(symmetric with :mod:`sglang.srt.plugins.kv_cache`).

Usage from a downstream plugin::

    from sglang.srt.plugins.attention import register

    def _build_my_backend(runner):
        from my_pkg.backend import MyAttnBackend
        return MyAttnBackend(runner)

    register("mybackend", _build_my_backend)

Then ``--attention-backend mybackend`` resolves to ``MyAttnBackend``.
The argparse ``choices=`` list for ``--attention-backend`` can be
extended via :func:`sglang.srt.server_args.add_attention_backend_choices`.
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


# Speculative-decode (EAGLE/NEXTN) multi-step DRAFT DECODE factories.
# Keyed by the same backend name as ATTENTION_BACKENDS; the factory
# builds the wrapper holding one per-step backend instance, matching
# the built-in multi-step constructors' signature
# ``(draft_model_runner, topk, speculative_num_steps)``.
# Consulted by ``DraftBackendFactory.create_decode_backend`` when the
# resolved backend name is not in its hardcoded map — so a plugin
# attention backend can serve EAGLE/NEXTN draft decode without an
# upstream backend_map entry (symmetric with the draft-extend plugin
# consultation already in draft_utils).
MULTI_STEP_ATTENTION_BACKENDS: dict[str, Callable[[Any, int, int], Any]] = {}


def register(
    name: str,
    factory: Callable[[Any], "AttentionBackend"],
    multi_step_factory: Callable[[Any, int, int], Any] | None = None,
) -> None:
    """Register an attention backend factory under ``name``.

    Equivalent to using :func:`register_attention_backend` as a
    decorator. Idempotent: re-registering the same ``name`` overrides
    the previous factory.

    Args:
        name: The string accepted via ``--attention-backend``.
        factory: ``(runner) -> AttentionBackend`` callable.
        multi_step_factory: optional
            ``(draft_model_runner, topk, speculative_num_steps) ->
            multi-step draft backend`` callable enabling EAGLE/NEXTN
            speculative decoding with this backend (the wrapper must
            expose ``attn_backends`` plus the ``init_forward_metadata*``
            / ``init_cuda_graph_state`` fan-out surface the draft CUDA
            graph runner drives). Omitting it keeps spec-decode gated
            off for this backend.
    """
    ATTENTION_BACKENDS[name] = factory
    # Mark as plugin-provided so guards that special-case the built-in
    # backend set (hybrid-GDN Blackwell guard) accept it.
    PLUGIN_ATTENTION_BACKENDS.add(name)
    if multi_step_factory is not None:
        MULTI_STEP_ATTENTION_BACKENDS[name] = multi_step_factory


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is in the attention-backend registry."""
    return name in ATTENTION_BACKENDS


def get_multi_step_factory(name: str) -> Callable[[Any, int, int], Any] | None:
    """Return the registered multi-step draft-decode factory for ``name``
    (``None`` when the plugin did not provide one)."""
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
