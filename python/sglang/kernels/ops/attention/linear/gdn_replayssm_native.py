# SPDX-License-Identifier: Apache-2.0
"""Recompute accepted GDN checkpoints with the unchanged native verifier.

Raw inputs live by request, while checkpoints live by physical state slot.
One snapshot workspace is reused across layers. No arithmetic kernel is
modified, and replay restores the original input dtypes and row strides.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_KEYS = ("q", "k", "v", "a", "b")
_FIELDS = tuple(f"replayssm_native_{key}" for key in _KEYS)
_DYNAMIC = frozenset(
    (
        *_KEYS,
        "ssm_states",
        "cache_indices",
        "query_start_loc",
        "intermediate_states_buffer",
        "intermediate_state_indices",
    )
)


@dataclass
class NativeVerifyContext:
    verify: Callable
    constants: dict
    tails: tuple
    dtypes: tuple
    row_strides: tuple
    state_index_dtype: torch.dtype
    snapshot_index_dtype: torch.dtype
    sequence_dtype: torch.dtype
    suppress_snapshots: bool

    def make_inputs(self, batch: int, steps: int, device):
        result = {}
        for key, tail, dtype, stride in zip(
            _KEYS, self.tails, self.dtypes, self.row_strides
        ):
            if key in ("q", "k", "v"):
                shape = (1, batch * steps, *tail)
                strides = (batch * steps * stride, stride, tail[-1], 1)
            else:
                shape, strides = (batch * steps, *tail), (stride, 1)
            result[key] = torch.empty_strided(
                shape, strides, dtype=dtype, device=device
            )
        return result


def capture_native_verify(
    pool,
    layer_index: int,
    request_indices,
    verify: Callable,
    *,
    suppress_snapshots: bool,
    **kwargs,
):
    """Save raw inputs and execute the original verifier without retained snapshots."""
    inputs = {key: kwargs[key] for key in _KEYS}
    physical = kwargs["cache_indices"]
    batch = kwargs["query_start_loc"].numel() - 1
    steps = inputs["q"].shape[1] // batch
    records = [getattr(pool.mamba_cache, field)[layer_index] for field in _FIELDS]
    if steps > records[0].shape[1]:
        raise ValueError(
            "GDN replay window exceeds the allocated adaptive draft maximum"
        )
    tails = tuple(
        value.shape[2:] if value.ndim == 4 else value.shape[1:]
        for value in inputs.values()
    )
    context = NativeVerifyContext(
        verify=verify,
        constants={k: v for k, v in kwargs.items() if k not in _DYNAMIC},
        tails=tails,
        dtypes=tuple(value.dtype for value in inputs.values()),
        row_strides=tuple(
            value.stride(1) if value.ndim == 4 else value.stride(0)
            for value in inputs.values()
        ),
        state_index_dtype=physical.dtype,
        snapshot_index_dtype=kwargs["intermediate_state_indices"].dtype,
        sequence_dtype=kwargs["query_start_loc"].dtype,
        suppress_snapshots=suppress_snapshots,
    )
    key = (layer_index, batch, steps)
    previous = pool.replayssm_native_contexts.get(key)
    if previous is not None and (
        previous.row_strides != context.row_strides or previous.dtypes != context.dtypes
    ):
        raise ValueError("Native GDN replay input layout changed for an existing shape")
    pool.replayssm_native_contexts[key] = context
    pool.replayssm_native_active_shape = (batch, steps)
    # A dedicated last row absorbs padded requests, without racing a live
    # request's records (even when a live request uses logical row zero).
    safe_requests = torch.where(
        physical[:batch] >= 0, request_indices[:batch], records[0].shape[0] - 1
    )
    for value, target, tail in zip(inputs.values(), records, tails):
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Native GDN replay supports activation precision up to FP32"
            )
        target[safe_requests.long(), :steps] = value.reshape(batch, steps, *tail).to(
            target.dtype
        )
    if suppress_snapshots:
        snapshot_indices = torch.full_like(
            kwargs["intermediate_state_indices"][:batch], -1
        )
    else:
        snapshot_indices = torch.where(
            physical[:batch] >= 0, request_indices[:batch], 0
        ).to(context.snapshot_index_dtype)
    return verify(
        **{
            **kwargs,
            "intermediate_states_buffer": pool.replayssm_native_scratch,
            "intermediate_state_indices": snapshot_indices,
        }
    )


@triton.jit
def _commit_snapshot(
    state,
    snapshots,
    physical,
    requests,
    accepted,
    track_indices,
    track_steps,
    STATE_STRIDE: tl.constexpr,
    SNAPSHOT_STRIDE: tl.constexpr,
    PHYSICAL_STRIDE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    TRACK_STRIDE: tl.constexpr,
    STEP_STRIDE: tl.constexpr,
    STATE_ELEMENTS: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    NULL_SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    slot = tl.load(physical + request * PHYSICAL_STRIDE).to(tl.int64)
    count = tl.load(accepted + request * ACCEPT_STRIDE)
    if slot <= NULL_SLOT or count <= 0:
        return
    record = tl.load(requests + request * REQUEST_STRIDE).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < STATE_ELEMENTS
    if HAS_TRACK:
        track = tl.load(track_indices + request * TRACK_STRIDE).to(tl.int64)
        step = tl.load(track_steps + request * STEP_STRIDE)
        if track > NULL_SLOT and step >= 0 and step < count:
            value = tl.load(
                snapshots + record * SNAPSHOT_STRIDE + step * STATE_ELEMENTS + offsets,
                mask=mask,
            )
            tl.store(state + track * STATE_STRIDE + offsets, value, mask=mask)
    # Accepted state wins when the tracking destination aliases this slot.
    value = tl.load(
        snapshots + record * SNAPSHOT_STRIDE + (count - 1) * STATE_ELEMENTS + offsets,
        mask=mask,
    )
    tl.store(state + slot * STATE_STRIDE + offsets, value, mask=mask)


def commit_native_ssm(
    pool,
    physical,
    requests,
    accepted,
    *,
    replay_shape=None,
    track_indices=None,
    track_steps=None,
    null_slot=-1,
):
    """Replay each layer using its original shape, then copy accepted/tracked states."""
    batch = accepted.numel()
    replay_batch, steps = replay_shape or pool.replayssm_native_active_shape
    if replay_batch < batch:
        raise ValueError("Native GDN replay shape is smaller than the accepted batch")
    pad = replay_batch - batch
    padded_physical = F.pad(physical[:batch], (0, pad), value=-1)
    padded_requests = F.pad(requests[:batch], (0, pad), value=0)
    shared = pool.replayssm_native_scratch
    cache = pool.mamba_cache
    has_track = track_indices is not None and track_steps is not None
    for layer_index in range(cache.temporal.shape[0]):
        context = pool.replayssm_native_contexts[(layer_index, replay_batch, steps)]
        records = [getattr(cache, field)[layer_index] for field in _FIELDS]
        safe_requests = torch.where(
            padded_physical >= 0, padded_requests, records[0].shape[0] - 1
        )
        inputs = context.make_inputs(replay_batch, steps, physical.device)
        for key, record in zip(_KEYS, records):
            inputs[key].copy_(
                record[safe_requests.long(), :steps].reshape(inputs[key].shape)
            )
        snapshot_indices = torch.where(
            padded_physical >= 0,
            padded_requests,
            -1 if context.suppress_snapshots else 0,
        ).to(context.snapshot_index_dtype)
        context.verify(
            **context.constants,
            **inputs,
            ssm_states=cache.temporal[layer_index],
            cache_indices=padded_physical.to(context.state_index_dtype),
            query_start_loc=torch.arange(
                0,
                (replay_batch + 1) * steps,
                steps,
                device=physical.device,
                dtype=context.sequence_dtype,
            ),
            intermediate_states_buffer=shared,
            intermediate_state_indices=snapshot_indices,
        )
        state = cache.temporal[layer_index]
        elements = state.shape[-3] * state.shape[-2] * state.shape[-1]
        _commit_snapshot[(batch, triton.cdiv(elements, 256))](
            state,
            shared,
            physical,
            requests,
            accepted,
            track_indices if has_track else physical,
            track_steps if has_track else accepted,
            STATE_STRIDE=state.stride(0),
            SNAPSHOT_STRIDE=shared.stride(0),
            PHYSICAL_STRIDE=physical.stride(0),
            REQUEST_STRIDE=requests.stride(0),
            ACCEPT_STRIDE=accepted.stride(0),
            TRACK_STRIDE=track_indices.stride(0) if has_track else 0,
            STEP_STRIDE=track_steps.stride(0) if has_track else 0,
            STATE_ELEMENTS=elements,
            HAS_TRACK=has_track,
            NULL_SLOT=null_slot,
            BLOCK=256,
        )


def commit_native_after_verify(
    *,
    pool,
    state_batch_indices,
    record_indices,
    accept_lens,
    last_correct_step_indices,
    replay_shape=None,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
    src_indices_raw=None,
    null_block_id=-1,
):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    commit_native_ssm(
        pool,
        state_batch_indices,
        record_indices,
        accept_lens,
        replay_shape=replay_shape,
        track_indices=mamba_track_indices,
        track_steps=mamba_steps_to_track,
        null_slot=null_block_id,
    )
    cache = pool.mamba_cache
    for conv, window in zip(cache.conv, cache.intermediate_conv_window):
        fused_conv_window_scatter_with_mask(
            conv,
            window,
            state_batch_indices,
            last_correct_step_indices,
            src_indices_raw=src_indices_raw,
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv,
                window,
                mamba_track_indices,
                mamba_steps_to_track,
                src_indices_raw=src_indices_raw,
            )
