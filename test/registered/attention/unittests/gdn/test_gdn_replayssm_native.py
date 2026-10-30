"""Native recomputation preserves published arithmetic and bounds scratch by requests."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_native import (
    capture_native_verify,
    commit_native_ssm,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

H, HV, K, V, WINDOW, LAYERS = 8, 24, 128, 128, 6, 2
GRAPH_COMMITS = 256
STATE_DTYPE = torch.float32


def make_pool():
    conv_dim = 2 * H * K + HV * V
    shape = Mamba2StateShape(
        conv=[(conv_dim, 3)],
        temporal=(HV, V, K),
        intermediate_size=HV * V,
        conv_dim=conv_dim,
        ssm_state_size=K,
        num_heads=HV,
        head_dim=V,
        state_size=K,
        conv_kernel=4,
        num_k_heads_per_tp=H,
    )
    params = Mamba2CacheParams(
        shape=shape,
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=STATE_DTYPE),
        layers=list(range(LAYERS)),
    )
    return MambaPool(
        size=16,
        spec_state_size=8,
        cache_params=params,
        mamba_layer_ids=list(range(LAYERS)),
        device="cuda",
        speculative_num_draft_tokens=WINDOW,
        enable_linear_replayssm_spec=True,
    ), params


def inputs(batch, steps, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        batch * steps,
        2 * H * K + HV * V,
        device="cuda",
        dtype=torch.bfloat16,
        generator=gen,
    )
    q, k, v = qkv.split([H * K, H * K, HV * V], -1)
    gates = torch.randn(
        batch * steps, 2 * HV, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    return dict(
        q=q.view(1, batch * steps, H, K),
        k=k.view(1, batch * steps, H, K),
        v=v.view(1, batch * steps, HV, V),
        a=gates[:, :HV],
        b=gates[:, HV:],
    )


class TestGdnNativeReplay(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pool, self.params = make_pool()
        self.pool.mamba_cache.temporal.normal_(0, 0.01)
        self.kernel = TritonGDNKernel()
        self.gates = [
            dict(
                A_log=torch.randn(HV, device="cuda") * 0.1,
                dt_bias=torch.randn(HV, device="cuda") * 0.1,
            )
            for _ in range(LAYERS)
        ]

    def run_verify(
        self, layer, values, state, physical, snapshots, snapshot_indices, steps
    ):
        return self.kernel.target_verify(
            **self.gates[layer],
            **values,
            ssm_states=state,
            cache_indices=physical,
            query_start_loc=torch.arange(
                0,
                (physical.numel() + 1) * steps,
                steps,
                device="cuda",
                dtype=torch.int32,
            ),
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=snapshot_indices,
            cache_steps=steps,
            retrieve_parent_token=None,
        )

    def capture(self, layer, values, physical, requests, steps):
        return capture_native_verify(
            self.pool,
            layer,
            requests,
            self.kernel.target_verify,
            suppress_snapshots=True,
            **self.gates[layer],
            **values,
            ssm_states=self.pool.mamba_cache.temporal[layer],
            cache_indices=physical,
            query_start_loc=torch.arange(
                0,
                (physical.numel() + 1) * steps,
                steps,
                device="cuda",
                dtype=torch.int32,
            ),
            intermediate_states_buffer=self.pool.replayssm_native_scratch,
            intermediate_state_indices=requests,
            cache_steps=steps,
            retrieve_parent_token=None,
        )

    def test_allocator_matches_accounting(self):
        cache = self.pool.mamba_cache
        records = [
            getattr(cache, f"replayssm_native_{name}")
            for name in ("q", "k", "v", "a", "b")
        ]
        self.assertIsNone(cache.intermediate_ssm)
        self.assertIsNone(cache.replayssm_d)
        self.assertEqual(
            sum(x.numel() * x.element_size() for x in records),
            self.params.replayssm_ring_bytes_per_req(WINDOW) * 10,
        )
        scratch = self.pool.replayssm_native_scratch
        self.assertEqual(
            scratch.numel() * scratch.element_size(),
            self.params.replayssm_shared_snapshot_bytes_per_req(WINDOW) * 9,
        )

    def test_request_padding_tracking_and_shared_layers(self):
        physical = torch.tensor([10, 12, -1, -1], device="cuda", dtype=torch.int32)
        requests = torch.tensor([0, 1, 0, 0], device="cuda", dtype=torch.int32)
        accepted = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
        track = torch.tensor([10, 5], device="cuda", dtype=torch.int32)
        track_steps = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        expected = self.pool.mamba_cache.temporal.clone()
        for layer in range(LAYERS):
            values = inputs(4, 4, 10 + layer)
            snapshots = torch.empty(
                17, WINDOW, HV, V, K, device="cuda", dtype=STATE_DTYPE
            )
            expected_out = self.run_verify(
                layer, values, expected[layer], physical, snapshots, physical, 4
            )
            actual_out = self.capture(layer, values, physical, requests, 4)
            self.assertTrue(torch.equal(expected_out, actual_out))
            expected[layer, 10] = snapshots[10, 1]
            expected[layer, 12] = snapshots[12, 2]
            expected[layer, 5] = snapshots[12, 1]
        commit_native_ssm(
            self.pool,
            physical[:2],
            requests[:2],
            accepted,
            track_indices=track,
            track_steps=track_steps,
            replay_shape=(4, 4),
        )
        self.assertTrue(torch.equal(expected, self.pool.mamba_cache.temporal))

    def test_cuda_graph_replay_is_bitwise(self):
        batch = 8
        physical = torch.arange(2, batch + 2, device="cuda", dtype=torch.int32)
        requests = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
        accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
        initial = self.pool.mamba_cache.temporal.clone()
        expected_state = initial.clone()
        values = [inputs(batch, WINDOW, 100 + layer) for layer in range(LAYERS)]
        snapshots = torch.empty(
            LAYERS, 17, WINDOW, HV, V, K, device="cuda", dtype=STATE_DTYPE
        )

        def baseline_step():
            outputs = []
            for layer in range(LAYERS):
                outputs.append(
                    self.run_verify(
                        layer,
                        values[layer],
                        expected_state[layer],
                        physical,
                        snapshots[layer],
                        physical,
                        WINDOW,
                    )
                )
                expected_state[layer, physical.long()] = snapshots[
                    layer, physical.long(), accepted.long() - 1
                ]
            return outputs

        def replay_step():
            outputs = [
                self.capture(layer, values[layer], physical, requests, WINDOW)
                for layer in range(LAYERS)
            ]
            commit_native_ssm(
                self.pool, physical, requests, accepted, replay_shape=(batch, WINDOW)
            )
            return outputs

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                baseline_step()
                replay_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        baseline_graph, replay_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            expected_outputs = baseline_step()
        with torch.cuda.graph(replay_graph):
            actual_outputs = replay_step()
        expected_state.copy_(initial)
        self.pool.mamba_cache.temporal.copy_(initial)
        gen = torch.Generator().manual_seed(7)
        for iteration in range(GRAPH_COMMITS):
            for layer in range(LAYERS):
                fresh = inputs(batch, WINDOW, 1000 + iteration * LAYERS + layer)
                for key in values[layer]:
                    values[layer][key].copy_(fresh[key])
            accepted.copy_(
                torch.randint(1, WINDOW + 1, (batch,), generator=gen, dtype=torch.int32)
            )
            baseline_graph.replay()
            replay_graph.replay()
            for expected, actual in zip(expected_outputs, actual_outputs):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected_state, self.pool.mamba_cache.temporal))


if __name__ == "__main__":
    unittest.main()
