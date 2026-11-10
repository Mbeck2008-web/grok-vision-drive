"""Opt-in CUDA graph capture for a fixed-shape fused train forward (W1).

Off by default. When the flag is off the trainer never constructs a runner.
On CPU, or when the window shape changes / capture fails, the runner uses the
eager ``forward_window`` and leaves training numerics unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable


def heads_allow_graph(heads: Any) -> bool:
    """Capture is only safe when the trained loss uses exactly ``lane_gt``.

    Default / empty ``--heads`` trains every default head. A successful graph
    that rebuilt only ``lane_gt`` would zero or crash those other terms, so
    those runs stay on eager.
    """
    if heads is None:
        return False
    try:
        names = {str(name) for name in heads}
    except TypeError:
        return False
    return names == {"lane_gt"}


def graph_eligible(*, enabled: bool, device: Any, shape: Hashable | None) -> bool:
    """True only when the flag is on, the device is CUDA, and a shape is known."""
    if not enabled:
        return False
    if shape is None:
        return False
    text = str(device).lower()
    if not text.startswith("cuda"):
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


class CudaGraphRunner:
    """Capture a static CUDA graph once per shape; fall back to eager otherwise.

    ``run`` is the mechanism under test: allocate static buffers, capture, replay,
    or fall back. ``forward_window`` is what the trainer calls: eligible CUDA
    shapes try ``torch.cuda.make_graphed_callables`` on a tensor-only wrapper and
    fall back to eager on any failure or when ages/cond/aux need the full pred.
    """

    def __init__(self, *, enabled: bool, device: Any, heads: Any = None) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.heads = heads
        self.last_shape: Hashable | None = None
        self.mode = "idle"
        self.fallback_count = 0
        self.recapture_count = 0
        self.replay_count = 0
        self._heads_note = False
        self._graph: Any = None
        self._static_in: Any = None
        self._static_out: Any = None
        self._captured_shape: Hashable | None = None
        self._graphed: Any = None
        self._graphed_shape: Hashable | None = None

    def run(
        self,
        *,
        shape: Hashable,
        allocate: Callable[[], tuple[Any, Any]],
        fill: Callable[[Any, Any], None],
        eager: Callable[[Any, Any], None],
        capture_body: Callable[[Any, Any], None] | None = None,
    ) -> Any:
        """Run one step. Returns the static output buffer (or eager's out)."""
        self.last_shape = shape
        if not graph_eligible(enabled=self.enabled, device=self.device, shape=shape):
            static_in, static_out = allocate()
            fill(static_in, static_out)
            eager(static_in, static_out)
            self.mode = "eager_fallback"
            self.fallback_count += 1
            self._drop_graph()
            return static_out

        import torch

        body = capture_body or eager
        if self._captured_shape != shape or self._graph is None:
            if self._graph is not None:
                self.recapture_count += 1
            self._drop_graph()
            self._static_in, self._static_out = allocate()
            fill(self._static_in, self._static_out)
            torch.cuda.synchronize()
            body(self._static_in, self._static_out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    body(self._static_in, self._static_out)
            except RuntimeError:
                self.mode = "eager_fallback"
                self.fallback_count += 1
                self._drop_graph()
                return self._static_out
            self._graph = graph
            self._captured_shape = shape
            self.mode = "captured"
            return self._static_out

        fill(self._static_in, self._static_out)
        self._graph.replay()
        self.mode = "replay"
        self.replay_count += 1
        return self._static_out

    def forward_window(
        self,
        net: Any,
        strips: Any,
        dts: Any,
        state: Any,
        ages: Any = None,
        cond: Any = None,
        route: Any = None,
    ) -> Any:
        """Trainer hook: graphed fused forward when eligible, else eager."""
        shape = tuple(strips.shape) if hasattr(strips, "shape") else None
        self.last_shape = shape
        if not heads_allow_graph(self.heads):
            if not self._heads_note:
                got = "default/all" if self.heads is None else ",".join(str(h) for h in self.heads) or "empty"
                print(
                    "[GVD] exp-cuda-graph: refuse graph unless --heads is exactly "
                    f"lane_gt (got {got}); using eager fallback",
                    flush=True,
                )
                self._heads_note = True
            self.mode = "eager_fallback"
            self.fallback_count += 1
            return net.forward_window(strips, dts, state, ages, cond, route)
        needs_side = (
            ages is not None
            or cond is not None
            or route is not None
            or bool(getattr(net, "cam_age", False))
            or int(getattr(net, "cond_dim", 0) or 0)
            or bool(getattr(net, "aux_sensors", False))
            or bool(getattr(net, "route_cond", False))
        )
        if needs_side or not graph_eligible(enabled=self.enabled, device=self.device, shape=shape):
            self.mode = "eager_fallback"
            self.fallback_count += 1
            return net.forward_window(strips, dts, state, ages, cond, route)

        import torch

        if self._graphed_shape != shape or self._graphed is None:
            if self._graphed is not None:
                self.recapture_count += 1
            self._graphed = None
            self._graphed_shape = None

            class _Window(torch.nn.Module):
                def __init__(self, scene: Any) -> None:
                    super().__init__()
                    self.scene = scene

                def forward(self, window: Any, step_dts: Any, hidden: Any) -> Any:
                    from python.train.scene_net import RESIDUAL_S, SceneState

                    batch = int(window.shape[0])
                    tokens = torch.zeros(
                        batch,
                        int(state.tokens.shape[1]),
                        int(state.tokens.shape[2]),
                        device=window.device,
                        dtype=window.dtype,
                    )
                    mask = torch.full(
                        (batch, int(state.mask.shape[1])),
                        float(RESIDUAL_S) + 1.0,
                        device=window.device,
                        dtype=window.dtype,
                    )
                    seed = SceneState(h=hidden, tokens=tokens, mask=mask)
                    pred, end, _extra = self.scene.forward_window(window, step_dts, seed)
                    # Tuple of tensors so make_graphed_callables can bind them.
                    lane = pred.lane_gt
                    if lane is None:
                        lane = end.h
                    return lane, end.h, end.tokens, end.mask

            module = _Window(net).to(device=strips.device)
            sample_strips = strips.detach()
            sample_dts = (
                dts.detach()
                if torch.is_tensor(dts)
                else torch.as_tensor(dts, device=strips.device, dtype=strips.dtype)
            )
            sample_h = state.h.detach()
            try:
                self._graphed = torch.cuda.make_graphed_callables(
                    module, (sample_strips, sample_dts, sample_h)
                )
                self._graphed_shape = shape
                self.mode = "captured"
            except (RuntimeError, TypeError, ValueError):
                self.mode = "eager_fallback"
                self.fallback_count += 1
                return net.forward_window(strips, dts, state, ages, cond, route)

        dts_t = dts if torch.is_tensor(dts) else torch.as_tensor(dts, device=strips.device, dtype=strips.dtype)
        try:
            lane, h_new, tokens_new, mask_new = self._graphed(strips, dts_t, state.h)
            self.mode = "replay"
            self.replay_count += 1
        except RuntimeError:
            self.mode = "eager_fallback"
            self.fallback_count += 1
            self._graphed = None
            self._graphed_shape = None
            return net.forward_window(strips, dts, state, ages, cond, route)

        # Rebuild a full prediction with eager so every head stays populated the
        # same way; the graphed call still warms/replays the heavy stem+GRU path
        # under make_graphed_callables (autograd-safe). Then one eager forward
        # would double cost — so prefer returning eager only when graph failed.
        # When graph succeeded, build ScenePrediction from graphed tensors.
        from python.train.scene_net import ScenePrediction, SceneState

        pred = ScenePrediction()
        pred.lane_gt = lane
        end = SceneState(h=h_new, tokens=tokens_new, mask=mask_new)
        return pred, end, end

    def _drop_graph(self) -> None:
        self._graph = None
        self._static_in = None
        self._static_out = None
        self._captured_shape = None
