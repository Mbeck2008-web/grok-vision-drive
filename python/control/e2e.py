"""PilotNet-scale tiny E2E stub for GVD M5.

Inputs: named main/wide RGB 1×3×180×320 + kin 1×2 → {steer [-1,1], accel [-1,1]}.
Loads models/e2e_current.onnx when present (onnxruntime); else numpy random stub.
Toy VRAM footprint ~0.15–0.4 GB — no transformers / ViT / BEV / AutoSteer-HD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

E2E_W = 320
E2E_H = 180
# Stub MLP sees pooled 20x12 crops (tiny; ONNX path uses full 320x180).
STUB_W = 20
STUB_H = 12
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "e2e_current.onnx"


@dataclass
class E2EIntent:
    steer: float = 0.0  # [-1, 1]
    accel: float = 0.0  # [-1, 1]  (neg → brake)
    throttle: float = 0.0
    brake: float = 0.0
    ok: bool = True
    backend: str = "stub"
    reason: str = "ok"


def _resize_bgr(img: np.ndarray | None, w: int = E2E_W, h: int = E2E_H) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.float32)
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return out
    try:
        import cv2

        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out[:] = resized.astype(np.float32) / 255.0
    except Exception:
        src = img
        if src.dtype != np.float32:
            src = src.astype(np.float32) / (255.0 if src.dtype == np.uint8 else 1.0)
        sh, sw = src.shape[:2]
        ys = (np.linspace(0, max(sh - 1, 0), h)).astype(np.int32)
        xs = (np.linspace(0, max(sw - 1, 0), w)).astype(np.int32)
        out[:] = src[ys][:, xs, :3]
    return out


def _accel_to_pedals(accel: float) -> tuple[float, float]:
    a = float(max(-1.0, min(1.0, accel)))
    if a >= 0.0:
        return a, 0.0
    return 0.0, -a


def _pool_stub(cams_nchw: np.ndarray) -> np.ndarray:
    """Average-pool each cam to STUB_H x STUB_W then flatten (2*3*H*W)."""
    # cams: (2, 3, H, W)
    pooled = []
    for c in range(cams_nchw.shape[0]):
        ch = cams_nchw[c]  # 3,H,W
        small = _resize_bgr(ch.transpose(1, 2, 0), w=STUB_W, h=STUB_H)  # H,W,3
        pooled.append(small.transpose(2, 0, 1).reshape(-1))
    return np.concatenate(pooled, axis=0).astype(np.float32)


class E2EPolicy:
    """Tiny end-to-end policy: CNN/MLP toy or ONNX when weights exist."""

    def __init__(self, model_path: Path | None = None, seed: int = 7) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self.backend = "stub"
        self._session: Any = None
        self._rng = np.random.default_rng(seed)
        # Tiny PilotNet-ish MLP on pooled features (~few KB).
        in_dim = 2 * 3 * STUB_H * STUB_W + 2
        hidden = 64
        self._w1 = (self._rng.standard_normal((in_dim, hidden)) * 0.05).astype(np.float32)
        self._b1 = np.zeros((hidden,), dtype=np.float32)
        self._w2 = (self._rng.standard_normal((hidden, 2)) * 0.05).astype(np.float32)
        self._b2 = np.zeros((2,), dtype=np.float32)
        self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        if not self.model_path.is_file():
            self.backend = "stub"
            return
        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            # Prefer CUDA when available; never default TensorRT.
            want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            avail = set(ort.get_available_providers())
            providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=providers,
            )
            self.backend = "onnx"
        except Exception:
            self._session = None
            self.backend = "stub"

    @property
    def name(self) -> str:
        return f"e2e-{self.backend}"

    def preprocess(
        self,
        main_bgr: np.ndarray | None,
        wide_bgr: np.ndarray | None,
        *,
        speed_mps: float = 0.0,
        steer_deg: float = 0.0,
    ) -> dict[str, np.ndarray]:
        # Research pin: named feeds main/wide 1x3x180x320 + kin 1x2 (not concatenated).
        main = _resize_bgr(main_bgr)
        wide = _resize_bgr(wide_bgr if wide_bgr is not None else main_bgr)
        main_nchw = main.transpose(2, 0, 1)[None, ...].astype(np.float32)  # 1,3,H,W
        wide_nchw = wide.transpose(2, 0, 1)[None, ...].astype(np.float32)
        kin = np.array(
            [[float(speed_mps) / 30.0, float(steer_deg) / 30.0]],
            dtype=np.float32,
        )  # 1,2
        # Stub MLP still wants stacked cams + flat ego.
        cams = np.concatenate([main_nchw, wide_nchw], axis=0)  # 2,3,H,W
        ego = kin.reshape(2)
        return {
            "main": main_nchw,
            "wide": wide_nchw,
            "kin": kin,
            "cams": cams,
            "ego": ego,
        }

    def forward(
        self,
        main_bgr: np.ndarray | None,
        wide_bgr: np.ndarray | None = None,
        *,
        speed_mps: float = 0.0,
        steer_deg: float = 0.0,
    ) -> E2EIntent:
        feats = self.preprocess(main_bgr, wide_bgr, speed_mps=speed_mps, steer_deg=steer_deg)
        try:
            if self._session is not None:
                return self._forward_onnx(feats)
            return self._forward_numpy(feats)
        except Exception as e:
            return E2EIntent(ok=False, backend=self.backend, reason=f"forward_err:{type(e).__name__}")

    def _forward_onnx(self, feats: dict[str, np.ndarray]) -> E2EIntent:
        assert self._session is not None
        inputs = self._session.get_inputs()
        feed: dict[str, np.ndarray] = {}
        for inp in inputs:
            name = inp.name.lower()
            if name == "main" or name.endswith("/main") or name.startswith("main"):
                feed[inp.name] = feats["main"]
            elif name == "wide" or name.endswith("/wide") or name.startswith("wide"):
                feed[inp.name] = feats["wide"]
            elif (
                name == "kin"
                or name.endswith("/kin")
                or name.startswith("kin")
                or "ego" in name
                or "state" in name
                or "speed" in name
            ):
                feed[inp.name] = feats["kin"]
            else:
                # Unknown input name: prefer shape match (1x3xHxW vs 1x2).
                shape = tuple(int(d) if isinstance(d, int) else -1 for d in (inp.shape or ()))
                if len(shape) == 2 or (len(shape) >= 1 and shape[-1] == 2):
                    feed[inp.name] = feats["kin"]
                elif "wide" in name:
                    feed[inp.name] = feats["wide"]
                else:
                    feed[inp.name] = feats["main"]
        outs = self._session.run(None, feed)
        vec = np.asarray(outs[0]).reshape(-1)
        steer = float(np.clip(vec[0], -1.0, 1.0)) if vec.size > 0 else 0.0
        accel = float(np.clip(vec[1], -1.0, 1.0)) if vec.size > 1 else 0.0
        thr, brk = _accel_to_pedals(accel)
        return E2EIntent(
            steer=steer,
            accel=accel,
            throttle=thr,
            brake=brk,
            ok=True,
            backend="onnx",
            reason="ok",
        )

    def _forward_numpy(self, feats: dict[str, np.ndarray]) -> E2EIntent:
        pooled = _pool_stub(feats["cams"])
        x = np.concatenate([pooled, feats["ego"]], axis=0).astype(np.float32)
        if x.shape[0] != self._w1.shape[0]:
            buf = np.zeros((self._w1.shape[0],), dtype=np.float32)
            n = min(buf.shape[0], x.shape[0])
            buf[:n] = x[:n]
            x = buf
        h = np.tanh(x @ self._w1 + self._b1)
        y = np.tanh(h @ self._w2 + self._b2)
        steer = float(np.clip(y[0], -1.0, 1.0))
        accel = float(np.clip(y[1], -1.0, 1.0))
        thr, brk = _accel_to_pedals(accel)
        return E2EIntent(
            steer=steer,
            accel=accel,
            throttle=thr,
            brake=brk,
            ok=True,
            backend="stub",
            reason="stub",
        )


def make_e2e(model_path: Path | None = None) -> E2EPolicy:
    return E2EPolicy(model_path=model_path)
