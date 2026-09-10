"""Optional Foxglove WebSocket publisher. Inference stays vision-only.

Install: pip install -r requirements-foxglove.txt
Enable: config/sensors.yaml foxglove.enabled=true, GVD_FOXGLOVE=1, or --foxglove.
Connect the Foxglove app to ws://127.0.0.1:8765. Missing SDK → honest skip.
"""

from __future__ import annotations

from typing import Any

from python.sensors.extras import SensorBundle


class FoxgloveBridge:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config if isinstance(config, dict) else {}
        fox = cfg.get("foxglove") if isinstance(cfg.get("foxglove"), dict) else {}
        self.enabled = bool(fox.get("enabled"))
        self.host = str(fox.get("host") or "127.0.0.1")
        self.port = int(fox.get("port") or 8765)
        self.want_cams = bool(fox.get("cameras", True))
        self.max_lidar = int(fox.get("max_lidar_points") or 4000)
        self.ok = False
        self.note = "disabled"
        self._fg: Any = None
        self._server: Any = None
        self._channels: dict[str, Any] = {}
        if not self.enabled:
            return
        try:
            import foxglove  # type: ignore
        except Exception as e:
            self.note = f"foxglove-sdk missing ({e})"
            print(f"[GVD] Foxglove off: {self.note}. pip install -r requirements-foxglove.txt", flush=True)
            return
        try:
            kwargs: dict[str, Any] = {"port": self.port}
            try:
                kwargs["host"] = self.host
                self._server = foxglove.start_server(**kwargs)
            except TypeError:
                kwargs.pop("host", None)
                self._server = foxglove.start_server(**kwargs)
            self._fg = foxglove
            self.ok = True
            self.note = f"ws://{self.host}:{self.port}"
            print(f"[GVD] Foxglove listening on {self.note} (viz only; planner stays vision-only).", flush=True)
        except Exception as e:
            self.note = f"start failed ({e})"
            print(f"[GVD] Foxglove off: {self.note}", flush=True)

    def _chan(self, topic: str) -> Any | None:
        if not self.ok or self._fg is None:
            return None
        if topic in self._channels:
            return self._channels[topic]
        ch = None
        try:
            Channel = getattr(self._fg, "Channel", None)
            if Channel is not None:
                ch = Channel(topic=topic, message_encoding="json")
        except Exception:
            ch = None
        self._channels[topic] = ch
        return ch

    def _log(self, topic: str, payload: dict[str, Any]) -> None:
        ch = self._chan(topic)
        if ch is None:
            return
        try:
            ch.log(payload)
        except Exception:
            try:
                self._fg.log(topic, payload)  # type: ignore[attr-defined]
            except Exception:
                pass

    def publish(self, bundle: SensorBundle, frames: dict[str, Any] | None = None) -> None:
        if not self.ok:
            return
        imu = bundle.imu
        self._log(
            "/gvd/imu",
            {
                "header": {"frame_id": "base_link"},
                "linear_acceleration": {"x": imu.gx, "y": imu.gy, "z": imu.gz},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": imu.yaw_rate},
                "source": imu.source,
            },
        )
        gps = bundle.gps
        self._log(
            "/gvd/gps",
            {
                "latitude": gps.lat,
                "longitude": gps.lon,
                "ok": gps.ok,
                "source": gps.source,
            },
        )
        pts = bundle.lidar.points[: self.max_lidar]
        self._log(
            "/gvd/lidar",
            {
                "frame_id": "base_link",
                "source": bundle.lidar.source,
                "n": len(bundle.lidar.points),
                "points": [{"x": p[0], "y": p[1], "z": p[2]} for p in pts[:200]],
            },
        )
        self._log(
            "/gvd/radar",
            {
                "source": bundle.radar.source,
                "n": len(bundle.radar.returns),
                "returns": bundle.radar.returns[:32],
            },
        )
        if self.want_cams and frames:
            main = frames.get("main") if isinstance(frames, dict) else None
            if main is None and isinstance(frames, dict):
                main = frames.get("cam_main")
            if main is not None:
                try:
                    h, w = int(main.shape[0]), int(main.shape[1])
                    self._log(
                        "/gvd/camera/main",
                        {
                            "encoding": "bgr8",
                            "width": w,
                            "height": h,
                            "note": "metadata only; full pixels stay in GVD VISION",
                        },
                    )
                except Exception:
                    pass

    def close(self) -> None:
        if self._server is not None:
            try:
                stop = getattr(self._server, "stop", None) or getattr(self._server, "close", None)
                if stop:
                    stop()
            except Exception:
                pass
        self._server = None
        self.ok = False
