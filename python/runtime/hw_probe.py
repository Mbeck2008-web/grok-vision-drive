"""Honest hardware probe for GVD boot line. No Tesla / HW3 claims."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HwReport:
    cpu: str
    ram_gb: float
    dgpu: str
    dgpu_vram_gb: float | None
    igpu: str
    qsv: bool
    backend: str
    profile: str = "gtx_1080_ti"
    notes: list[str] | None = None

    @property
    def is_retail(self) -> bool:
        return (self.backend or "").lower() == "window"

    def boot_line(self) -> str:
        vram = f"{self.dgpu_vram_gb:.0f}GB" if self.dgpu_vram_gb else "?"
        qsv = "yes" if self.qsv else "no"
        return (
            f"[GVD] cpu={self.cpu} ram={self.ram_gb:.0f}GB "
            f"dgpu={self.dgpu} {vram} igpu={self.igpu} qsv={qsv} backend={self.backend} "
            f"{cam_claim(self.backend)}"
        )


def cam_claim(backend: str) -> str:
    """Honest camera claim per backend — the retail window path never says 8 cams."""
    b = (backend or "stub").lower()
    if b == "window":
        return "cams=1/8 path=retail (1 window capture; not 8)"
    if b == "beamngpy":
        return "cams<=8 path=tech (BeamNGpy cameras.yaml; live UNPROVEN)"
    return "cams=0/8 path=stub"


def _cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _ram_gb() -> float:
    if platform.system() == "Linux":
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return kb / (1024 * 1024)
        except Exception:
            pass
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 0.0


def _nvidia() -> tuple[str, float | None]:
    if not shutil.which("nvidia-smi"):
        return "none", None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        name = parts[0]
        vram = float(parts[1]) / 1024.0 if len(parts) > 1 else None  # MiB → GiB approx if already MiB
        # nvidia-smi memory.total is MiB
        if vram and vram > 64:  # clearly MiB
            vram = float(parts[1]) / 1024.0
        return name, vram
    except Exception:
        return "nvidia-unknown", None


def _qsv_available() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True, timeout=5, stderr=subprocess.STDOUT)
        return "h264_qsv" in out or "hevc_qsv" in out
    except Exception:
        return False


def _igpu_label(cpu: str) -> str:
    # 9900KF has no iGPU; 9900K has UHD 630
    if re.search(r"9900KF", cpu, re.I):
        return "none(9900KF)"
    if re.search(r"9900K", cpu, re.I):
        return "UHD630"
    if platform.system() == "Linux":
        return "unknown"
    return "UHD630?"


def load_profile(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    hw = path or (root / "config" / "hardware.yaml")
    data: dict[str, Any] = {}
    try:
        for line in hw.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.strip().startswith("#") and not line.startswith(" "):
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    except Exception:
        pass
    return data


def beamng_process_running() -> bool:
    """True if a BeamNG process looks running (best-effort)."""
    names = ("beamng", "beamng.drive", "beamngtech", "beamng.tech")
    try:
        import psutil  # type: ignore

        for p in psutil.process_iter(["name"]):
            n = (p.info.get("name") or "").lower()
            if any(x in n for x in names):
                return True
    except Exception:
        pass
    if platform.system() == "Linux":
        try:
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    cmd = (proc / "cmdline").read_bytes().decode("utf-8", "ignore").lower()
                except Exception:
                    continue
                if "beamng" in cmd:
                    return True
        except Exception:
            pass
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(["tasklist"], text=True, timeout=5, stderr=subprocess.DEVNULL)
            if "BeamNG" in out:
                return True
        except Exception:
            pass
    return False


def refuse_live_start(report: HwReport, *, vision_only: bool = False) -> str | None:
    """Return a refuse reason, or None if live start is allowed.

    Spec: refuse if VRAM < 10 GB and BeamNG is up, unless --vision-only.
    """
    if vision_only:
        return None
    vram = report.dgpu_vram_gb
    if vram is None:
        return None
    # Target profile placeholder "GTX 1080 Ti (target)" reports 11 — real probe uses nvidia-smi
    if vram < 10.0 and beamng_process_running():
        return (
            f"dGPU VRAM {vram:.1f} GB < 10 GB while BeamNG is up — "
            "refuse live start (pass --vision-only to override)"
        )
    return None


def probe(backend: str = "stub") -> HwReport:
    cpu = _cpu_name()
    # Prefer configured profile labels when probing a foreign CI box
    profile = load_profile()
    dgpu_name, vram = _nvidia()
    if dgpu_name == "none" and profile.get("profile") == "gtx_1080_ti":
        # Offline/CI: report target profile honestly as target, not detected
        dgpu_name = "GTX 1080 Ti (target)"
        vram = 11.0
    cpu_label = "i9-9900K" if re.search(r"9900K(?!F)", cpu, re.I) else (
        "i9-9900KF" if re.search(r"9900KF", cpu, re.I) else cpu.split("@")[0].strip()[:48]
    )
    ram = _ram_gb() or float(profile.get("system_ram_gb") or 0) or 32.0
    igpu = _igpu_label(cpu)
    qsv = _qsv_available()
    notes = []
    if vram is not None and vram < 10 and os.environ.get("GVD_VISION_ONLY", "") not in ("1", "true"):
        notes.append("VRAM<10GB with BeamNG up may OOM — use --vision-only to force")
    return HwReport(
        cpu=cpu_label,
        ram_gb=ram,
        dgpu=dgpu_name,
        dgpu_vram_gb=vram,
        igpu=igpu,
        qsv=qsv,
        backend=backend,
        profile=str(profile.get("profile") or "gtx_1080_ti"),
        notes=notes,
    )
