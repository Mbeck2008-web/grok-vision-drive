# gvd-ci: requires torch
"""Default SceneNet stays byte-identical. A preset on a checkpoint has to match."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from python.train.epoch_ckpt import build_net, load_into_net, write_epoch_ckpt  # noqa: E402
from python.train.model_size import PresetMismatch, scene_net_kwargs  # noqa: E402
from python.train.scene_ckpt_runtime import SceneCkptRuntime  # noqa: E402
from python.train.scene_net import SceneNet, architecture_parameter_count  # noqa: E402


def _weights(net: SceneNet) -> dict:
    return {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}


def check_default_path_parity() -> None:
    torch.manual_seed(0)
    left = SceneNet(cams=("main",))
    torch.manual_seed(0)
    right = SceneNet(cams=("main",), **scene_net_kwargs("small"))
    assert scene_net_kwargs("small") == {}
    assert left.preset == "small"
    assert left.gru_hidden == 512
    assert not hasattr(left, "pause_arch")
    left_sd = left.state_dict()
    right_sd = right.state_dict()
    assert list(left_sd) == list(right_sd)
    for key in left_sd:
        assert torch.equal(left_sd[key], right_sd[key]), key
    full = SceneNet()
    assert sum(p.numel() for p in full.parameters()) == architecture_parameter_count()



def check_ckpt_meta_matrix() -> None:
    """Old / preset-only / route-cond-only / both must all rebuild correctly."""
    old = build_net({"cams": ["main"], "scale": 1.0})
    assert old.preset == "small"
    assert old.gru_hidden == 512
    assert not bool(getattr(old, "route_cond", False))
    assert not hasattr(old, "route_proj")
    assert ":preset=" not in str(getattr(old, "pause_arch", ""))
    assert ":route_cond=" not in str(getattr(old, "pause_arch", ""))

    preset_only = build_net({"preset": "tiny", "cams": ["main"], "scale": 1.0, "backbone": "scratch"})
    assert preset_only.preset == "tiny"
    assert preset_only.gru_hidden == 256
    assert not bool(preset_only.route_cond)
    assert str(preset_only.pause_arch).endswith(":preset=tiny")
    assert ":route_cond=" not in str(preset_only.pause_arch)

    route_only = build_net({"cams": ["main"], "scale": 1.0, "route_cond": True})
    assert route_only.preset == "small"
    assert route_only.gru_hidden == 512
    assert bool(route_only.route_cond)
    assert hasattr(route_only, "route_proj")
    assert ":route_cond=" in str(route_only.pause_arch)
    assert ":preset=" not in str(route_only.pause_arch)

    both = build_net(
        {"preset": "tiny", "cams": ["main"], "scale": 1.0, "backbone": "scratch", "route_cond": True}
    )
    assert both.preset == "tiny"
    assert both.gru_hidden == 256
    assert bool(both.route_cond)
    assert hasattr(both, "route_proj")
    assert ":route_cond=" in str(both.pause_arch)
    assert str(both.pause_arch).endswith(":preset=tiny")


def check_preset_roundtrip_and_mismatch(tmp: Path) -> None:
    torch.manual_seed(1)
    tiny = SceneNet(cams=("main",), **scene_net_kwargs("tiny"))
    assert tiny.preset == "tiny"
    assert tiny.gru_hidden == 256
    assert str(tiny.pause_arch).endswith(":preset=tiny")
    meta = {"preset": "tiny", "cams": ["main"], "scale": 1.0, "backbone": "scratch"}
    rebuilt = build_net(meta)
    assert rebuilt.preset == "tiny"
    assert rebuilt.gru_hidden == tiny.gru_hidden
    load_into_net(rebuilt, _weights(tiny))
    old = build_net({"cams": ["main"], "scale": 1.0})
    assert old.preset == "small"
    assert old.gru_hidden == 512
    path = tmp / "tiny.npz"
    write_epoch_ckpt(path, _weights(tiny), meta, min_free_bytes=0, disk_free=lambda _p: 10**15)
    loaded = SceneCkptRuntime(path, requested_preset="tiny")
    assert loaded.net.preset == "tiny"
    try:
        SceneCkptRuntime(path, requested_preset="medium")
    except PresetMismatch as exc:
        assert "tiny" in str(exc) and "medium" in str(exc)
    else:
        raise AssertionError("a medium load of a tiny checkpoint must fail")
    legacy = tmp / "legacy.npz"
    write_epoch_ckpt(
        legacy,
        _weights(old),
        {"cams": ["main"], "scale": 1.0},
        min_free_bytes=0,
        disk_free=lambda _p: 10**15,
    )
    kept = SceneCkptRuntime(legacy, requested_preset="small")
    assert kept.net.preset == "small"


def main() -> None:
    import tempfile

    check_default_path_parity()
    check_ckpt_meta_matrix()
    with tempfile.TemporaryDirectory() as tmp:
        check_preset_roundtrip_and_mismatch(Path(tmp))
    print("test_gpu_model_size_net: ok")


if __name__ == "__main__":
    main()
