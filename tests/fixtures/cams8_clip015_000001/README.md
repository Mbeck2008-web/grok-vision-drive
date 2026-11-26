# cams8_clip015_000001: one real 8-camera tick

One tick (frame index 1) of clip015, all eight cameras, copied byte-for-byte: original PNG encoding, native resolution, no resize, no re-encode.
It is the only index present in all eight camera zips (the repeaters and rear are recorded at a lower rate, so they skip most ticks); no other tick has all 8 cameras.
Used by `scripts/test_scene_ckpt_8cam.py` (`check_train_live_input_identical`).

## Source (read only, never modified)

- Recording: `/workspace/rv-preview/v2src/clip015/` (clip tags: heavy_traffic, west_coast_usa, `quality: pretrain_only`). clip015 is not a held-out clip (005 and 003 are).
- Each file is byte-identical to BOTH the loose copy `/workspace/rv-preview/v2src/clip015/raw/<cam>/000/000001.png` and the zip member in `/workspace/rv-preview/v2src/<cam>.zip` (verified by comparing the bytes).
- Only the folder layout differs: the source stores `raw/<cam>/000/000001.png` and has no `raw_layout.json`, which `python/data/raw_frames.py` needs for numbered subfolders. Here the files sit flat as `raw/<cam>/000001.png`, the flat layout the loader reads without a layout file. File contents are unchanged.

## Files

| cam | file here | zip member | bytes | sha256 |
| --- | --- | --- | ---: | --- |
| repeatL | `raw/repeatL/000001.png` | `repeatL.zip` : `repeatL/000/000001.png` | 199636 | `f694e64e39dad361e7a211d2b343de44d554f106f9424cf4b27fc106acef7bb5` |
| pillarL | `raw/pillarL/000001.png` | `pillarL.zip` : `pillarL/000/000001.png` | 216872 | `4905d68091ac01f9fbabbd3b1b2a2a2d841c482eff368ee1618a0478eca056ad` |
| wide | `raw/wide/000001.png` | `wide.zip` : `wide/000/000001.png` | 163329 | `6821ac4cca0a02a27dbef6c158d0ace7e9861a661317f9c0f577d457e3985e5c` |
| main | `raw/main/000001.png` | `main.zip` : `main/000/000001.png` | 177241 | `4e7fed544022ac151743712097ad7b82f1b625b4433e05237f0253308ed0adba` |
| narrow | `raw/narrow/000001.png` | `narrow.zip` : `narrow/000/000001.png` | 140001 | `aa827d65a11c89e2ff1b8f4d3ed7ea54df75d5238eaffb74ea72c3d2d763e20b` |
| pillarR | `raw/pillarR/000001.png` | `pillarR.zip` : `pillarR/000/000001.png` | 145796 | `881c862e4c62d4f2c9e585ba6ea03a291891257e3001a98de1982f78a3f473da` |
| repeatR | `raw/repeatR/000001.png` | `repeatR.zip` : `repeatR/000/000001.png` | 226640 | `d4b89560cb4f97a68e822803b757946a7a99850435f87211016250832a339bb6` |
| rear | `raw/rear/000001.png` | `rear.zip` : `rear/000/000001.png` | 188592 | `ceced09f794aca2280e25d7d44a914b58de79ba59701378aef404a7dd8b4602b` |

Total 1458107 bytes.

## Pinned downstream hashes (see the test)

- composed canvas (`compose_strip`, uint8 192x2272x3): `1a678eac4a83bd912f18830a7a85a88f8dbba1183b7bd200d8eac872209a21bd`
- float32 NCHW tensor (1x3x192x2272): `4e39b640d592fd3d2094bf5112cb3cd25939e890a906f9397f4f863455dfe329`
