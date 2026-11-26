# Migrating a pre-v2 checkpoint (96 object-class outputs) — opt-in

Main refuses a checkpoint whose `object_class_head` is not 80 wide:

```
epoch checkpoint object_class_head.weight: object_class has 96 outputs, this build expects 80 (16 x 5 classes ('vehicle', 'car', 'truck', 'bus', 'bike')). ...
```

That refusal is the **default and is unchanged**. `scripts/migrate_ckpt_classes.py` is a separate tool you run by hand.
It writes a new file; you then point a loader at that file explicitly. No loader detects, migrates, or falls back
to a migrated file, and no module under `python/` imports the tool (`scripts/test_ckpt_migrate.py` pins both).

## How to opt in

```
python scripts/migrate_ckpt_classes.py OLD.ckpt_ep11_step86808.npz NEW.migrated5.npz
# OUTPUT optional: default is <input>.migrated5.npz next to the input
```

* The input is read only, never modified. The output is create-only: **any existing path is refused with exit 2 and
  left untouched** (a real model file, an earlier migrated file, a dangling symlink, anything). There is no overwrite
  switch; `--force` is not an option and is an argument error (exit 2). To redo a migration, pick a new output name.
* An unreadable input (missing, a directory, empty, truncated, not an npz, not a `gvd-epoch-ckpt-1` file) is refused
  with exit 2 and a one-line `REFUSED:` message, never a traceback. The input is read once; the `source_sha256` in
  `meta["migration"]` is the hash of exactly those bytes.
* The write keeps the trainer's low-disk floor (`ensure_free_space` in `python/train/epoch_ckpt.py`, shared with
  `write_epoch_ckpt`): refused if it would leave under 8 GB free on the output volume. `--min-free-gb N` changes the
  floor for this run (the trainer's `--ckpt-min-free-gb` equivalent). `0` disables the floor. Non-finite or negative
  values (`nan`, `inf`, `-1`) are refused with exit 2 (an argument error, no traceback).
* The file is written to a temp name and linked into place create-only, so a file that appears at the output path
  during the run is refused, not replaced. Any other OS error while creating the output is also a refusal (exit 2) and
  the temp file is removed.
* Input must be a `gvd-epoch-ckpt-1` `.npz` with exactly 96 outputs on both `object_class_head.weight` and `.bias`.
  An 80-output file, any other count, a missing head, or any other tensor with a 96-wide leading dimension is refused.
* The summary is printed, and the same facts are written to `meta["migration"]` in the new file (mapping id, old and
  new class lists, where the old list came from, kept / dropped / merged, source file name and sha256).
* Load the result the usual way with an explicit path, e.g. `export_epoch_onnx.py --ckpt <NEW.migrated5.npz>` /
  `load_epoch_ckpt(path)`. The name `*.migrated5.npz` does not parse as an epoch-ckpt name, so `--ckpt-dir` +
  `--export-epoch` lookups skip it.
* Pause checkpoints (the trainer's `--resume` file) are a different format and are not handled.

## Where the old class list comes from

A `gvd-epoch-ckpt-1` file carries no class list: its meta has run, epoch, step, cams, arch, scale and so on, and
no class names. The old list is the `OBJECT_CLASSES` constant in `python/train/scene_net.py` at `4e41994`
(`4e41994b508b3ca2f77ce0727ce93d75bf77756d`). `git log -G'^OBJECT_CLASSES\s*=' 4e41994..main` lists one commit,
so the list is the same at every commit from `4e41994` to `29f6f0f^`. This is recorded in the output as
`old_classes_source`.

## Which commit changed 96 to 80

`29f6f0f` "recorder v2 ... no pedestrian class" (merged by PR #239, `4a14b73`, 2026-11-23): the people class was
removed on Michael's decision. `OBJECT_CLASSES` went from 6 to 5 names; `N_OBJECTS` stayed 16, so
16 x 6 = 96 became 16 x 5 = 80. `git bisect run` with a class-count probe on the first-parent history between
`4e41994` and `8ce34d5` lands on that merge. The bisect log is not part of this repo or PR; it lives in the swarm
notes folder outside the repo (`/workspace/swarm/swarm-1/notes/ckpt_migrate_bisect.log`).

## The mapping (`objclass-6to5-drop-people-v1`)

By class **name**, never by position. 5 kept (copied bit-for-bit), 1 dropped, 0 merged.

| old index | old class | new index | new class | action |
| --- | --- | --- | --- | --- |
| 0 | vehicle | 0 | vehicle | kept |
| 1 | car | 1 | car | kept |
| 2 | truck | 2 | truck | kept |
| 3 | bus | 3 | bus | kept |
| 4 | pedestrian | – | – | **DROPPED** (class removed by `29f6f0f`; no replacement) |
| 5 | bike | 4 | bike | kept |

There are no merges: nothing is averaged. If one is ever needed it must be written into `CLASS_POLICY` with a
reason; two old classes pointing at one new class raises `MappingError` (pinned by
`check_two_old_classes_one_new_class_refused`).

Row layout: `object_class_head` is `nn.Linear(hidden, 16 * n_classes)` read as `view(width, N_OBJECTS, n_classes)`,
so flat output row = `slot * n_classes + class`. New row `slot * 5 + j` is old row `slot * 6 + old_index(j)`,
for all 16 slots, on both `weight` (rows) and `bias`. These two tensors are the only ones whose leading dimension is
the class count (`scene_head_sizes()`); every other tensor is copied unchanged.

Unmapped names fail loudly. An old class name with no policy entry, a policy target that is not a current class, or a
current class with no old source raises `MappingError` naming it and nothing is written. A change to
`OBJECT_CLASSES` therefore breaks `scripts/test_ckpt_migrate.py` rather than shipping a wrong mapping.

## What the result is, and is not

A warm start or offline-eval file for the 5-class head. The weights were trained with the people class present, so
the remaining class logits were learned against that competitor. It is not a substitute for retraining; the guard
text still says so, and it is still what a plain load prints.
