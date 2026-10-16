"""Undo Windows text-mode ``\\n`` -> ``\\r\\n`` damage in PNG files.

The collector used to ``os.open`` PNG bytes without ``O_BINARY``. On Windows
every ``0x0A`` was written as ``0x0D 0x0A``, so the signature
``\\x89PNG\\r\\n\\x1a\\n`` landed as ``\\x89PNG\\r\\r\\n\\x1a``.

Dropping the ``0x0D`` in every ``0x0D 0x0A`` pair reverses that translation.
A chunk whose CRC still fails is rebuilt by searching, per ambiguous
``0x0D 0x0A``, which of those CR bytes were inserted. The search checks the
PNG length field and the chunk CRC32. Repaired bytes are written beside the
originals. Originals are never deleted.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

from python.data.binary_io import PNG_SIG, write_atomic

_POLY = 0xEDB88320
_MAX_FULL = 18
_MAX_SIDE = 18
_MAX_CHUNK = 64 * 1024 * 1024


@dataclass(frozen=True)
class RepairResult:
    ok: bool
    data: bytes | None
    method: str


def undo_inserted_cr(data: bytes) -> bytes:
    """Drop the CR from every CR LF pair. The inverse of always expanding LF."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if data[i] == 0x0D and i + 1 < n and data[i + 1] == 0x0A:
            out.append(0x0A)
            i += 2
        else:
            out.append(data[i])
            i += 1
    return bytes(out)


def png_chunks_ok(blob: bytes) -> bool:
    """True when the signature, chunk lengths, and CRC32 values all match."""
    if not blob.startswith(PNG_SIG) or len(blob) < 8:
        return False
    i = 8
    saw_iend = False
    while i < len(blob):
        if i + 12 > len(blob):
            return False
        length = int.from_bytes(blob[i : i + 4], "big")
        if length < 0 or length > _MAX_CHUNK or i + 12 + length > len(blob):
            return False
        ctype = blob[i + 4 : i + 8]
        if not _type_ok(ctype):
            return False
        expect = zlib.crc32(blob[i + 4 : i + 8 + length]) & 0xFFFFFFFF
        got = int.from_bytes(blob[i + 8 + length : i + 12 + length], "big")
        if expect != got:
            return False
        i += 12 + length
        if ctype == b"IEND":
            saw_iend = True
            break
    return saw_iend and i == len(blob)


def png_decodes(blob: bytes) -> bool:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8).copy(), cv2.IMREAD_UNCHANGED)
    return image is not None and int(image.size) > 0


def repair_png_bytes(data: bytes) -> RepairResult:
    """Return repaired PNG bytes, or ``ok=False`` when the file cannot be saved."""
    if png_chunks_ok(data):
        return RepairResult(True, data, "valid")
    undone = undo_inserted_cr(data)
    if undone != data and png_chunks_ok(undone):
        return RepairResult(True, undone, "drop_cr")
    searched = rebuild_by_crc(data)
    if searched is not None and png_chunks_ok(searched):
        return RepairResult(True, searched, "crc_search")
    return RepairResult(False, None, "failed")


def rebuild_by_crc(data: bytes) -> bytes | None:
    """Walk chunks, and at each ambiguous CR keep the choice whose CRC matches."""
    sig = _search_logical(data, 0, 8, lambda buf: buf == PNG_SIG)
    if sig is None:
        return None
    _logical, pos = sig
    out = bytearray(PNG_SIG)
    for _guard in range(100_000):
        if pos >= len(data):
            return None
        chosen: tuple[bytes, int, bytes] | None = None
        for length_bytes, pos2 in _length_options(data, pos):
            length = int.from_bytes(length_bytes, "big")
            if length > _MAX_CHUNK or length > len(data):
                continue
            found = _search_chunk_rest(data, pos2, length_bytes)
            if found is None:
                continue
            rest, pos3 = found
            chunk = length_bytes + rest
            ctype = chunk[4:8]
            if len(out) == 8 and ctype != b"IHDR":
                continue
            if not _type_ok(ctype):
                continue
            chosen = (chunk, pos3, ctype)
            break
        if chosen is None:
            return None
        chunk, pos, ctype = chosen
        out += chunk
        if ctype == b"IEND":
            blob = bytes(out)
            if png_chunks_ok(blob):
                return blob
            return None
    return None


def repair_dataset(
    root: Path | str,
    *,
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, object]:
    """Repair PNGs under ``root``. Write copies next to the capture. Never delete.

    A dataset directory that contains ``raw/`` is written to ``raw_repaired/``.
    A single file is written as ``<stem>.repaired.png`` in the same folder.
    """
    root = Path(root)
    scan_root, dataset_root = _scan_layout(root)
    files = _pngs(scan_root)
    if limit > 0:
        files = files[: int(limit)]
    total = len(files)
    rescued = 0
    already = 0
    repaired = 0
    failed = 0
    failed_names: list[str] = []
    for index, src in enumerate(files, start=1):
        original = _read(src)
        if original is None:
            failed += 1
            failed_names.append(src.name)
            continue
        if png_chunks_ok(original) and png_decodes(original):
            already += 1
            rescued += 1
            _tick(index, total, rescued)
            continue
        result = repair_png_bytes(original)
        if not result.ok or result.data is None or not png_decodes(result.data):
            failed += 1
            failed_names.append(src.name)
            _tick(index, total, rescued)
            continue
        if not dry_run and result.data != original:
            dest = _dest_for(src, scan_root, dataset_root)
            if dest.resolve() == src.resolve():
                raise RuntimeError(f"refusing to overwrite original {src}")
            if dataset_root is not None:
                raw = (dataset_root / "raw").resolve()
                if _is_inside(dest, raw):
                    raise RuntimeError(f"refusing to write repaired bytes into {raw}")
            write_atomic(dest, result.data)
        repaired += 1
        rescued += 1
        _tick(index, total, rescued)
    fraction = (float(rescued) / float(total)) if total else 0.0
    return {
        "total": total,
        "rescued": rescued,
        "fraction": fraction,
        "already_valid": already,
        "repaired": repaired,
        "failed": failed,
        "failed_names": failed_names[:20],
        "dry_run": bool(dry_run),
    }


def _tick(index: int, total: int, rescued: int) -> None:
    if index == total or index % 500 == 0:
        print(f"repair progress {index}/{total} rescued {rescued}", flush=True)


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _scan_layout(path: Path) -> tuple[Path, Path | None]:
    if path.is_file():
        return path, None
    if path.name == "raw" and path.is_dir():
        return path, path.parent
    if (path / "raw").is_dir():
        return path / "raw", path
    return path, None


def _pngs(scan_root: Path) -> list[Path]:
    if scan_root.is_file():
        return [scan_root] if scan_root.suffix.lower() == ".png" else []
    found: list[Path] = []
    for path in scan_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() != ".png":
            continue
        if path.name.endswith(".tmp") or ".repaired" in path.name:
            continue
        if "raw_repaired" in path.parts:
            continue
        found.append(path)
    return sorted(found)


def _dest_for(src: Path, scan_root: Path, dataset_root: Path | None) -> Path:
    if dataset_root is not None and scan_root == dataset_root / "raw":
        return dataset_root / "raw_repaired" / src.relative_to(scan_root)
    return src.with_name(src.stem + ".repaired.png")


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except ValueError:
        return False
    return True


def _type_ok(ctype: bytes) -> bool:
    if len(ctype) != 4:
        return False
    return all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in ctype)


def _rest_crc_ok(length_bytes: bytes, rest: bytes) -> bool:
    length = int.from_bytes(length_bytes, "big")
    if len(rest) != length + 8:
        return False
    if not _type_ok(rest[:4]):
        return False
    expect = zlib.crc32(rest[: 4 + length]) & 0xFFFFFFFF
    got = int.from_bytes(rest[4 + length :], "big")
    return expect == got


def crc32_reverse(data: bytes, crc_out: int) -> int:
    """Initial zlib CRC such that ``crc32(data, initial) == crc_out``."""
    crc = int(crc_out) & 0xFFFFFFFF
    crc ^= 0xFFFFFFFF
    for byte in reversed(data):
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc ^ _POLY) << 1) | 1
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            crc &= 0xFFFFFFFF
        crc ^= byte
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _read_drop_all(data: bytes, pos: int, need: int) -> tuple[bytes, int] | None:
    out = bytearray()
    i = pos
    n = len(data)
    while len(out) < need:
        if i >= n:
            return None
        if data[i] == 0x0D and i + 1 < n and data[i + 1] == 0x0A:
            out.append(0x0A)
            i += 2
        else:
            out.append(data[i])
            i += 1
    return bytes(out), i


def sites_covering(data: bytes, pos: int, need: int) -> list[int]:
    """Indexes of ``0x0D 0x0A`` that a read of ``need`` logical bytes can touch."""
    if need < 0 or pos < 0:
        return []
    end = min(len(data), pos + need)
    while True:
        sites = [
            i
            for i in range(pos, max(pos, end - 1))
            if data[i] == 0x0D and i + 1 < end and data[i + 1] == 0x0A
        ]
        new_end = min(len(data), pos + need + len(sites))
        if new_end == end:
            return sites
        end = new_end


def _length_options(data: bytes, pos: int) -> list[tuple[bytes, int]]:
    opts: list[tuple[bytes, int]] = []
    seen: set[tuple[bytes, int]] = set()
    fast = _read_drop_all(data, pos, 4)
    if fast is not None:
        opts.append(fast)
        seen.add(fast)
    sites = sites_covering(data, pos, 4)
    if len(sites) > 12:
        return opts
    for logical, end, _mask in _enum_all(data, pos, 4, sites):
        key = (logical, end)
        if key in seen or len(logical) != 4:
            continue
        seen.add(key)
        opts.append((logical, end))
    return opts


def _search_logical(data: bytes, pos: int, need: int, accept) -> tuple[bytes, int] | None:
    fast = _read_drop_all(data, pos, need)
    if fast is not None and accept(fast[0]):
        return fast
    sites = sites_covering(data, pos, need)
    if len(sites) > _MAX_FULL:
        return _flip(data, pos, need, sites, accept)
    for logical, end, _mask in _enum_all(data, pos, need, sites):
        if accept(logical):
            return logical, end
    return None


def _search_chunk_rest(data: bytes, pos: int, length_bytes: bytes) -> tuple[bytes, int] | None:
    length = int.from_bytes(length_bytes, "big")
    need = 8 + length

    def accept(rest: bytes, lb: bytes = length_bytes) -> bool:
        return _rest_crc_ok(lb, rest)

    fast = _read_drop_all(data, pos, need)
    if fast is not None and accept(fast[0]):
        return fast
    sites = sites_covering(data, pos, need)
    if len(sites) <= _MAX_FULL:
        for logical, end, _mask in _enum_all(data, pos, need, sites):
            if accept(logical):
                return logical, end
        return None
    found = _mitm_crc(data, pos, length_bytes, length)
    if found is not None:
        return found
    return _flip(data, pos, need, sites, accept)


def _enum_all(data: bytes, pos: int, need: int, sites: list[int]):
    buf = bytearray()
    yield from _enum_rec(data, pos, need, sites, 0, 0, buf, 0)


def _enum_rec(data, i, need, sites, si, mask, buf, start):
    """Branch at each CR LF. Restore ``buf`` before returning to the caller."""
    checkpoint = len(buf)
    try:
        n = len(data)
        while need > 0:
            if i >= n:
                return
            while si < len(sites) and sites[si] < i:
                si += 1
            if si < len(sites) and sites[si] == i and i + 1 < n and data[i + 1] == 0x0A:
                buf.append(0x0A)
                yield from _enum_rec(
                    data, i + 2, need - 1, sites, si + 1, mask | (1 << si), buf, start
                )
                buf.pop()
                buf.append(data[i])
                yield from _enum_rec(data, i + 1, need - 1, sites, si + 1, mask, buf, start)
                buf.pop()
                return
            buf.append(data[i])
            i += 1
            need -= 1
        if need == 0:
            yield bytes(buf[start:]), i, mask
    finally:
        del buf[checkpoint:]


def _produce(data: bytes, pos: int, need: int, sites: list[int], mask: int) -> tuple[bytes, int] | None:
    out = bytearray()
    i = pos
    si = 0
    n = len(data)
    while need > 0:
        if i >= n:
            return None
        while si < len(sites) and sites[si] < i:
            si += 1
        if si < len(sites) and sites[si] == i and i + 1 < n and data[i + 1] == 0x0A:
            if (mask >> si) & 1:
                out.append(0x0A)
                i += 2
            else:
                out.append(data[i])
                i += 1
            si += 1
            need -= 1
            continue
        out.append(data[i])
        i += 1
        need -= 1
    return bytes(out), i


def _balance(data: bytes, pos: int, body_n: int) -> int:
    n1 = max(1, body_n // 2)
    for _ in range(24):
        left_n = len(sites_covering(data, pos, n1))
        right_n = len(sites_covering(data, pos + n1, body_n - n1 + 4))
        if left_n <= _MAX_SIDE and right_n <= _MAX_SIDE:
            return n1
        if left_n > right_n and n1 > 1:
            scaled = int(n1 * _MAX_SIDE / max(left_n, 1))
            n1 = max(1, min(body_n - 1, scaled))
        elif n1 < body_n - 1:
            n1 = min(body_n - 1, n1 + max(1, (body_n - n1) // 3))
        else:
            break
    return max(1, min(body_n - 1, n1))


def _mitm_crc(data: bytes, pos: int, length_bytes: bytes, length: int) -> tuple[bytes, int] | None:
    """Split the chunk body and join halves whose CRC32 meets the stored CRC."""
    body_n = 4 + length
    if body_n < 2:
        return None
    n1 = _balance(data, pos, body_n)
    n2 = body_n - n1
    left_sites = sites_covering(data, pos, n1)
    if len(left_sites) > _MAX_SIDE:
        return None
    if len(sites_covering(data, pos + n1, n2 + 4)) > _MAX_SIDE:
        return None
    groups: dict[int, dict[int, list[int]]] = {}
    for logical, end, mask in _enum_all(data, pos, n1, left_sites):
        crc_l = zlib.crc32(logical) & 0xFFFFFFFF
        groups.setdefault(end, {}).setdefault(crc_l, []).append(mask)
    for end, by_crc in groups.items():
        right_sites = sites_covering(data, end, n2 + 4)
        if len(right_sites) > _MAX_SIDE:
            continue
        for logical, end_r, _mask_r in _enum_all(data, end, n2 + 4, right_sites):
            if len(logical) != n2 + 4:
                continue
            body = logical[:n2]
            target = int.from_bytes(logical[n2:], "big")
            need_crc = crc32_reverse(body, target)
            masks = by_crc.get(need_crc)
            if not masks:
                continue
            for mask_l in masks:
                left = _produce(data, pos, n1, left_sites, mask_l)
                if left is None or left[1] != end:
                    continue
                rest = left[0] + logical
                if _rest_crc_ok(length_bytes, rest):
                    return rest, end_r
    return None


def _flip(data, pos, need, sites, accept) -> tuple[bytes, int] | None:
    k = len(sites)
    if k == 0 or k > 4000:
        return None
    base = (1 << k) - 1
    depths = (1,) if k > 64 else (1, 2)
    for depth in depths:
        for bits in _combinations(k, depth):
            mask = base
            for bit in bits:
                mask ^= 1 << bit
            got = _produce(data, pos, need, sites, mask)
            if got is not None and accept(got[0]):
                return got
    return None


def _combinations(k: int, depth: int):
    if depth == 1:
        for i in range(k):
            yield (i,)
        return
    for i in range(k):
        for j in range(i + 1, k):
            yield (i, j)
