"""Stage 3: calibrate (if masters exist), register and stack one target/filter."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import siril
from .siril import MIN_VERSION

REJECTION = "w 3 3"  # winsorized sigma, low/high


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "x"


@dataclass
class Masters:
    dark: Optional[Path] = None
    flat: Optional[Path] = None
    bias: Optional[Path] = None


def build_script(masters: Masters, name: str, hot: bool = False) -> str:
    lines = [f"requires {MIN_VERSION}", "convert light"]
    seq = "light"
    args = [f"-{k}={getattr(masters, k)}" for k in ("bias", "dark", "flat") if getattr(masters, k)]
    if args:
        lines.append(f"calibrate light {' '.join(args)}" + (" -cc=dark" if masters.dark else ""))
        seq = "pp_light"
    if hot:
        # Sensor hot pixels, found from the lights themselves (see hotpixels.py), fixed before registration
        # so they are not smeared into streaks. hot.lst is linked into the work folder by stack_filter.
        lines.append(f"seqcosme {seq} hot.lst")
        seq = f"cosme_{seq}"
    lines += [
        f"register {seq} -2pass",
        f"seqapplyreg {seq} -framing=min",
        f"stack r_{seq} rej {REJECTION} -norm=addscale -weight=wfwhm -32b -out={name}",
        "close",
    ]
    return "\n".join(lines) + "\n"


def dominant_exposure(frames: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split frames into (main exposure group, the rest). Main = most total integration time."""
    totals: dict = {}
    for f in frames:
        totals[f["exptime"]] = totals.get(f["exptime"], 0) + (f["exptime"] or 0) or 1
    main = max(totals, key=lambda e: (totals[e], e or 0))
    return [f for f in frames if f["exptime"] == main], [f for f in frames if f["exptime"] != main]


def stamp(frames: list[dict], script: str) -> str:
    h = hashlib.sha256(script.encode())
    for f in sorted(frames, key=lambda f: f["path"]):
        st = os.stat(f["path"])
        h.update(f"{f['path']}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def stack_filter(
    target: str, flt: str, frames: list[dict], out: Path, siril_bin: Optional[Path],
    masters: Masters = Masters(), dry_run: bool = False, keep: bool = False,
    hot_list: Optional[Path] = None,
) -> Optional[Path]:
    """Returns the stack path, or None on dry run. Re-runs are skipped when inputs are unchanged."""
    d = out / slug(target) / slug(flt)
    work = d / "work"
    name = f"stack_{slug(flt)}"
    result = d / f"{name}.fit"
    # A matching master dark calibrates the hot pixels away, so the list found from the lights is only a fallback
    # (and for a one-frame "stack", which Siril cannot calibrate, it is still the only fix available).
    master_dark = masters.dark
    if master_dark:
        masters = Masters(dark=Path("master_dark.fit"), flat=masters.flat, bias=masters.bias)
    script = (build_script(masters, name, hot=hot_list is not None and not master_dark) if len(frames) > 1
              else "# one frame: scaled to 0..1 in Python, no Siril run\n")
    script_path = d / f"{name}.ssf"
    stamp_path = d / ".stamp"
    extra = ""
    if hot_list and hot_list.exists() and not (master_dark and len(frames) > 1):
        extra += hashlib.sha256(hot_list.read_bytes()).hexdigest()
    if master_dark:
        st = master_dark.stat()
        extra += f"dark:{master_dark.name}:{st.st_size}:{st.st_mtime_ns}"
    new_stamp = stamp(frames, script + extra)

    d.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    if dry_run:
        print(f"--- {target} / {flt}: {len(frames)} frames, script {script_path}\n{script}")
        return None
    if result.exists() and stamp_path.exists() and stamp_path.read_text() == new_stamp:
        print(f"{target} / {flt}: up to date, skipping")
        return result

    if len(frames) == 1:
        # Siril cannot register or stack a one-frame sequence. Use the frame as it is, scaled to 0..1 like
        # Siril's own stacks, so the later steps (cross-filter alignment) can treat it like any other filter.
        print(f"warning: {target} / {flt}: only 1 frame, using it as is (this is not a real stack)")
        _single_frame(Path(frames[0]["path"]), result, hot_list)
        stamp_path.write_text(new_stamp)
        return result

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    for i, f in enumerate(sorted(frames, key=lambda f: f["path"])):
        os.symlink(f["path"], work / f"{i:04d}_{Path(f['path']).name}")  # input stays read-only
    if hot_list and not master_dark:
        os.symlink(hot_list.resolve(), work / "hot.lst")
    if master_dark:
        os.symlink(master_dark.resolve(), work / "master_dark.fit")
    print(f"{target} / {flt}: stacking {len(frames)} frames ...")
    try:
        siril.run_script(siril_bin, work, script_path, d / f"{name}.log")
    finally:
        if not keep:
            _clean(work)
    produced = work / f"{name}.fit"
    if not produced.exists():
        raise siril.SirilError(f"siril produced no {name}.fit (see {d / (name + '.log')})")
    shutil.move(str(produced), result)
    stamp_path.write_text(new_stamp)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _single_frame(src: Path, dest: Path, hot_list: Optional[Path] = None) -> None:
    from astropy.io import fits
    import numpy as np

    from . import hotpixels

    with fits.open(src) as hdul:
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.float32(np.iinfo(data.dtype).max)   # 16-bit counts -> 0..1, as Siril does
    if hot_list:
        top_down = str(header.get("ROWORDER", "BOTTOM-UP")).strip().upper() == "TOP-DOWN"
        data = hotpixels.apply(data, hotpixels.read_list(hot_list, data.shape, top_down))
    for key in ("BZERO", "BSCALE"):
        header.remove(key, ignore_missing=True)
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(dest, overwrite=True)


def _clean(work: Path) -> None:
    """Drop calibrated/registered sequences (many GB); keep only the stack."""
    for p in work.iterdir():
        if re.match(r"(pp_|r_|bkg_|cosme_)*light", p.name) or p.name in ("cache", "process", "master_dark.fit"):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
