"""A library of master calibration frames, kept across nights and keyed by what makes them match.

A dark must match the light's camera, binning, gain and offset exactly, its exposure within 0.5 s, and its sensor temperature
closely: dark current doubles every few degrees. It does not care about filter, target or night, so one
library serves every session until the settings or the camera change. Masters are built with Siril from raw
dark frames (found by IMAGETYP) and cached; building is skipped when the same source frames are already in.

Only darks are handled so far. The index records a `kind` so flats and bias can slot in later.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Optional

import numpy as np

from . import siril
from .stack import slug, stamp

DEFAULT_LIBRARY = Path.home() / ".firstlight" / "library"
DEFAULT_TEMP_TOLERANCE = 1.5   # degrees C between a master dark and the lights it is applied to
MIN_FRAMES = 3                  # fewer than this and a master would mostly add noise
GOOD_FRAMES = 10                # below this a master is built, with a warning
EXPOSURE_TOLERANCE = 0.5        # seconds: an 18.08 s light takes an 18 s dark


def library_path(explicit: Optional[str | Path] = None) -> Path:
    return Path(explicit or os.environ.get("FIRSTLIGHT_LIBRARY") or DEFAULT_LIBRARY).expanduser()


@dataclass(frozen=True)
class Conditions:
    """What has to agree between a dark and the lights it is applied to."""
    instrument: Optional[str]
    binning: Optional[int]
    gain: Optional[int]
    offset: Optional[int]
    exposure: Optional[float]

    @staticmethod
    def of(frame: dict) -> "Conditions":
        e = frame.get("exptime", frame.get("exposure"))
        return Conditions(frame.get("instrument"), frame.get("binning"), frame.get("gain"), frame.get("offset"),
                          round(float(e), 2) if e is not None else None)

    def label(self) -> str:
        bits = [self.instrument or "?", f"bin {self.binning}" if self.binning else None,
                f"gain {self.gain}" if self.gain is not None else None, f"offset {self.offset}" if self.offset is not None else None,
                f"{self.exposure:g} s" if self.exposure is not None else None]
        return ", ".join(b for b in bits if b)


def _cluster_temps(frames: list[dict], tol: float) -> list[list[dict]]:
    """Split frames taken at different sensor temperatures into groups (each within about tol/2 of its mean)."""
    known = sorted((f for f in frames if f.get("temp") is not None), key=lambda f: f["temp"])
    unknown = [f for f in frames if f.get("temp") is None]
    groups: list[list[dict]] = []
    for f in known:
        if groups and abs(f["temp"] - mean(g["temp"] for g in groups[-1])) <= tol / 2:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups + ([unknown] if unknown else [])


def group_darks(frames: list[dict], tol: float = DEFAULT_TEMP_TOLERANCE) -> dict[tuple[Conditions, int], list[dict]]:
    """Group dark frames by conditions, then by temperature cluster."""
    by_cond: dict[Conditions, list[dict]] = defaultdict(list)
    for f in frames:
        by_cond[Conditions.of(f)].append(f)
    return {(c, i): g for c, fs in by_cond.items() for i, g in enumerate(_cluster_temps(fs, tol))}


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        try:
            self.data = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.data = {"version": 1, "masters": []}

    def masters(self, kind: str = "dark") -> list[dict]:
        return [m for m in self.data["masters"] if m.get("kind", "dark") == kind]

    def path_of(self, entry: dict) -> Path:
        return self.root / entry["file"]

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.data, indent=1) + "\n")

    def add(self, entry: dict) -> None:
        self.data["masters"].append(entry)
        self.save()

    def has_sources(self, source_stamp: str) -> Optional[dict]:
        return next((m for m in self.data["masters"] if m.get("source_stamp") == source_stamp and self.path_of(m).exists()), None)

    # ---- matching -----------------------------------------------------------------------

    def find_dark(self, light: dict, tol: float = DEFAULT_TEMP_TOLERANCE) -> tuple[Optional[dict], str]:
        """The best master dark for these lights, and a plain-language reason when there is none.

        `light` has instrument, binning, gain, offset, exposure and temp (the median of the lights' sensor temps).
        """
        want = Conditions.of(light)
        temp = light.get("temp")
        pool = self.masters("dark")
        if not pool:
            return None, "the library has no dark frames yet"

        def same(m: dict) -> bool:
            c = Conditions.of(m)
            return (_eq(want.instrument, c.instrument) and _eq(want.binning, c.binning) and _eq(want.gain, c.gain)
                    and _eq(want.offset, c.offset)
                    and (want.exposure is None or c.exposure is None or abs(want.exposure - c.exposure) <= EXPOSURE_TOLERANCE))

        matching = [m for m in pool if same(m)]
        if matching:
            def dt(m): return abs(m["temp"] - temp) if (m.get("temp") is not None and temp is not None) else 0.0
            close = [m for m in matching if dt(m) <= tol]
            if close:
                best = min(close, key=lambda m: (dt(m), -m.get("n_frames", 0), "".join(chr(255 - ord(ch)) for ch in m.get("built", ""))))
                return best, f"matches on {want.label()}" + (f", {best['temp']:+.1f} C vs {temp:+.1f} C" if temp is not None and best.get("temp") is not None else "")
            nearest = min(matching, key=dt)
            return None, (f"darks exist for {want.label()} but at {nearest['temp']:+.1f} C and the lights were {temp:+.1f} C "
                          f"(tolerance {tol:g} C)")
        have = sorted({Conditions.of(m).label() for m in pool})
        return None, f"no dark for {want.label()}; the library has: {'; '.join(have[:4])}" + (" ..." if len(have) > 4 else "")


def _eq(a, b) -> bool:
    return a is None or b is None or a == b


# ---- building masters ------------------------------------------------------------------------

def master_script() -> str:
    return (f"requires {siril.MIN_VERSION}\nconvert dark\n"
            "stack dark rej w 3 3 -nonorm -32b -out=master_dark\nclose\n")


def _name(c: Conditions, temp: Optional[float], digest: str) -> str:
    t = f"t{temp:+.1f}" if temp is not None else "tNA"
    return (f"dark_{slug(c.instrument or 'cam')}_b{c.binning or 1}_g{c.gain}_o{c.offset}_{c.exposure:g}s_{t}_{digest[:8]}.fit")


def build_masters(frames: list[dict], library: Library, siril_bin: Optional[Path], tol: float = DEFAULT_TEMP_TOLERANCE,
                  dry_run: bool = False, run: Optional[Callable] = None, log: Callable[[str], None] = print) -> list[dict]:
    """Build a master dark for every group of raw darks that is not already in the library. Returns the new entries."""
    run = run or siril.run_script
    built = []
    for (cond, _), group in sorted(group_darks(frames, tol).items(), key=lambda kv: (kv[0][0].label(), kv[0][1])):
        temps = [f["temp"] for f in group if f.get("temp") is not None]
        temp = mean(temps) if temps else None
        label = f"{cond.label()}" + (f", {temp:+.1f} C" if temp is not None else "")
        if cond.exposure is None or cond.gain is None:
            log(f"darks: skipping {len(group)} frames without exposure or gain in the header ({label})")
            continue
        if len(group) < MIN_FRAMES:
            log(f"darks: only {len(group)} frame(s) for {label}; need at least {MIN_FRAMES}, skipping")
            continue
        src = stamp([{"path": f["path"]} for f in group], f"dark:{cond}:{tol}")
        existing = library.has_sources(src)
        if existing:
            log(f"darks: {label}: already in the library ({existing['file']})")
            continue
        if len(group) < GOOD_FRAMES:
            log(f"darks: warning: only {len(group)} frames for {label}; {GOOD_FRAMES}+ gives a cleaner master")
        file = _name(cond, temp, src)
        if dry_run:
            log(f"--- darks: would build {file} from {len(group)} frames")
            continue
        work = library.root / ".work" / file[:-4]
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            for i, f in enumerate(sorted(group, key=lambda f: f["path"])):
                os.symlink(f["path"], work / f"{i:04d}_{Path(f['path']).name}")
            script = work / "master.ssf"
            script.write_text(master_script())
            log(f"darks: building {label} from {len(group)} frames ...")
            run(siril_bin, work, script, work / "master.log")
            produced = work / "master_dark.fit"
            if not produced.exists():
                raise siril.SirilError(f"siril produced no master dark (see {work / 'master.log'})")
            library.root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), library.root / file)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        entry = {"kind": "dark", "file": file, "instrument": cond.instrument, "binning": cond.binning, "gain": cond.gain,
                 "offset": cond.offset, "exposure": cond.exposure, "temp": temp,
                 "temp_spread": pstdev(temps) if len(temps) > 1 else 0.0, "n_frames": len(group),
                 "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_stamp": src}
        library.add(entry)
        built.append(entry)
    return built


# ---- applying ----------------------------------------------------------------------------------

def light_conditions(frames: list[dict]) -> dict:
    """The attributes of a group of lights used for matching (temperature = median of the frames)."""
    first = frames[0]
    temps = [f["temp"] for f in frames if f.get("temp") is not None]
    return {"instrument": first.get("instrument"), "binning": first.get("binning"), "gain": first.get("gain"),
            "offset": first.get("offset"), "exposure": first.get("exptime"),
            "temp": float(np.median(temps)) if temps else None}


def orient_like(master: np.ndarray, master_top_down: bool, target_top_down: bool) -> np.ndarray:
    """A master dark has the row order Siril saved it in; match it to the frame it is applied to."""
    return master if master_top_down == target_top_down else master[::-1]


# ---- command line: firstlight-library ------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    import sys

    from .classify import DARK, build_manifest, scan

    ap = argparse.ArgumentParser(prog="firstlight-library", description="Manage the master dark library.")
    ap.add_argument("--library", type=Path, help="library folder (default: $FIRSTLIGHT_LIBRARY or ~/.firstlight/library)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the masters in the library")
    add = sub.add_parser("add", help="build masters from the dark frames in a folder")
    add.add_argument("folder", type=Path)
    add.add_argument("--siril")
    add.add_argument("--dry-run", action="store_true")
    chk = sub.add_parser("check", help="say which library dark each filter of a night's lights would get")
    chk.add_argument("folder", type=Path)
    args = ap.parse_args(argv)

    lib = Library(library_path(args.library))
    if args.cmd == "list":
        ms = lib.masters()
        if not ms:
            print(f"{lib.root}: no master darks yet (add some with: firstlight-library add <folder>)")
        for m in sorted(ms, key=lambda m: (Conditions.of(m).label(), m.get("temp") or 0)):
            t = f"{m['temp']:+.1f} C" if m.get("temp") is not None else "temp unknown"
            print(f"{Conditions.of(m).label()}  {t}  {m['n_frames']} frames  built {m['built'][:10]}  {m['file']}")
        return 0
    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 2
    manifest = build_manifest(scan(args.folder))
    if args.cmd == "add":
        darks = manifest["calibration"][DARK]
        if not darks:
            print("no dark frames found (IMAGETYP must say Dark)")
            return 1
        try:
            siril_bin = None if args.dry_run else siril.find_siril(args.siril)
            build_masters(darks, lib, siril_bin, dry_run=args.dry_run)
        except siril.SirilError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    for target, filters in manifest["targets"].items():     # check
        for flt, frames in sorted(filters.items()):
            found, why = lib.find_dark(light_conditions(frames))
            print(f"{target} / {flt}: " + (f"{found['file']} ({why})" if found else f"no dark: {why}"))
    return 0
