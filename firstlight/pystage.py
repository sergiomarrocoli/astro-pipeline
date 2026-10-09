"""Helpers for pipeline stages that run in Python (no Siril): cached by stamp, FITS in and out."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .stack import stamp


def py_stage(d: Path, name: str, inputs: list[Path], tag: str, outputs: list[str],
             fn: Callable[[], None], dry_run: bool) -> bool:
    """Like finish._stage but for in-process work: skip when outputs exist and the stamp matches."""
    d.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"--- {d.name}/{name}: python stage ({tag})")
        return False
    new = stamp([{"path": str(p)} for p in inputs], tag)
    sp = d / f".{name}.stamp"
    if all((d / o).exists() for o in outputs) and sp.exists() and sp.read_text() == new:
        print(f"{d.parent.name} / {name}: up to date, skipping")
        return False
    print(f"{d.parent.name} / {name}: running ...")
    fn()
    sp.write_text(new)
    return True


def read(path: Path) -> np.ndarray:
    from astropy.io import fits
    return fits.getdata(path).astype(np.float32)


def write(path: Path, data: np.ndarray, like: Path) -> None:
    """Write float32 data with the header of `like` (so FILTER, OBJECT, ROWORDER and the rest carry over)."""
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=fits.getheader(like)).writeto(path, overwrite=True)
