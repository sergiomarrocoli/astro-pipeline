"""StarNet2 (starnetastro.com): neural-network star removal, run as a local command-line tool.

The program and its weights are not part of this repo and are not installed by it. They are found, in
order, at `--starnet PATH`, `$FIRSTLIGHT_STARNET`, `~/tools/starnet2*/starnet2`, then `starnet2` on PATH.
Run directly rather than through Siril's `starnet` command, which was written for the older starnet++.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .siril import SirilError

ENV = "FIRSTLIGHT_STARNET"
_override: Optional[str] = None


class StarNetError(SirilError):   # a SirilError, so callers that already skip a failed layer build keep working
    pass


def set_exe(path: Optional[str]) -> None:
    global _override
    _override = path


def find_exe() -> Optional[Path]:
    """The starnet2 executable, or None. An explicit path that does not exist is not replaced by a search."""
    explicit = _override or os.environ.get(ENV)
    if explicit:
        cands = [Path(explicit).expanduser()]
    else:
        cands = sorted(Path.home().glob("tools/starnet2*/starnet2"), reverse=True)
        on_path = shutil.which("starnet2")
        if on_path:
            cands.append(Path(on_path))
    return next((c for c in cands if c.is_file() and os.access(c, os.X_OK)), None)


def remove_stars(exe: Path, src: Path, scratch: Path) -> np.ndarray:
    """The starless version of the linear FITS `src`, as float32 (same size). `scratch` is an existing folder.

    `--linear` stretches the data for the network and reverses the stretch afterwards. Samples above 1 are
    clipped on the way in, so a few saturated pixels may come back lower: starless + (src - starless) still
    equals src exactly because the caller builds the stars layer as the difference.
    """
    from astropy.io import fits
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        out = Path(tmp) / "starless.fit"
        proc = subprocess.run([str(exe), "--linear", "--quiet", "-i", str(src), "-o", str(out)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
            raise StarNetError(f"starnet2 failed on {src.name} (exit {proc.returncode}):\n{tail}")
        return fits.getdata(out).astype(np.float32)
