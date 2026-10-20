"""Local command-line programs that are not part of this repo (StarNet2, DeepSNR): finding and running them.

Each is found, in order, at an explicit path (a flag, then its environment variable), then
`~/tools/<program>*/<program>`, then on PATH. They are run on a FITS file and give a FITS file back.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .siril import SirilError


class ToolError(SirilError):   # a SirilError, so callers that already skip a failed layer build keep working
    pass


def find(program: str, env: str, override: Optional[str] = None) -> Optional[Path]:
    """The executable, or None. An explicit path that does not exist is not replaced by a search."""
    explicit = override or os.environ.get(env)
    if explicit:
        cands = [Path(explicit).expanduser()]
    else:
        cands = sorted(Path.home().glob(f"tools/{program}*/{program}"), reverse=True)
        on_path = shutil.which(program)
        if on_path:
            cands.append(Path(on_path))
    return next((c for c in cands if c.is_file() and os.access(c, os.X_OK)), None)


def run_on_fits(exe: Path, args: Sequence[str], src: Path, scratch: Path, error: type = ToolError) -> np.ndarray:
    """Run `exe <args> -i src -o out.fit` and return the output image as float32. `scratch` is an existing folder."""
    from astropy.io import fits
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        out = Path(tmp) / "out.fit"
        proc = subprocess.run([str(exe), *args, "-i", str(src), "-o", str(out)], capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
            raise error(f"{exe.name} failed on {src.name} (exit {proc.returncode}):\n{tail}")
        return fits.getdata(out).astype(np.float32)
