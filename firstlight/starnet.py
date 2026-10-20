"""StarNet2 (starnetastro.com): neural-network star removal, run as a local command-line tool.

Not part of this repo and not installed by it; see localtools.py for where it is looked for
(`--starnet PATH`, `$FIRSTLIGHT_STARNET`, `~/tools/starnet2*/starnet2`, PATH). Run directly rather than through
Siril's `starnet` command, which was written for the older starnet++.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import localtools
from .localtools import ToolError

ENV = "FIRSTLIGHT_STARNET"
_override: Optional[str] = None


class StarNetError(ToolError):
    pass


def set_exe(path: Optional[str]) -> None:
    global _override
    _override = path


def find_exe() -> Optional[Path]:
    return localtools.find("starnet2", ENV, _override)


def remove_stars(exe: Path, src: Path, scratch: Path) -> np.ndarray:
    """The starless version of the linear FITS `src`, as float32 (same size).

    `--linear` stretches the data for the network and reverses the stretch afterwards. Samples above 1 are
    clipped on the way in, so a few saturated pixels may come back lower: starless + (src - starless) still
    equals src exactly because the caller builds the stars layer as the difference.
    """
    return localtools.run_on_fits(exe, ["--linear", "--quiet"], src, scratch, StarNetError)
