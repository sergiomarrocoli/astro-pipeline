"""DeepSNR: neural-network denoising, run as a local command-line tool (model 2, which also takes mono).

Not part of this repo and not installed by it; see localtools.py for where it is looked for
(`--deepsnr PATH`, `$FIRSTLIGHT_DEEPSNR`, `~/tools/deepsnr*/deepsnr`, PATH). Meant for the starless layer. Its README
warns that it expects uncorrelated noise, and a registered stack's is correlated, so check what it does to faint
structure before trusting it (see CLAUDE.md for what was measured).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import localtools
from .localtools import ToolError

ENV = "FIRSTLIGHT_DEEPSNR"
_override: Optional[str] = None


class DeepSNRError(ToolError):
    pass


def set_exe(path: Optional[str]) -> None:
    global _override
    _override = path


def find_exe() -> Optional[Path]:
    return localtools.find("deepsnr", ENV, _override)


def denoise(exe: Path, src: Path, scratch: Path) -> np.ndarray:
    """The denoised version of the linear FITS `src`, as float32 (same size)."""
    return localtools.run_on_fits(exe, ["--linear", "--quiet"], src, scratch, DeepSNRError)
