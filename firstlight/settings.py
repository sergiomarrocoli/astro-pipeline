"""Per-target settings: measured from the data, saved to <target>/settings.json, overridable.

The file has an "auto" block (rewritten each run with what was measured and chosen) and an "override"
block (yours, never touched). The effective value is the override if present, else the auto one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import background, quality, starsplit

# Chosen defaults (not measured). Each was tuned on a real stack; see enhance.py / starsplit.py.
DEFAULTS = {
    "star_threshold": starsplit.DEFAULT_THRESH,  # star detection, in band-pass noise multiples
    # Richardson-Lucy with a round PSF over-sharpens the short axis of an elongated star, leaving thin streaks.
    # On the test night sharpening saturates by ~4 iterations (star FWHM 2.52 -> 1.98 px, 22% at 10) while stars
    # lose 13% of their roundness at 4 iterations and 33% at 10. So 4, the knee of that curve.
    "rl_iterations": 4,                           # multiplicative Richardson-Lucy iterations on the stars layer
    # Fully denoised (1.0) turns noise into a patchwork of curved strokes (non-local-means artefact) that a steep
    # stretch makes obvious. Blending 30% of the original back restores natural grain that hides it; colour noise is
    # handled separately by the widget's colour smoothing, so luminance can keep some grain.
    "denoise_amount": 0.7,                        # blend of the denoised starless layer (1 = fully denoised)
    "star_gate": 2.5,                             # noise gate on the stars layer, in sky-noise sigmas
    **quality.DEFAULTS,                           # frame rejection thresholds (see quality.py)
    "dark_temp_tolerance": 1.5,                   # max degrees C between a library dark and the lights it calibrates
    "background_degree": background.DEFAULT_DEGREE,   # sky gradient removal: surface degree (2 = quadratic, 1 = plane), 0 = off
}


def measure_fwhm(a: np.ndarray) -> float:
    """Median star FWHM in pixels of a stack. One definition for the whole tool: the same stars and the
    same elliptical adaptive-moments fit as the frame-quality measurement and the report's field map."""
    st = quality.measure_stars(np.nan_to_num(np.asarray(a, dtype=np.float32)))
    st = st[np.all(np.isfinite(st), axis=1)] if len(st) else st
    return float(np.clip(np.median(st[:, 2]), 1.2, 8.0)) if len(st) else starsplit.DEFAULT_FWHM


def load(tdir: Path) -> dict:
    p = tdir / "settings.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def current(tdir: Path) -> dict:
    """The chosen defaults with this target's overrides applied, without measuring anything.

    For stages that run before the measured settings exist (frame rejection, background removal).
    """
    return {**DEFAULTS, **load(tdir).get("override", {})}


def effective(tdir: Path, auto: dict) -> dict:
    """Merge: the override block wins. Rewrites the file's auto block, keeps the override block."""
    saved = load(tdir)
    override = saved.get("override", {})
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "settings.json").write_text(json.dumps({"auto": auto, "override": override}, indent=2) + "\n")
    return {**auto, **override}


def resolve(tdir: Path, linear_paths: dict[str, Path]) -> dict:
    """Measure what can be measured, add the chosen defaults, apply overrides and save."""
    from astropy.io import fits

    # Prefer H (the strongest filter in narrowband) for the PSF; any filter works.
    pick = "H" if "H" in linear_paths else sorted(linear_paths)[0]
    first = fits.getdata(linear_paths[pick]).astype(np.float32)
    # Sky noise of the plain stacks. The widget's black point is set in these units, so it still
    # clips the grain after the layers have been denoised.
    noise = {n: float(starsplit.noise_sigma(fits.getdata(path).astype(np.float32))) for n, path in linear_paths.items()}
    auto = {"fwhm": round(measure_fwhm(first), 2), "fwhm_measured_on": pick, "noise": noise, **DEFAULTS}
    return effective(tdir, auto)
