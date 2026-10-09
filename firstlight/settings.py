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


def measure_fwhm(a: np.ndarray, n: int = 200, min_sep: int = 14) -> float:
    """Median star FWHM in pixels, from the half-maximum radius of bright, unsaturated, isolated stars."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    bg, sg = float(np.median(a[::7, ::7])), starsplit.noise_sigma(a)
    top = float(np.percentile(a, 99.995))
    ys, xs = np.nonzero((a > bg + 40 * sg) & (a < 0.7 * top))        # bright but not saturated
    order = np.argsort(a[ys, xs])[::-1]
    taken: dict[tuple[int, int], tuple[int, int]] = {}
    picks = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        cell = (y // min_sep, x // min_sep)
        near = [taken.get((cell[0] + dy, cell[1] + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        if any(p and abs(p[0] - y) < min_sep and abs(p[1] - x) < min_sep for p in near):
            continue
        # must be the local maximum of its neighbourhood, with room around it
        if 12 <= y < a.shape[0] - 12 and 12 <= x < a.shape[1] - 12 and a[y, x] == a[y - 2:y + 3, x - 2:x + 3].max():
            taken[cell] = (y, x)
            picks.append((y, x))
        if len(picks) >= n:
            break
    yy, xx = np.mgrid[-8:9, -8:9]
    out = []
    for y, x in picks:
        p = a[y - 8:y + 9, x - 8:x + 9]
        edge = np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]])
        w = np.maximum(p - np.median(edge), 0)
        if w[8, 8] <= 0:
            continue
        # Adaptive Gaussian-window moments: the window's own width shrinks the measured variance, so divide
        # it back out (1/s_true^2 = 1/s_eff^2 - 1/s_win^2) and iterate. Converges to the fitted sigma.
        sig = 2.5
        r2 = yy ** 2 + xx ** 2
        for _ in range(10):
            ww = w * np.exp(-r2 / (2 * sig ** 2))
            tot = ww.sum()
            if tot <= 0:
                break
            eff2 = 0.5 * (ww * r2).sum() / tot
            inv = 1.0 / eff2 - 1.0 / sig ** 2
            sig = float(np.clip(np.sqrt(1.0 / inv) if inv > 1e-9 else sig * 2, 0.5, 6.0))
        out.append(2.355 * sig)
    return float(np.clip(np.median(out), 1.2, 8.0)) if out else starsplit.DEFAULT_FWHM


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
