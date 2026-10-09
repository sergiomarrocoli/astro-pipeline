"""Per-frame quality: measure every light, compare it with its own group, and reject clear outliers.

A group is the frames of one target, filter and exposure. A frame is judged against its group's median,
so a slow, gentle change over the night is not penalised, but a frame that is suddenly soft, elongated or
starved of stars (a gust, a bump, a passing cloud) is. Rejection is deliberately conservative: it only
takes out frames that would visibly hurt a stack.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from . import starsplit

MIN_GROUP = 4        # need a few frames to know what "normal" is for a group
MIN_SHAPE_FWHM = 1.8  # below this stars are too undersampled for their shape (roundness) to be measured reliably

DEFAULTS = {
    "reject_fwhm_ratio": 1.20,       # reject if stars are this much wider than the group's median
    "reject_star_ratio": 0.75,       # ... or there are this much fewer stars (cloud, haze, dew)
    "reject_roundness_ratio": 0.88,  # ... or stars are this much less round (tracking, wind)
}


def _starlike(a: np.ndarray, y: int, x: int, bg: float) -> bool:
    """A real star has its eight neighbours clearly raised; a hot pixel does not."""
    p = a[y - 1:y + 2, x - 1:x + 2] - bg
    peak = p[1, 1]
    return peak > 0 and (p.sum() - peak) / 8 > 0.22 * peak


def measure_stars(a: np.ndarray, n: int = 250, min_sep: int = 16) -> np.ndarray:
    """Rows of (y, x, fwhm px, roundness minor/major, peak) for bright, unsaturated, isolated stars."""
    bg, sg = float(np.median(a[::7, ::7])), starsplit.noise_sigma(a)
    top = float(np.percentile(a, 99.995))
    ys, xs = np.nonzero((a > bg + 40 * sg) & (a < 0.85 * top))
    order = np.argsort(a[ys, xs])[::-1]
    taken: dict = {}
    picks = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if not (10 <= y < a.shape[0] - 10 and 10 <= x < a.shape[1] - 10) or a[y, x] != a[y - 2:y + 3, x - 2:x + 3].max():
            continue
        if not _starlike(a, y, x, bg):
            continue
        cell = (y // min_sep, x // min_sep)
        near = [taken.get((cell[0] + dy, cell[1] + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        if any(p and abs(p[0] - y) < min_sep and abs(p[1] - x) < min_sep for p in near):
            continue
        taken[cell] = (y, x)
        picks.append((y, x))
        if len(picks) >= n:
            break
    yy, xx = np.mgrid[-8:9, -8:9]
    out = []
    for y, x in picks:
        with np.errstate(all="ignore"):          # degenerate fits overflow harmlessly and are skipped below
            res = _fit_star(a, y, x, yy, xx)
        if res is not None:
            out.append(res)
    return np.array(out)


def _fit_star(a: np.ndarray, y: int, x: int, yy: np.ndarray, xx: np.ndarray):
    """(y, x, fwhm, roundness, peak) for the star at (y, x), or None if the fit is degenerate."""
    p = a[y - 8:y + 9, x - 8:x + 9]
    edge = np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]])
    w = np.maximum(p - np.median(edge), 0)
    # Adaptive moments with an elliptical Gaussian window. The window shrinks the measured covariance
    # (S_eff^-1 = S_true^-1 + S_win^-1), so divide it back out and iterate. A circular window would clip
    # the long axis of an elongated star and report it rounder than it is.
    cov = np.eye(2) * 2.5 ** 2
    for _ in range(12):
        try:
            inv_w = np.linalg.inv(cov)
            ww = w * np.exp(-0.5 * (inv_w[0, 0] * xx ** 2 + 2 * inv_w[0, 1] * xx * yy + inv_w[1, 1] * yy ** 2))
            tot = ww.sum()
            if not tot > 0:
                return None
            eff = np.array([[(ww * xx ** 2).sum(), (ww * xx * yy).sum()],
                            [(ww * xx * yy).sum(), (ww * yy ** 2).sum()]]) / tot
            true_inv = np.linalg.inv(eff) - inv_w
            if np.linalg.eigvalsh(true_inv).min() <= 1e-9:          # not resolvable: grow the window and retry
                cov = cov * 1.6
                continue
            cov = np.clip(np.linalg.inv(true_inv), -36, 36)
        except np.linalg.LinAlgError:                              # degenerate patch (flat, or a lone pixel)
            return None
    if not np.all(np.isfinite(cov)):
        return None
    ev = np.clip(np.linalg.eigvalsh(cov), 0.25 ** 2, 6.0 ** 2)
    return (y, x, 2.355 * float(np.sqrt(np.sqrt(ev[0] * ev[1]))), float(np.sqrt(ev[0] / ev[1])), float(w[8, 8]))


def count_stars(a: np.ndarray) -> int:
    """Stars above 8 sigma in a star-size band-pass (half resolution), each confirmed at full resolution.

    A hot pixel or cosmic ray can survive the smoothing, so a candidate only counts if its eight neighbours
    are clearly raised (see _starlike). Otherwise a rising cosmic-ray rate would hide a real loss of stars.
    """
    a = np.asarray(a, dtype=np.float32)
    half = a[::2, ::2]
    r = starsplit.gblur(half, 0.8) - starsplit.gblur(half, 3.0)
    rs = starsplit.robust_sigma(r)
    c = r[1:-1, 1:-1]
    mx = np.maximum.reduce([r[dy:r.shape[0] - 2 + dy, dx:r.shape[1] - 2 + dx] for dy in range(3) for dx in range(3)])
    ys, xs = np.nonzero((c > 8 * rs) & (c >= mx))
    bg = float(np.median(a[::7, ::7]))
    n = 0
    for y, x in zip(ys + 1, xs + 1):
        fy, fx = 2 * int(y), 2 * int(x)
        if not (3 <= fy < a.shape[0] - 3 and 3 <= fx < a.shape[1] - 3):
            continue
        block = a[fy - 2:fy + 3, fx - 2:fx + 3]
        j = np.unravel_index(int(np.argmax(block)), block.shape)
        py, px = fy - 2 + j[0], fx - 2 + j[1]
        if _starlike(a, py, px, bg) or _starlike_soft(a, py, px, bg):
            n += 1
    return n


def _starlike_soft(a: np.ndarray, y: int, x: int, bg: float) -> bool:
    """Looser version for the faint stars the band-pass finds: neighbours above ~12% of the peak."""
    p = a[y - 1:y + 2, x - 1:x + 2] - bg
    peak = p[1, 1]
    return peak > 0 and (p.sum() - peak) / 8 > 0.12 * peak


def measure_array(a: np.ndarray) -> dict:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    st = measure_stars(a)
    st = st[np.all(np.isfinite(st), axis=1)] if len(st) else st
    return {"fwhm": float(np.median(st[:, 2])) if len(st) else None,
            "roundness": float(np.median(st[:, 3])) if len(st) else None,
            "stars": count_stars(a), "measured": int(len(st)),
            "noise": float(starsplit.noise_sigma(a)), "sky": float(np.median(a[::5, ::5]))}


def measure_frame(path: Path) -> dict:
    from astropy.io import fits

    return measure_array(fits.getdata(path))


# ---- cache ----------------------------------------------------------------------------

MEASURE_VERSION = 2   # bump when the measurement changes, so cached numbers from the old method are never reused


def _key(path: Path) -> str:
    st = os.stat(path)
    return f"v{MEASURE_VERSION}:{Path(path).resolve()}:{st.st_size}:{st.st_mtime_ns}"


def measure_all(paths: list[Path], cache_file: Optional[Path] = None) -> dict[str, dict]:
    """Measure every frame once; results are cached by file path, size and mtime."""
    cache = {}
    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except ValueError:
            cache = {}
    out, dirty = {}, False
    for p in paths:
        k = _key(p)
        if k not in cache:
            cache[k] = measure_frame(p)
            dirty = True
        out[str(p)] = cache[k]
    if cache_file and dirty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    return out


# ---- judgement ------------------------------------------------------------------------

NEIGHBOURS = 4   # each frame is compared with this many of its nearest frames in time (same group)


def judge(metrics: dict[str, dict], groups: dict[str, str], cfg: Optional[dict] = None,
          order: Optional[dict[str, str]] = None) -> dict[str, list[str]]:
    """metrics: path -> measurements; groups: path -> group id; order: path -> sortable time (else input order).

    Returns path -> reasons to reject (empty if the frame is fine). A frame is compared with its nearest
    neighbours in time, not with the whole night, so a slow change (sky brightening toward dawn, focus
    creep) is not held against the frames at the end, while a sudden dip (a gust, a bump, a passing cloud)
    stands out against the frames around it.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    by_group = defaultdict(list)
    for p, g in groups.items():
        by_group[g].append(p)
    verdict: dict[str, list[str]] = {p: [] for p in groups}
    for g, paths in by_group.items():
        if len(paths) < MIN_GROUP:
            continue
        paths = sorted(paths, key=lambda p: (order or {}).get(p, ""))
        for i, p in enumerate(paths):
            others = sorted((q for q in paths if q != p), key=lambda q: abs(paths.index(q) - i))[:NEIGHBOURS]
            med = {}
            for k in ("fwhm", "roundness", "stars"):
                vals = [metrics[q][k] for q in others if metrics[q].get(k) is not None]
                if vals:
                    med[k] = float(np.median(vals))
            m, why = metrics[p], verdict[p]
            if m.get("fwhm") is not None and "fwhm" in med and m["fwhm"] > cfg["reject_fwhm_ratio"] * med["fwhm"]:
                why.append(f"soft stars: FWHM {m['fwhm']:.2f} px vs {med['fwhm']:.2f} in nearby frames")
            if "stars" in med and m["stars"] < cfg["reject_star_ratio"] * med["stars"]:
                why.append(f"few stars: {m['stars']} vs {int(med['stars'])} in nearby frames")
            shape_ok = med.get("fwhm", 0) >= MIN_SHAPE_FWHM      # undersampled stars: roundness is noise, skip it
            if (shape_ok and m.get("roundness") is not None and "roundness" in med
                    and m["roundness"] < cfg["reject_roundness_ratio"] * med["roundness"]):
                why.append(f"elongated stars: roundness {m['roundness']:.2f} vs {med['roundness']:.2f} in nearby frames")
    return verdict


MAX_REJECT_FRACTION = 0.25   # never throw away more than this share of a group, whatever the verdicts say


def cap(verdict: dict[str, list[str]], groups: dict[str, str], metrics: dict[str, dict],
        max_fraction: float = MAX_REJECT_FRACTION) -> dict[str, list[str]]:
    """Keep rejection modest: if a group would lose too many frames, spare the least bad ones.

    A night where "many frames are bad" is more likely a wrong threshold or a systematic problem than a
    string of bad luck, and stacking the lot beats stacking almost nothing.
    """
    by_group = defaultdict(list)
    for p, g in groups.items():
        by_group[g].append(p)
    out = {p: list(r) for p, r in verdict.items()}
    for g, paths in by_group.items():
        bad = [p for p in paths if out[p]]
        allowed = int(max_fraction * len(paths))
        if len(bad) <= allowed:
            continue
        # worst first: more reasons, then the wider stars
        bad.sort(key=lambda p: (len(out[p]), metrics[p].get("fwhm") or 0), reverse=True)
        for p in bad[allowed:]:
            out[p] = []
    return out
