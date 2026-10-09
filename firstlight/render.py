"""Render the draft image: the widget's default look, in numpy, so the pipeline can produce a picture by itself.

This is a port of widget_dsp.js and the widget's render path (normalise, midtones stretch with contrast,
SCNR with an auto-detected amount, saturation, colour smoothing of chrominance only, then the stars layer
stretched with asinh and screen-blended on top). Keep the two in step: tests/test_render.py checks the
primitives against the JavaScript ones in a real browser.
"""
from __future__ import annotations

import numpy as np

from . import starsplit
from .widget import channel_pack, default_assignment, downsample, quantise, star_stretch_k

# the widget's default control values
TARGET_BG = 0.22
BLACK_SIGMA = -2.8
CONTRAST = 0.15
SATURATION = 1.2
COLOUR_SMOOTH = 2.0     # px at a 1000 px wide image; scaled with the output width
STAR_BRIGHTNESS = 1.0
STAR_COLOUR = 0.5
STAR_COLOUR_KNEE = 0.35


def mtf(m: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = (m - 1) * x / ((2 * m - 1) * x - m)
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, y)).astype(np.float32)


def solve_m(x: float, t: float) -> float:
    return 0.5 if (x <= 0 or x >= 1) else x * (t - 1) / (2 * t * x - t - x)


def stretch(c: dict, p: np.ndarray, contrast: float) -> np.ndarray:
    x = np.clip((p - c["lo"]) / c["span"], 0.0, 1.0)
    y = mtf(c["m"], x)
    if contrast:
        y = y + contrast * (y * y * (3 - 2 * y) - y)
    return y


def star_stretch(k: float, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.where(q <= 0, 0.0, np.arcsinh(k * q) / np.arcsinh(k)).astype(np.float32)


def scnr(r: np.ndarray, g: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    cap = (r + b) / 2
    return np.where(g > cap, g - amount * (g - cap), g)


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) * (1 - b)


def auto_scnr(r: float, g: float, b: float) -> float:
    ratio = g / max((r + b) / 2, 1e-6)
    return 0.0 if ratio <= 1.05 else min(1.0, (ratio - 1.05) * 4)


def chroma_smooth(r: np.ndarray, g: np.ndarray, b: np.ndarray, sigma: float):
    """Blur Cb and Cr only, keeping luminance: returns new (r, g, b)."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb, cr = starsplit.gblur(b - y, sigma), starsplit.gblur(r - y, sigma)
    r2, b2 = y + cr, y + cb
    return r2, (y - 0.299 * r2 - 0.114 * b2) / 0.587, b2


def render(layers: dict[str, tuple[np.ndarray, np.ndarray]], noise: dict[str, float] | None = None,
           width: int = 1600, assignment: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """layers: filter -> (starless, stars) full-resolution arrays. Returns (uint8 RGB image, the choices made)."""
    names = list(layers)
    assignment = assignment or default_assignment(names)
    small = {n: (downsample(sl, width), downsample(st, width)) for n, (sl, st) in layers.items()}
    chan = {}
    for n in assignment:
        if n in chan:
            continue
        sl, st = small[n]
        pack = channel_pack(sl)
        sig = (noise[n] / pack["vmax"]) if noise and n in noise else pack["sigma"]
        svmax = float(np.percentile(st, 99.99)) or 1.0
        v = np.clip(np.nan_to_num(sl) / pack["vmax"], 0, 1)
        sv = np.clip(np.nan_to_num(st) / max(svmax, 1e-12), 0, 1)
        chan[n] = {"v": v, "sv": sv, "bg": pack["bg"], "p99": pack["p99"], "sig": sig, "k": star_stretch_k(sv)}
    ref = float(np.mean([chan[n]["p99"] for n in assignment]))
    for c in chan.values():                                  # per-channel stretch state, as in the page
        scale = ref / c["p99"]
        lo, hi = BLACK_SIGMA * c["sig"] * scale, (1 - c["bg"]) * scale
        c.update(scale=scale, lo=lo, span=hi - lo)
        c["m"] = solve_m((0 - lo) / (hi - lo), TARGET_BG)

    def nebula(scnr_amount):
        rgb = [stretch(chan[n], (chan[n]["v"] - chan[n]["bg"]) * chan[n]["scale"], CONTRAST) for n in assignment]
        r, g, b = rgb
        if scnr_amount > 0:
            g = scnr(r, g, b, scnr_amount)
        return r, g, b

    r, g, b = nebula(0.0)
    amount = auto_scnr(float(r.mean()), float(g.mean()), float(b.mean()))
    r, g, b = nebula(amount)
    y = (r + g + b) / 3                                      # saturation
    r, g, b = (y + SATURATION * (c - y) for c in (r, g, b))
    h, w = r.shape
    r, g, b = chroma_smooth(r, g, b, COLOUR_SMOOTH * w / 1000)
    k = float(np.median([chan[n]["k"] for n in assignment]))
    a0, a1, a2 = (star_stretch(k, chan[n]["sv"]) for n in assignment)
    lum = (a0 + a1 + a2) / 3
    wgt = STAR_COLOUR * np.where(lum >= STAR_COLOUR_KNEE, 1.0, lum / STAR_COLOUR_KNEE)
    out = [screen(np.clip(c, 0, 1), np.maximum(0, (lum + wgt * (a - lum)) * STAR_BRIGHTNESS)) for c, a in ((r, a0), (g, a1), (b, a2))]
    img = (np.clip(np.stack(out, axis=-1), 0, 1) * 255 + 0.5).astype(np.uint8)
    # Layers are stored bottom-up like Siril's stacks; flip so the picture is the right way up, as in the widget.
    return img[::-1], {"assignment": assignment, "scnr": round(amount, 2), "star_k": round(k, 1), "width": w, "height": h}
