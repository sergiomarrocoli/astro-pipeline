"""Sky background (gradient) removal on the aligned stacks.

The image is cut into blocks and each block is reduced to its median. A low-order surface (a plane, or a
quadratic when the data supports it) is fitted to the darkest part of the image only: at each pass the brightest
blocks are thrown away, so the fit follows the lower envelope, which is the sky, and never the nebula. Blocks
that are textured (stars, nebula edges) are rejected first.

This is deliberately conservative. Eating real nebulosity is worse than leaving a little gradient, so the fit
never models a bright region. The price is that it cannot remove a glow, since a glow is also just a bright
region; it removes tilt and gentle curvature of the sky, and the small differences in sky level between filters
that turn into a colour cast once the image is stretched hard. The global sky level is preserved.
"""
from __future__ import annotations

import numpy as np

from .starsplit import noise_sigma

DEFAULT_DEGREE = 2    # 0 switches the correction off; 2 falls back to a plane when the clean blocks do not cover the frame
BLOCK = 96            # px; large enough that a block's median ignores stars
KEEP_FRACTION = 0.6   # fit to the darkest 60% of blocks
LOW_CLIP = 3.0        # also drop blocks this many sigma below the fit (dead areas)
TEXTURE = 1.8         # reject blocks whose own scatter is this much above typical (stars, edges)
LARGE = 8.0           # a correction bigger than this many noise sigma is flagged: it may be nebulosity, not sky


def _terms(degree: int) -> list[tuple[int, int]]:
    return [(i, t - i) for t in range(degree + 1) for i in range(t + 1)]


def _block_medians(a: np.ndarray, block: int):
    h, w = a.shape
    gy, gx = h // block, w // block
    y0, x0 = (h - gy * block) // 2, (w - gx * block) // 2          # centre the grid on the frame
    b = a[y0:y0 + gy * block, x0:x0 + gx * block].reshape(gy, block, gx, block).transpose(0, 2, 1, 3).reshape(gy, gx, -1)
    med = np.median(b, axis=2)
    mad = np.median(np.abs(b - med[:, :, None]), axis=2)
    cy = y0 + (np.arange(gy) + 0.5) * block
    cx = x0 + (np.arange(gx) + 0.5) * block
    return med, mad, cy, cx


def fit(a: np.ndarray, degree: int = DEFAULT_DEGREE, block: int = BLOCK, iters: int = 12):
    """Fit the sky surface to the darkest blocks. Returns (parameters, info) or (None, info) if it cannot be done safely."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    h, w = a.shape
    med, mad, cy, cx = _block_medians(a, block)
    gy, gx = med.shape
    cyy, cxx = np.meshgrid(cy, cx, indexing="ij")
    u, v = (cxx.ravel() - w / 2) / (w / 2), (cyy.ravel() - h / 2) / (h / 2)
    X = np.stack([u ** i * v ** j for i, j in _terms(degree)], axis=1)
    y = med.ravel()
    ok = (mad.ravel() <= TEXTURE * np.median(mad)) | (mad.ravel() <= 0)
    keep = ok.copy()
    deg_used, coef = degree, None
    for _ in range(iters):
        # A quadratic extrapolates wildly where it has no data: only trust it when every quadrant has clean blocks.
        kq = keep.reshape(gy, gx)
        covered = all(kq[ya:yb, xa:xb].mean() >= 0.2 for ya, yb in ((0, gy // 2), (gy // 2, gy)) for xa, xb in ((0, gx // 2), (gx // 2, gx)))
        deg_used = degree if covered else min(degree, 1)
        n = len(_terms(deg_used))
        if keep.sum() < 3 * n:
            return None, {"reason": "too few clean blocks", "kept": int(keep.sum()), "blocks": int(y.size)}
        coef = np.linalg.lstsq(X[keep, :n], y[keep], rcond=None)[0]
        r = y - X[:, :n] @ coef
        cut = np.quantile(r[ok], KEEP_FRACTION)                    # the darkest share of all candidate blocks
        mid = np.median(r[ok][r[ok] <= cut])
        sig = max(1.4826 * np.median(np.abs(r[ok][r[ok] <= cut] - mid)), 1e-4 * max(float(np.abs(y).max()), 1.0))
        new = ok & (r <= cut) & (r > -LOW_CLIP * sig - 0.0)
        if (new == keep).all():
            break
        keep = new
    rr = (y - X[:, :n] @ coef)[keep]
    return {"coef": coef, "degree": deg_used}, {"kept": int(keep.sum()), "blocks": int(y.size), "residual_sigma": float(rr.std()), "degree": deg_used}


def evaluate(shape: tuple[int, int], params: dict) -> np.ndarray:
    """The fitted surface at every pixel (separable powers, so it stays cheap at full resolution)."""
    h, w = shape
    degree = params["degree"]
    u = (np.arange(w, dtype=np.float32) + 0.5 - w / 2) / (w / 2)
    v = (np.arange(h, dtype=np.float32) + 0.5 - h / 2) / (h / 2)
    up = [u ** i for i in range(degree + 1)]
    vp = [v ** j for j in range(degree + 1)]
    out = np.zeros(shape, np.float32)
    for c, (i, j) in zip(params["coef"], _terms(degree)):
        out += np.float32(c) * vp[j][:, None] * up[i][None, :]
    return out


def remove(a: np.ndarray, degree: int = DEFAULT_DEGREE, block: int = BLOCK):
    """Subtract the sky gradient, keeping the overall level. Returns (corrected image, info)."""
    a = np.asarray(a, dtype=np.float32)
    if degree <= 0:
        return a.copy(), {"applied": False, "reason": "off"}
    params, info = fit(a, degree, block)
    if params is None:
        return a.copy(), {"applied": False, **info}
    model = evaluate(a.shape, params)
    model -= np.median(model[::7, ::7])                            # variation only: the sky level stays where it was
    noise = float(noise_sigma(a))                                  # pixel noise: the scatter of the image minus a gentle blur
    over = float((model.max() - model.min()) / max(noise, 1e-9))
    info.update(applied=True, amplitude=float(model.max() - model.min()), amplitude_over_noise=over, large=over > LARGE)
    return a - model, info


def stage(tdir, names: list[str], degree: int, dry_run: bool = False):
    """aligned/aligned_<F>.fit -> linear/linear_<F>.fit with the sky gradient of each filter removed.

    Each filter is fitted on its own, which also removes small differences in sky level between filters; those
    are what turn into a colour cast at the corners once the image is stretched hard.
    """
    import json

    from .pystage import py_stage, read, write
    from .stack import slug

    aligned, linear = tdir / "aligned", tdir / "linear"
    src = [aligned / f"aligned_{slug(n)}.fit" for n in names]
    report: dict = {}

    def run():
        for n, p in zip(names, src):
            out, info = remove(read(p), degree, BLOCK)
            write(linear / f"linear_{slug(n)}.fit", out, p)
            report[n] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()}
            if info.get("large"):
                print(f"warning: {tdir.name} / {n}: the sky correction is large ({info['amplitude_over_noise']:.0f} noise sigma). "
                      f"If this field is filled with smooth nebulosity it may be eating signal; set "
                      f"'background_degree': 0 under override in {tdir.name}/settings.json to switch it off.")
        (tdir / "background.json").write_text(json.dumps(report, indent=1))

    return py_stage(linear, "background", src, f"{degree}:{BLOCK}:{KEEP_FRACTION}:{LOW_CLIP}:{TEXTURE}",
                    [f"linear_{slug(n)}.fit" for n in names], run, dry_run)
