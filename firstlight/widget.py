"""Preview widget: a self-contained HTML page that composites the linear stacks live.

Python downsamples each linear stack and computes robust stats; the page does the
normalisation and stretch in JS so sliders respond instantly. The embedded overview is block-averaged;
full-resolution tiles sit beside the page in widget_tiles/ and load lazily when zoomed in. Band assignment is fixed
by default (SHO, or HOO for two filters) but can be reassigned in the page.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .finish import composite_channels, role
from .stack import slug, stamp

TEMPLATE = Path(__file__).with_name("widget_template.html")
DSP = Path(__file__).with_name("widget_dsp.js")
TARGET_WIDTH = 1000
EDGE = 4          # px cropped from every border of the layers before the widget is built
TILE_SIZE = 512  # power of two: the page finds a tile with bit shifts


def _factor(shape, width: int = TARGET_WIDTH) -> int:
    return max(1, round(max(shape) / width))


def downsample(a: np.ndarray, width: int = TARGET_WIDTH) -> np.ndarray:
    """Integer block-mean downsample so the longest side is about `width` px."""
    f = _factor(a.shape, width)
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def quantise(a: np.ndarray, vmax: float) -> np.ndarray:
    """Linear values -> little-endian uint16 where 65535 == vmax."""
    return np.round(np.clip(np.nan_to_num(np.asarray(a, dtype=np.float64)) / vmax, 0.0, 1.0) * 65535).astype("<u2")


def channel_pack(a: np.ndarray) -> dict:
    """Quantise a linear channel to uint16 and return it with stats in the same units [0, 1]."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    vmax = float(np.percentile(a, 99.99))
    vmax = vmax if vmax > 0 else 1.0
    v = np.clip(a / vmax, 0.0, 1.0)
    bg = float(np.median(v))
    sigma = float(1.4826 * np.median(np.abs(v - bg))) or 1e-6
    p99 = float(np.percentile(v - bg, 99))
    return {"bg": bg, "sigma": sigma, "p99": max(p99, 1e-6), "vmax": vmax,
            "data": base64.b64encode(quantise(a, vmax).tobytes()).decode("ascii")}


def write_tiles(full: dict[str, np.ndarray], vmax: dict[str, float], tdir: Path, size: int = TILE_SIZE) -> dict:
    """Write full-resolution tiles as <script>-loadable .js files (works from file://).

    `full` arrays must already be top-down and cropped to match the overview. Each tile file
    holds every channel, quantised with the *overview's* vmax so stats stay comparable.
    """
    h, w = next(iter(full.values())).shape
    rows, cols = -(-h // size), -(-w // size)
    tdir.mkdir(parents=True, exist_ok=True)
    for old in tdir.glob("t_*.js"):
        old.unlink()
    for ty in range(rows):
        for tx in range(cols):
            y0, x0 = ty * size, tx * size
            y1, x1 = min(h, y0 + size), min(w, x0 + size)
            obj = {n: base64.b64encode(quantise(a[y0:y1, x0:x1], vmax[n]).tobytes()).decode("ascii")
                   for n, a in full.items()}
            (tdir / f"t_{ty}_{tx}.js").write_text(f"__tile({ty},{tx},{x1 - x0},{y1 - y0},{json.dumps(obj)});\n")
    return {"dir": tdir.name, "size": size, "shift": size.bit_length() - 1, "w": w, "h": h,
            "cols": cols, "rows": rows}


def default_assignment(names: list[str]) -> list[str]:
    """Filter names for (R, G, B): SHO / HOO when recognisable, else cycle through the filters."""
    by_role = {role(n): n for n in names if role(n)}
    palette = composite_channels(set(by_role))
    if palette:
        return [by_role[r] for r in palette[1]]
    if {"H", "S"} <= set(by_role):                      # no O: an H/S bicolour (H red, S cyan) rather than a guess
        return [by_role["H"], by_role["S"], by_role["S"]]
    ordered = sorted(names)
    return [ordered[i % len(ordered)] for i in range(3)]


def star_stretch_k(stars_norm: np.ndarray, target: float = 0.7) -> float:
    """asinh strength k that lifts moderately bright stars well, without lifting the faint residue.

    The 95th percentile of the star pixels (not the median, which is mostly faint noise left in masked
    areas) is mapped to `target` brightness.
    """
    v = stars_norm[stars_norm > 0.01]
    q = float(np.percentile(v, 95)) if v.size else 0.1
    lo, hi = 0.5, 5000.0
    for _ in range(60):                       # bisection: asinh(k q)/asinh(k) rises with k
        k = (lo * hi) ** 0.5
        if np.arcsinh(k * q) / np.arcsinh(k) < target:
            lo = k
        else:
            hi = k
    return float(k)


def build_payload(layers: dict[str, tuple[np.ndarray, np.ndarray]], tile_dir: Optional[Path] = None,
                  width: int = TARGET_WIDTH, tile_size: int = TILE_SIZE, info: Optional[str] = None,
                  noise: Optional[dict[str, float]] = None) -> dict:
    """layers: filter name -> (starless, stars) linear arrays. Returns the JSON blob the page consumes.

    With `tile_dir`, also writes full-resolution tiles (both layers) there and describes them under "tiles".
    `noise` (filter -> sky noise of the plain stack, in the layers' units) sets the page's black point; without
    it the noise is measured on the (possibly denoised) starless layer, which is smaller.
    """
    first = next(iter(layers.values()))[0]
    f = _factor(first.shape, width)
    small = {n: (downsample(sl, width), downsample(st, width)) for n, (sl, st) in layers.items()}
    h, w = small[next(iter(small))][0].shape
    channels, vmax = [], {}
    for n in sorted(small):
        sl, st = small[n]
        pack = channel_pack(sl)
        if noise and n in noise:
            pack["sigma"] = max(noise[n] / pack["vmax"], 1e-9)   # the page works in units of vmax
        svmax = float(np.percentile(st, 99.99)) or 1.0
        svmax = svmax if svmax > 0 else 1.0
        vmax[n], vmax[n + "*"] = pack["vmax"], svmax
        channels.append({"name": n, **{k: v for k, v in pack.items() if k != "vmax"},
                         "sdata": base64.b64encode(quantise(st, svmax).tobytes()).decode("ascii"),
                         "k": star_stretch_k(np.clip(st / svmax, 0, 1))})
    payload = {"width": w, "height": h, "factor": f, "channels": channels,
               "default": default_assignment(list(small)), "tiles": None, "info": info}
    if tile_dir is not None:
        # Crop to exactly what the overview covers, then flip so tile rows run top-down like the canvas.
        full = {}
        for n, (sl, st) in layers.items():
            full[n] = np.flipud(np.asarray(sl)[:h * f, :w * f])
            full[n + "*"] = np.flipud(np.asarray(st)[:h * f, :w * f])
        payload["tiles"] = write_tiles(full, vmax, tile_dir, tile_size)
    return payload


def render_html(payload: dict) -> str:
    # "</" inside JSON would end the script tag early; base64 and numbers can't contain it.
    html = TEMPLATE.read_text().replace("/*__DSP__*/", DSP.read_text())
    return html.replace("__PAYLOAD__", json.dumps(payload))


def write_widget(target: str, layers_dir: Path, names: list[str], out: Path, dry_run=False,
                 auto: Optional[dict] = None) -> Optional[Path]:
    """Build the widget from <layers_dir>/starless_<F>.fit and stars_<F>.fit."""
    from astropy.io import fits

    if len(names) < 2:
        print(f"{target}: widget needs at least two filters, skipping")
        return None
    dest = out / slug(target) / "preview" / "widget.html"
    if dry_run:
        print(f"--- {target}/widget: would write {dest}")
        return None
    inputs = [layers_dir / f"{k}_{slug(n)}.fit" for n in names for k in ("starless", "stars")]
    sp = dest.with_name(".widget.stamp")
    info = None
    if auto:
        info = (f"star FWHM {auto.get('fwhm')} px · denoise {round(100 * auto.get('denoise_amount', 0))}% · "
                f"deconvolution {auto.get('rl_iterations')} iterations")
    new = stamp([{"path": str(p)} for p in inputs],
                TEMPLATE.read_text() + DSP.read_text() + f"{TARGET_WIDTH}:{TILE_SIZE}:{EDGE}:{info}:{(auto or {}).get('noise')}")
    tdir = dest.parent / "widget_tiles"
    if dest.exists() and tdir.is_dir() and sp.exists() and sp.read_text() == new:
        print(f"{target} / widget: up to date, skipping")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    e = EDGE   # registration leaves a few unreliable pixels along the border
    layers = {n: (fits.getdata(layers_dir / f"starless_{slug(n)}.fit")[e:-e, e:-e],
                  fits.getdata(layers_dir / f"stars_{slug(n)}.fit")[e:-e, e:-e]) for n in names}
    dest.write_text(render_html(build_payload(layers, tile_dir=tdir, info=info, noise=(auto or {}).get("noise"))))
    sp.write_text(new)
    print(f"{target} / widget: {dest}")
    return dest
