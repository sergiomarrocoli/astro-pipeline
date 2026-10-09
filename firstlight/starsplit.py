"""Classical star / starless split of aligned linear images (no StarNet needed).

Same idea as the widget's star tools: find compact peaks, replace them with a smooth estimate of
the background behind them (normalised convolution), and call the difference the stars layer.
Sizes are in pixels and scale with the star FWHM, so this works at full resolution.
"""
from __future__ import annotations

import numpy as np

DEFAULT_FWHM = 2.5     # px, typical for this setup at full resolution
PROTECT_LO = 25.0       # nebulosity this many noise sigma above the sky starts to protect a star from removal ...
PROTECT_HI = 40.0       # ... and fully protects it here
FULL_AT = 0.7          # blurred-mask value at which a pixel counts as fully masked
MEDIUM = 5.0           # response multiples (of the threshold) where the mask tiers step up
BRIGHT = 25.0
DEFAULT_THRESH = 8.0   # multiples of the band-pass noise: clears faint stars without eating nebula detail (~16% coverage on a very dense field)


def gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via FFT with reflected edges."""
    if sigma < 0.05:
        return np.array(a, dtype=np.float32)
    pad = int(np.ceil(4 * sigma)) + 1
    ap = np.pad(np.asarray(a, dtype=np.float32), pad, mode="reflect")
    fy = np.fft.fftfreq(ap.shape[0]).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(ap.shape[1]).astype(np.float32)[None, :]
    g = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2)).astype(np.float32)
    out = np.fft.irfft2(np.fft.rfft2(ap) * g, s=ap.shape)
    return out[pad:-pad, pad:-pad].astype(np.float32)


def robust_sigma(a: np.ndarray, stride: int = 7) -> float:
    s = np.asarray(a).ravel()[::stride]
    return float(1.4826 * np.median(np.abs(s - np.median(s)))) or 1e-6


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r pixels (r repeated 3x3 steps)."""
    m = mask
    for _ in range(int(r)):
        p = np.pad(m, 1)
        m = np.maximum.reduce([p[dy:dy + m.shape[0], dx:dx + m.shape[1]] for dy in range(3) for dx in range(3)])
    return m


def luminance(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the background-subtracted channels, each in units of its own noise."""
    lum = None
    for a in channels.values():
        a = np.nan_to_num(np.asarray(a, dtype=np.float32))
        bg = np.median(a[::7, ::7])
        n = (a - bg) / robust_sigma(a - bg)
        lum = n if lum is None else lum + n
    return lum / len(channels)


def star_response(lum: np.ndarray, fwhm: float = DEFAULT_FWHM) -> np.ndarray:
    """Band-pass tuned to star size: smooth a little (matched to the PSF) minus smooth a lot.

    Compared with a plain top-hat against a very wide blur, this ignores fine nebula structure
    (knots and filaments are bigger than a star) while still responding to faint, compact stars.
    """
    return gblur(lum, 0.4 * fwhm) - gblur(lum, 1.6 * fwhm)


def star_mask(lum: np.ndarray, fwhm: float = DEFAULT_FWHM, thresh: float = DEFAULT_THRESH) -> np.ndarray:
    """Soft mask in [0,1] with the mask size growing with the star's brightness.

    Three tiers (response above thresh, MEDIUM*thresh, BRIGHT*thresh) are grown by 0.8, 1.6 and 4 FWHM:
    a faint star needs a small hole, a bright one has wings that reach far beyond where it first clears
    the threshold. Growing every star equally either leaves bright halos or erases the nebula.
    Only compact features count (see `compact`), so nebula knots are not mistaken for stars.
    """
    r = star_response(lum, fwhm)
    cut = thresh * robust_sigma(r)
    # Size filter: a star responds most at star scale; a nebula knot responds more one scale up.
    compact = r > (gblur(lum, 1.6 * fwhm) - gblur(lum, 6.4 * fwhm))
    tiers = [(1.0, 0.8), (MEDIUM, 1.6), (BRIGHT, 4.0)]
    grown = np.zeros_like(r)
    for mult, grow in tiers:
        grown = np.maximum(grown, dilate(((r > mult * cut) & compact).astype(np.float32), round(grow * fwhm)))
    soft = gblur(grown, 1.2 * fwhm)
    # Blurring a small mask lowers its centre (0.87 for a modest star), which would let 13% of the star
    # leak through. A smoothstep makes anything above ~0.7 fully masked and keeps a soft edge below it.
    t = np.clip(soft / FULL_AT, 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def inpaint(a: np.ndarray, mask: np.ndarray, r: float, _cache: dict | None = None) -> np.ndarray:
    """Replace masked pixels with blur(a*(1-M))/blur(1-M); fall back to radius 3r where holes are big."""
    inv = 1 - mask
    c = _cache if _cache is not None else {}
    if "w1" not in c:
        c["w1"], c["w3"] = gblur(inv, r), gblur(inv, 3 * r)
    w1, w3 = c["w1"], c["w3"]
    am = a * inv
    n1, n3 = gblur(am, r), gblur(am, 3 * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(w1 > 0.15, n1 / w1, np.where(w3 > 1e-3, n3 / w3, a))
    return (a * inv + mask * fill).astype(np.float32)


def noise_sigma(a: np.ndarray) -> float:
    """Sky noise level: robust sigma of the image minus a gentle blur."""
    return robust_sigma(a - gblur(a, 3.0))


def grain(like: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """White noise at the sky's noise level, for filling masked regions.

    Tried and rejected: blurring it to match the stack's measured pixel correlation (lag-1 ~0.7).
    It made worm-like texture in the fill and left marks after denoise; plain white noise blends in and
    denoises cleanly. Leaving the fill noise-free gives flat patches that denoise turns into bright blobs.
    """
    return rng.normal(0.0, noise_sigma(like), like.shape).astype(np.float32)


def _fill_all(chans: dict[str, np.ndarray], mask: np.ndarray, fwhm: float) -> dict[str, np.ndarray]:
    cache: dict = {}
    return {n: inpaint(a, mask, 4.7 * fwhm, cache) for n, a in chans.items()}


def split(channels: dict[str, np.ndarray], fwhm: float = DEFAULT_FWHM, thresh: float = DEFAULT_THRESH):
    """channels: name -> aligned linear image. Returns (mask, {name: (starless, stars)}).

    One mask is shared by all channels (the union of the per-filter detections), so stars line up in
    every filter. stars = original - starless exactly, so starless + stars is the original (it can be
    slightly negative in the noise; clip it where that matters, e.g. before deconvolution). Stars on very
    bright nebulosity are left in the starless layer (see PROTECT_LO).
    """
    chans = {n: np.nan_to_num(np.asarray(a, dtype=np.float32)) for n, a in channels.items()}
    # Detect in each filter on its own and take the union. Averaging the filters first would hide a star
    # that is strong in only one of them (a red star in S, say), leaving it behind as a coloured dot.
    mask = np.maximum.reduce([star_mask(luminance({n: a}), fwhm, thresh) for n, a in chans.items()])
    rng = np.random.default_rng(0)  # fixed seed: same input, same output, so stamps stay valid
    bgs = {n: float(np.median(a[::7, ::7])) for n, a in chans.items()}
    sgs = {n: noise_sigma(a) for n, a in chans.items()}
    fills = _fill_all(chans, mask, fwhm)
    # Where the nebulosity under a star is very bright, a flat fill cannot know the peak it hides and leaves a
    # dimple (10-60 sigma on the test night, about 1 sigma elsewhere). Rather than invent a hole, leave the star
    # in the starless layer there, as real star-removal tools do in bright cores. Starless + stars is still exact.
    level = np.max([(fills[n] - bgs[n]) / max(sgs[n], 1e-9) for n in chans], axis=0)
    t = np.clip((level - PROTECT_LO) / (PROTECT_HI - PROTECT_LO), 0.0, 1.0)
    protect = t * t * (3 - 2 * t)
    if ((mask * protect) > 0.02).any():
        mask = (mask * (1 - protect)).astype(np.float32)
        fills = _fill_all(chans, mask, fwhm)
    out = {}
    for n, a in chans.items():
        # A blurred fill is noise-free and shows up as flat patches; give it the sky's own grain.
        starless = (fills[n] + mask * grain(a, rng)).astype(np.float32)
        out[n] = (starless, (a - starless).astype(np.float32))
    return mask, out
