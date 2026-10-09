"""Optional enhancement: split stars from the rest, process each layer, recombine.

    linear (aligned, background removed)
        |-- split --> starless + stars        (classical, or StarNet if configured)
        |      starless --> denoise  (--denoise)
        |      stars    --> deconvolve, clipped >= 0   (--deconvolve)
        |-- recombine: starless + stars   (or starless alone with --remove-stars)   -> linear+rl+dn/
        '-- final layers, kept apart (what the preview widget plays with)           -> final+rl+dn/

Settings (star FWHM, thresholds, iterations, blend amounts) are measured or chosen per target and
saved in <target>/settings.json; see settings.py.

Deconvolution is what rings (negative lobes around bright stars). Doing it on the stars layer and
clipping at zero means those lobes cannot survive recombination, and denoise never touches stars.
Without deconvolution nothing is clipped, so split + recombine returns the original exactly.
Layers are cached in <target>/layers; each recombination lands in its own variant folder
(linear+rl+dn, ...), so the main output and the widget's "assume it is on" data share all the work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import settings, siril, starsplit
from .finish import _stage
from .pystage import py_stage, read as pystage_read, write as pystage_write
from .stack import slug, stamp


@dataclass(frozen=True)
class Step:
    key: str        # short id used in folder names
    flag: str       # CLI flag
    label: str
    in_widget: bool  # the widget assumes this is on

    @property
    def attr(self) -> str:
        return self.flag.lstrip("-").replace("-", "_")


DECONVOLVE = Step("rl", "--deconvolve", "deconvolve the stars layer (Richardson-Lucy, PSF measured from the stars)", True)
DENOISE = Step("dn", "--denoise", "denoise the starless layer (non-local Bayesian)", True)
# Not in the widget: the widget has its own star tools and needs the stars in its data.
REMOVE_STARS = Step("sx", "--remove-stars", "output the starless layer only", False)
ORDER: tuple[Step, ...] = (DECONVOLVE, DENOISE, REMOVE_STARS)

SPLITTERS = ("classical", "starnet")

# Multiplicative RL with total-variation regularisation (the default gradient-descent step does almost nothing,
# and 25 iterations rings badly). The iteration count lives in settings.py: stars sharpen ~21% by 4 iterations,
# and every iteration after that only makes elongated stars into streaks.
def rl_commands(iterations: int = 4) -> tuple[str, ...]:
    return (f"rl -mul -iters={int(iterations)} -tv -alpha=1000",)


def default_cfg() -> dict:
    """Settings to use when none were resolved (tests, library use): the chosen defaults + a nominal FWHM."""
    return {**settings.DEFAULTS, "fwhm": starsplit.DEFAULT_FWHM}


def unavailable(splitter: str) -> Optional[str]:
    if splitter == "starnet" and siril.starnet_exe() is None:
        return "StarNet is not configured in Siril (Preferences > Miscellaneous)"
    return None


def normalise(steps: Sequence[Step]) -> tuple[Step, ...]:
    """Canonical order; deconvolving stars is pointless when the stars are discarded."""
    keys = {s.key for s in steps}
    if "sx" in keys:
        keys.discard("rl")
    return tuple(s for s in ORDER if s.key in keys)


def variant_name(steps: Sequence[Step]) -> str:
    return "linear" if not steps else "linear+" + "+".join(s.key for s in steps)


# ---- Siril scripts for single layers -------------------------------------------------

def _script(flt: str, body: Sequence[str], out: Optional[str]) -> str:
    save = [f"save {out}"] if out else []
    return "\n".join([f"requires {siril.MIN_VERSION}", f"load in_{slug(flt)}", *body, *save, "close"]) + "\n"


def starnet_script(flt: str) -> str:
    return _script(flt, ["starnet -stretch -nostarmask"], f"starless_{slug(flt)}")


def denoise_script(flt: str) -> str:
    # Default denoise also applies cosmetic correction, which helps since hot pixels are not calibrated out.
    return _script(flt, ["denoise"], f"starless_dn_{slug(flt)}")


def psf_script(flt: str) -> str:
    return _script(flt, ["makepsf stars -sym -savepsf=psf_%s.fit" % slug(flt)], None)


def deconvolve_script(flt: str, iterations: int = 4) -> str:
    return _script(flt, ["makepsf load psf_%s.fit" % slug(flt), *rl_commands(iterations)], f"stars_rl_{slug(flt)}")


# ---- Python stages -------------------------------------------------------------------

# in-process stage helpers live in pystage.py; the old names stay so call sites and tests are unchanged
_py_stage, _read, _write = py_stage, pystage_read, pystage_write


def split_layers(tdir: Path, names: list[str], splitter: str, siril_bin, dry_run, keep, cfg: Optional[dict] = None) -> Path:
    cfg = cfg or default_cfg()
    layers, base = tdir / "layers", tdir / "linear"
    lin = [base / f"linear_{slug(n)}.fit" for n in names]
    starless = [f"starless_{slug(n)}.fit" for n in names]
    stars = [f"stars_{slug(n)}.fit" for n in names] + [f"stars_pos_{slug(n)}.fit" for n in names]
    if splitter == "starnet":
        for n, src in zip(names, lin):
            _stage(layers, f"sn_{slug(n)}", starnet_script(n), [src],
                   lambda w, src=src, n=n: (w / f"in_{slug(n)}.fit").symlink_to(src),
                   [f"starless_{slug(n)}.fit"], siril_bin, dry_run, keep)

        def stars_from_starless():
            for n, src in zip(names, lin):
                d = _read(src) - _read(layers / f"starless_{slug(n)}.fit")
                _write(layers / f"stars_{slug(n)}.fit", d, src)
                _write(layers / f"stars_pos_{slug(n)}.fit", np.maximum(d, 0), src)   # RL input must not be negative
        _py_stage(layers, "stars", lin + [layers / o for o in starless], "starnet", stars, stars_from_starless, dry_run)
    else:
        def classical():
            data = {n: _read(src) for n, src in zip(names, lin)}
            _, out = starsplit.split(data, fwhm=cfg["fwhm"], thresh=cfg["star_threshold"])
            for n, src in zip(names, lin):
                _write(layers / f"starless_{slug(n)}.fit", out[n][0], src)
                _write(layers / f"stars_{slug(n)}.fit", out[n][1], src)
                _write(layers / f"stars_pos_{slug(n)}.fit", np.maximum(out[n][1], 0), src)
        _py_stage(layers, "split", lin, f"classical-union-protect:{cfg['fwhm']}:{cfg['star_threshold']}:{starsplit.MEDIUM}:{starsplit.BRIGHT}:{starsplit.PROTECT_LO}:{starsplit.PROTECT_HI}",
                  starless + stars, classical, dry_run)
    return layers


def _gate(stars: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-threshold: shrink toward zero by `threshold`, so noise-level values vanish and bright ones barely change."""
    return np.sign(stars) * np.maximum(np.abs(stars) - threshold, 0)


def _final_layers(tdir: Path, n: str, keys: set, cfg: dict, noise: dict) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """The processed (starless, stars) pair for one filter. stars is None when the stars are discarded."""
    layers, base = tdir / "layers", tdir / "linear"
    plain = _read(layers / f"starless_{slug(n)}.fit")
    starless = plain
    if "dn" in keys:
        amount = cfg["denoise_amount"]
        starless = amount * _read(layers / f"starless_dn_{slug(n)}.fit") + (1 - amount) * plain
    if "sx" in keys:
        return starless, None
    stars = _read(layers / (f"stars_rl_{slug(n)}.fit" if "rl" in keys else f"stars_{slug(n)}.fit"))
    if keys & {"dn", "rl"}:
        # Where stars were masked, the stars layer still carries the sky's own noise (random per filter).
        # Next to a denoised starless layer that shows up as coloured speckle at every former star, so
        # shrink anything within a few sigma of zero (real stars keep nearly all their flux).
        if n not in noise:
            noise[n] = starsplit.noise_sigma(_read(base / f"linear_{slug(n)}.fit"))
        stars = _gate(stars, cfg["star_gate"] * noise[n])
    if "rl" in keys:
        stars = np.maximum(stars, 0)   # deconvolution's negative lobes (the dark rings) must not come back
    return starless, stars


def _layer_files(tdir: Path, names: list[str], keys: set) -> list[Path]:
    layers = tdir / "layers"
    files = [layers / (f"starless_dn_{slug(n)}.fit" if "dn" in keys else f"starless_{slug(n)}.fit") for n in names]
    if "dn" in keys:
        files += [layers / f"starless_{slug(n)}.fit" for n in names]
    if "sx" not in keys:
        files += [layers / (f"stars_rl_{slug(n)}.fit" if "rl" in keys else f"stars_{slug(n)}.fit") for n in names]
    return files


def _cfg_tag(keys: set, cfg: dict) -> str:
    return "+".join(sorted(keys)) + f":{cfg['denoise_amount']}:{cfg['star_gate']}"


def recombine(tdir: Path, names: list[str], steps: Sequence[Step], dry_run, cfg: Optional[dict] = None) -> Path:
    """linear+<steps>/: starless + stars added back together (the main output)."""
    cfg = cfg or default_cfg()
    dest, keys = tdir / variant_name(steps), {s.key for s in steps}

    def combine():
        noise: dict = {}
        for n in names:
            starless, stars = _final_layers(tdir, n, keys, cfg, noise)
            _write(dest / f"linear_{slug(n)}.fit", starless if stars is None else starless + stars,
                   tdir / "linear" / f"linear_{slug(n)}.fit")
    _py_stage(dest, "recombine", _layer_files(tdir, names, keys), _cfg_tag(keys, cfg),
              [f"linear_{slug(n)}.fit" for n in names], combine, dry_run)
    return dest


def final_layers(tdir: Path, names: list[str], steps: Sequence[Step], dry_run, cfg: Optional[dict] = None) -> Path:
    """final+<steps>/: starless_<F>.fit and stars_<F>.fit kept apart, for the preview widget."""
    cfg = cfg or default_cfg()
    dest, keys = tdir / ("final" + variant_name(steps)[len("linear"):]), {s.key for s in steps}

    def write():
        noise: dict = {}
        for n in names:
            starless, stars = _final_layers(tdir, n, keys, cfg, noise)
            like = tdir / "linear" / f"linear_{slug(n)}.fit"
            _write(dest / f"starless_{slug(n)}.fit", starless, like)
            _write(dest / f"stars_{slug(n)}.fit", np.zeros_like(starless) if stars is None else stars, like)
    _py_stage(dest, "final", _layer_files(tdir, names, keys), _cfg_tag(keys, cfg),
              [f"{k}_{slug(n)}.fit" for n in names for k in ("starless", "stars")], write, dry_run)
    return dest


def _process_layers(tdir: Path, names: list[str], steps: Sequence[Step], splitter: str, siril_bin,
                    dry_run: bool, keep: bool, cfg: dict) -> None:
    """Split, then denoise the starless layer / deconvolve the stars layer as the steps ask."""
    layers = split_layers(tdir, names, splitter, siril_bin, dry_run, keep, cfg)
    keys = {s.key for s in steps}
    base = tdir / "linear"
    if "dn" in keys:
        for n in names:
            src = layers / f"starless_{slug(n)}.fit"
            _stage(layers, f"dn_{slug(n)}", denoise_script(n), [src],
                   lambda w, src=src, n=n: (w / f"in_{slug(n)}.fit").symlink_to(src),
                   [f"starless_dn_{slug(n)}.fit"], siril_bin, dry_run, keep)
    if "rl" in keys:
        for n in names:
            lin, stars = base / f"linear_{slug(n)}.fit", layers / f"stars_pos_{slug(n)}.fit"
            _stage(layers, f"psf_{slug(n)}", psf_script(n), [lin],          # PSF from the real stars, not the stars layer
                   lambda w, lin=lin, n=n: (w / f"in_{slug(n)}.fit").symlink_to(lin),
                   [f"psf_{slug(n)}.fit"], siril_bin, dry_run, keep)
            psf = layers / f"psf_{slug(n)}.fit"

            def link(w, stars=stars, psf=psf, n=n):
                (w / f"in_{slug(n)}.fit").symlink_to(stars)
                (w / f"psf_{slug(n)}.fit").symlink_to(psf)
            _stage(layers, f"rl_{slug(n)}", deconvolve_script(n, cfg["rl_iterations"]), [stars, psf], link,
                   [f"stars_rl_{slug(n)}.fit"], siril_bin, dry_run, keep)


def build_variant(tdir: Path, names: list[str], steps: Sequence[Step], splitter: str,
                  siril_bin: Optional[Path], dry_run: bool = False, keep: bool = False,
                  cfg: Optional[dict] = None) -> Path:
    """Folder holding linear_<F>.fit for this combination of steps (the plain stacks if there are none)."""
    steps = normalise(steps)
    if not steps:
        return tdir / "linear"
    cfg = cfg or default_cfg()
    _process_layers(tdir, names, steps, splitter, siril_bin, dry_run, keep, cfg)
    return recombine(tdir, names, steps, dry_run, cfg)


def build_layers(tdir: Path, names: list[str], steps: Sequence[Step], splitter: str,
                 siril_bin: Optional[Path], dry_run: bool = False, keep: bool = False,
                 cfg: Optional[dict] = None) -> Path:
    """Folder holding starless_<F>.fit and stars_<F>.fit for this combination of steps. Always splits."""
    steps = normalise(steps)
    cfg = cfg or default_cfg()
    _process_layers(tdir, names, steps, splitter, siril_bin, dry_run, keep, cfg)
    return final_layers(tdir, names, steps, dry_run, cfg)
