"""Find the sensor's persistent hot pixels from the lights themselves (no darks needed) and correct them.

A hot pixel belongs to the sensor, so it sits at the same pixel in every field, while stars move with the
pointing. Per field we take the median of several frames (cosmic rays vanish) and flag isolated spikes: a
pixel far above the sky whose eight neighbours are not raised. A pixel flagged in at least two fields is a
sensor defect; with only one field a stricter isolation rule is used, so an undersampled star is not mistaken
for one. The list is written in Siril's cosmetic-correction format and applied before registration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import starsplit

K_SIGMA = 10.0            # a spike must clear this many sigma of sky noise
MAX_RING = 0.30           # neighbours may carry at most this fraction of the spike (several fields)
MAX_RING_SINGLE = 0.15    # ... and this fraction when only one field is available
BORDER = 3                # px ignored at the frame edge
FRAMES_PER_FIELD = 9      # evenly spaced through the night


def _ring_mean(a: np.ndarray) -> np.ndarray:
    """Mean of the 8 neighbours of every pixel (edges wrap, which is harmless: the border is ignored)."""
    s = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                s += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    return s / 8.0


def candidates(med: np.ndarray, max_ring: float = MAX_RING, k: float = K_SIGMA) -> np.ndarray:
    """Boolean mask of isolated spikes in a (median) image."""
    med = np.nan_to_num(np.asarray(med, dtype=np.float32))
    bg, sg = float(np.median(med[::7, ::7])), starsplit.noise_sigma(med)
    excess = med - bg
    ring = _ring_mean(med) - bg
    mask = (excess > k * sg) & (ring < max_ring * excess)
    mask[:BORDER] = mask[-BORDER:] = False
    mask[:, :BORDER] = mask[:, -BORDER:] = False
    return mask


def median_image(paths: list[Path], limit: int = FRAMES_PER_FIELD) -> np.ndarray:
    from astropy.io import fits

    if len(paths) > limit:
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, limit)]
    return np.median(np.stack([fits.getdata(p).astype(np.float32) for p in paths]), axis=0)


def find(fields: dict[str, list[Path]]) -> tuple[np.ndarray, dict]:
    """fields: field (target) name -> its light frames, all with the same exposure.

    Returns (mask of stable hot pixels, info). Fields with fewer than 3 frames are ignored.
    """
    usable = {n: p for n, p in fields.items() if len(p) >= 3}
    if not usable:
        return np.zeros((0, 0), bool), {"fields": [], "count": 0, "note": "need at least 3 frames of one field"}
    masks = []
    for name, paths in usable.items():
        masks.append(candidates(median_image(paths), MAX_RING if len(usable) > 1 else MAX_RING_SINGLE))
    need = 2 if len(masks) > 1 else 1
    mask = np.sum(masks, axis=0) >= need
    return mask, {"fields": sorted(usable), "count": int(mask.sum()), "per_field": [int(m.sum()) for m in masks],
                  "rule": f"flagged in >= {need} of {len(masks)} field(s)"}


def row_order_top_down(path: Path) -> bool:
    """Frames stored top-down (NINA writes ROWORDER=TOP-DOWN) have row 0 at the top; FITS default is bottom-up."""
    from astropy.io import fits

    return str(fits.getheader(path).get("ROWORDER", "BOTTOM-UP")).strip().upper() == "TOP-DOWN"


def to_list(mask: np.ndarray, top_down: bool) -> str:
    """Siril cosmetic-correction list: 'P x y H', 0-based, x = column, y counted from the bottom."""
    h = mask.shape[0]
    ys, xs = np.nonzero(mask)
    lines = [f"P {int(x)} {int(h - 1 - y) if top_down else int(y)} H" for y, x in zip(ys, xs)]
    return "\n".join(lines) + ("\n" if lines else "")


def apply(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace flagged pixels with the mean of their unflagged neighbours (for frames handled outside Siril)."""
    out = np.array(data, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    h, w = out.shape
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        patch, m = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        keep = ~m
        if keep.any():
            out[y, x] = patch[keep].mean()
    return out


def read_list(path: Path, shape: tuple[int, int], top_down: bool) -> np.ndarray:
    """Parse a list written by to_list back into a mask for an image of this shape."""
    mask = np.zeros(shape, bool)
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "P":
            x, y = int(parts[1]), int(parts[2])
            mask[(shape[0] - 1 - y) if top_down else y, x] = True
    return mask


def build_list(fields: dict[str, list[Path]], out_dir: Path, dry_run: bool = False) -> Optional[Path]:
    """Write <out_dir>/hot_pixels.lst from the longest-exposure frames of every field. Cached by input frames.

    `fields` maps a field name to (path, exposure) pairs for every light; only the longest exposure is used,
    because hot pixels show best there. Returns the list path, or None when there is not enough data.
    """
    from .stack import stamp

    longest = max((e for items in fields.values() for _, e in items), default=None)
    chosen = {n: [p for p, e in items if e == longest] for n, items in fields.items()}
    chosen = {n: sorted(ps) for n, ps in chosen.items() if len(ps) >= 3}
    if not chosen:
        return None
    dest = out_dir / "hot_pixels.lst"
    if dry_run:
        print(f"--- hot pixels: would build {dest} from {sum(len(v) for v in chosen.values())} frames in {len(chosen)} field(s)")
        return dest
    new = stamp([{"path": str(p)} for ps in chosen.values() for p in ps],
                f"{K_SIGMA}:{MAX_RING}:{MAX_RING_SINGLE}:{FRAMES_PER_FIELD}")
    sp = out_dir / ".hot_pixels.stamp"
    if dest.exists() and sp.exists() and sp.read_text() == new:
        print(f"hot pixels: up to date ({dest})")
        return dest
    mask, info = find(chosen)
    if not mask.size:
        return None
    first = next(iter(chosen.values()))[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(to_list(mask, row_order_top_down(first)))
    sp.write_text(new)
    print(f"hot pixels: {info['count']} stable hot pixels ({100 * info['count'] / mask.size:.2f}% of the sensor), "
          f"{info['rule']}; list in {dest}")
    return dest
