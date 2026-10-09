"""Stage 1: classify FITS frames by header contents (never by folder name)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

FITS_SUFFIXES = {".fits", ".fit", ".fts"}

LIGHT, DARK, FLAT, BIAS, DARKFLAT, OTHER = "light", "dark", "flat", "bias", "darkflat", "other"


@dataclass(frozen=True)
class Frame:
    path: str
    kind: str
    target: Optional[str]
    filter: Optional[str]
    exptime: Optional[float]
    gain: Optional[int]
    offset: Optional[int]
    temp: Optional[float]
    date_obs: Optional[str]
    imagetyp: Optional[str]
    binning: Optional[int] = None       # XBINNING; part of what makes a dark match a light
    instrument: Optional[str] = None    # INSTRUME

    def to_dict(self) -> dict:
        return asdict(self)


def classify_imagetyp(value) -> str:
    """Map NINA / generic IMAGETYP strings onto our frame kinds."""
    if not value:
        return OTHER
    v = re.sub(r"[^a-z]", "", str(value).lower())
    if "darkflat" in v or "flatdark" in v:
        return DARKFLAT
    if v in ("light", "lightframe", "science", "object"):
        return LIGHT
    if v.startswith("dark"):
        return DARK
    if v.startswith("flat"):
        return FLAT
    if v.startswith("bias") or v == "zero" or v.startswith("offset"):
        return BIAS
    return OTHER  # SNAPSHOT, TEST, etc. are ignored


def _num(header: Mapping, *keys, cast=float):
    for k in keys:
        if k in header and header[k] not in (None, ""):
            try:
                return cast(header[k])
            except (TypeError, ValueError):
                pass
    return None


def _text(header: Mapping, key: str) -> Optional[str]:
    v = header.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def frame_from_header(path: str | Path, header: Mapping) -> Frame:
    kind = classify_imagetyp(header.get("IMAGETYP", header.get("FRAME")))
    # NINA writes snapshots as IMAGETYP=LIGHT; OBJECT is the only header marker.
    if kind == LIGHT and (_text(header, "OBJECT") or "").lower() == "snapshot":
        kind = OTHER
    gain = _num(header, "GAIN", cast=lambda x: int(float(x)))
    offset = _num(header, "OFFSET", cast=lambda x: int(float(x)))
    return Frame(
        path=str(path),
        kind=kind,
        target=_text(header, "OBJECT"),
        filter=_text(header, "FILTER"),
        exptime=_num(header, "EXPTIME", "EXPOSURE"),
        gain=gain,
        offset=offset,
        temp=_num(header, "CCD-TEMP", "SET-TEMP"),
        date_obs=_text(header, "DATE-OBS"),
        imagetyp=_text(header, "IMAGETYP"),
        binning=_num(header, "XBINNING", cast=lambda x: int(float(x))),
        instrument=_text(header, "INSTRUME"),
    )


def read_header(path: Path) -> Mapping:
    from astropy.io import fits

    return fits.getheader(path, 0)


def find_fits(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FITS_SUFFIXES
    )


def scan(root: Path, header_reader=read_header) -> list[Frame]:
    frames = []
    for p in find_fits(root):
        try:
            frames.append(frame_from_header(p, header_reader(p)))
        except Exception as exc:  # unreadable file: record and move on
            frames.append(Frame(str(p), OTHER, None, None, None, None, None, None, None, f"unreadable: {exc}"))
    return frames


def build_manifest(frames: Iterable[Frame]) -> dict:
    """Group lights by target and filter; keep calibration frames flat for later matching."""
    frames = list(frames)
    targets: dict[str, dict[str, list[dict]]] = {}
    for f in frames:
        if f.kind != LIGHT:
            continue
        t = f.target or "unknown"
        targets.setdefault(t, {}).setdefault(f.filter or "none", []).append(f.to_dict())
    calib = {k: [f.to_dict() for f in frames if f.kind == k] for k in (DARK, FLAT, BIAS, DARKFLAT)}
    return {
        "targets": targets,
        "calibration": calib,
        "ignored": [f.path for f in frames if f.kind == OTHER],
    }
