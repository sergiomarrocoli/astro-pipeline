"""Unpack the 2025-02-28 Andromeda Siril FITSEQ files into one FITS per frame.

Source: <src>/andromeda/{red,green,blue}.fit (50 frames each, 18.08 s, SharpCap
captures converted by Siril, headers stripped) and darks.fit (20 x 18 s). The
missing header values come from the SharpCap stack header and stacklog.csv
beside each filter.

    .venv/bin/python scripts/unpack_andromeda.py ~/Documents/astro-photos/andromeda <out-night-folder>
"""
import csv
import glob
import sys
from pathlib import Path

from astropy.io import fits

FILTERS = {"red": "R", "green": "G", "blue": "B"}
OBJECT = "M31"


def base_header(stack: fits.Header, filt: str) -> fits.Header:
    h = fits.Header()
    h["IMAGETYP"] = "LIGHT"
    h["OBJECT"] = OBJECT
    h["FILTER"] = filt
    h["EXPTIME"] = 18.08  # SharpCap "Exposure=18.080s"; the stack header holds the 904 s total
    h["XBINNING"] = h["YBINNING"] = 1
    for k in ("INSTRUME", "GAIN", "OFFSET", "CCD-TEMP", "XPIXSZ", "YPIXSZ", "FOCALLEN"):
        if k in stack:
            h[k] = stack[k]
    return h


def unpack(seq: Path, hdr: fits.Header, times: list, outdir: Path, stem: str) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    with fits.open(seq, memmap=False) as hl:  # BZERO (uint16) rules out memmap
        for i, hdu in enumerate(hl):
            h = hdr.copy()
            if i < len(times):
                h["DATE-OBS"] = times[i]
            fits.PrimaryHDU(hdu.data, h).writeto(outdir / f"{stem}_{i:04d}.fits", overwrite=True)
            n += 1
    return n


def main(src: Path, out: Path):
    for name, filt in FILTERS.items():
        stack = fits.getheader(glob.glob(str(src / name / "Stack_*.fits"))[0])
        with open(src / name / "stacklog.csv", encoding="utf-8-sig") as f:
            times = [r[0].strip().replace("+00:00", "") for r in list(csv.reader(f))[1:]]
        n = unpack(src / f"{name}.fit", base_header(stack, filt), times, out / "lights",
                   f"{OBJECT}_{filt}")
        print(f"{name}: {n} frames, {len(times)} timestamps")
    # darks: keep the real header, fix the type (SharpCap/Siril wrote IMAGETYP=Light)
    with fits.open(src / "darks.fit", memmap=False) as hl:
        first = hl[0].header
        d = fits.Header([c for c in first.cards if c.keyword not in
                         ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO",
                          "BSCALE", "XTENSION", "PCOUNT", "GCOUNT", "EXTNAME")])
    d["IMAGETYP"] = "DARK"
    d.remove("OBJECT", ignore_missing=True)
    d["XBINNING"] = d["YBINNING"] = 1
    n = unpack(src / "darks.fit", d, [], out / "darks", "dark")
    print(f"darks: {n} frames")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
