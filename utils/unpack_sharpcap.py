"""Unpack Siril FITSEQ files from SharpCap captures into one FITS per frame.

Each filter folder holds SharpCap's stack (`Stack_*.fits`, `*.CameraSettings.txt`, `stacklog.csv`) and a
sibling `<name>.fit` FITSEQ made by Siril, whose headers are stripped. The missing header values come from
the stack header (object, gain, temperature, instrument) and the settings file (the per-frame exposure; the
stack's EXPTIME is the total), and per-frame times from stacklog.csv. `darks.fit` gets the settings of
`darks/` when that folder exists (SharpCap saved only the stack's settings, so this assumes the 16-20 darks
were taken with them), else its own header.

    .venv/bin/python utils/unpack_sharpcap.py SRC OUT --object M31 red=R green=G blue=B
    .venv/bin/python utils/unpack_sharpcap.py SRC OUT --object IC1805 halpha=H o3=O si=S

SRC/<name>.fit (+ SRC/<name>/ for the headers) and SRC/darks.fit are read; OUT/lights and OUT/darks written.
Rows stay as Siril stored them (no ROWORDER, so bottom-up).
"""
import argparse
import csv
import glob
import re
from pathlib import Path

from astropy.io import fits

STRUCTURAL = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO", "BSCALE", "XTENSION",
              "PCOUNT", "GCOUNT", "EXTNAME")


def settings(folder: Path) -> dict:
    """The `key=value` lines of SharpCap's CameraSettings file."""
    out = {}
    for p in glob.glob(str(folder / "*.CameraSettings.txt")):
        for line in open(p, encoding="utf-8-sig"):
            k, _, v = line.strip().partition("=")
            out[k] = v
    return out


def exposure_of(s: dict) -> float:
    return float(re.sub(r"[^0-9.]", "", s["Exposure"]))


def light_header(stack: fits.Header, obj: str, filt: str, exposure: float) -> fits.Header:
    h = fits.Header()
    h["IMAGETYP"] = "LIGHT"
    h["OBJECT"] = obj
    h["FILTER"] = filt
    h["EXPTIME"] = exposure
    h["XBINNING"] = h["YBINNING"] = 1
    for k in ("INSTRUME", "GAIN", "OFFSET", "CCD-TEMP", "XPIXSZ", "YPIXSZ", "FOCALLEN"):
        if k in stack:
            h[k] = stack[k]
    return h


def unpack(seq: Path, hdr: fits.Header, times: list, outdir: Path, stem: str) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    with fits.open(seq, memmap=False) as hl:  # BZERO (uint16) rules out memmap
        for i, hdu in enumerate(hl):          # frame 0 sits in the primary HDU
            h = hdr.copy()
            if i < len(times):
                h["DATE-OBS"] = times[i]
            fits.PrimaryHDU(hdu.data, h).writeto(outdir / f"{stem}_{i:04d}.fits", overwrite=True)
            n += 1
    return n


def dark_header(src: Path) -> fits.Header:
    folder = src / "darks"
    s = settings(folder) if folder.exists() else {}
    stacks = glob.glob(str(folder / "Stack_*.fits"))
    with fits.open(src / "darks.fit", memmap=False) as hl:
        own = hl[0].header
        d = fits.Header([c for c in own.cards if c.keyword not in STRUCTURAL])
    if s and stacks:                     # the folder's settings are the better source
        st = fits.getheader(stacks[0])
        d = light_header(st, "", "", exposure_of(s))
        d.remove("OBJECT"), d.remove("FILTER")
        d["GAIN"], d["CCD-TEMP"] = int(s["Gain"]), float(s["Temperature"])
    d["IMAGETYP"] = "DARK"
    d.remove("OBJECT", ignore_missing=True)
    d["XBINNING"] = d["YBINNING"] = 1
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--object", required=True, help="target name for the OBJECT header")
    ap.add_argument("filters", nargs="+", help="name=LETTER, e.g. red=R halpha=H")
    a = ap.parse_args()
    src, out = a.src.expanduser(), a.out.expanduser()
    for pair in a.filters:
        name, _, filt = pair.partition("=")
        s = settings(src / name)
        stack = fits.getheader(glob.glob(str(src / name / "Stack_*.fits"))[0])
        with open(src / name / "stacklog.csv", encoding="utf-8-sig") as f:
            times = [r[0].strip().replace("+00:00", "") for r in list(csv.reader(f))[1:]]
        n = unpack(src / f"{name}.fit", light_header(stack, a.object, filt, exposure_of(s)), times,
                   out / "lights", f"{a.object}_{filt}")
        print(f"{name} -> {filt}: {n} frames ({exposure_of(s):g} s, gain {stack.get('GAIN')}), {len(times)} timestamps")
    if (src / "darks.fit").exists():
        d = dark_header(src)
        n = unpack(src / "darks.fit", d, [], out / "darks", "dark")
        print(f"darks: {n} frames ({d.get('EXPTIME'):g} s, gain {d.get('GAIN')}, {d.get('CCD-TEMP')} C)")


if __name__ == "__main__":
    main()
