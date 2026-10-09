# firstlight

One command turns a night's astrophotography frames into previews, clean linear stacks and an HTML report.

![One 18 s H-alpha frame of the Heart Nebula (left) and the SHO preview firstlight made from 100 + 101 + 100 such frames (right)](docs/heart-before-after.jpg)

*Left: a single 18 s H-alpha sub of the Heart Nebula (IC 1805), stretched. Right: the SHO preview from 301 frames, produced by one command.*

```
firstlight ~/astro/2026-10-03 -o ~/astro/2026-10-03-out
```

firstlight reads the FITS frames, sorts them by header (not by folder name), calibrates, registers and stacks each target and filter with [Siril](https://siril.org), aligns the filters to each other, and writes:

- **Linear stacks**, one 32-bit FITS per filter, aligned and cropped to a common area with the sky gradient removed. This is the real output, ready for finishing in Siril.
- **Previews**: auto-stretched JPGs per filter and an SHO (three filters) or HOO (H + O) composite. These are throwaway; judge the night with them.
- **A report** (`report.html`): a raw sub next to the result, frames used and rejected, integration per filter, star size, frame-quality charts and a sensor-tilt check.
- **A widget** (`preview/widget.html`): a self-contained page for playing with palette, stretch, colour and framing on the processed layers, with full-resolution tiles on zoom. It needs no server; open it next to its `widget_tiles/` folder.

Colour balance and the final stretch are left to you; the tool stops where taste starts.

## What it does for you

- Finds the sensor's **hot pixels from the lights** (no darks needed) and fixes them before stacking.
- Keeps a **dark library** (`~/.firstlight/library`), matched on camera, binning, gain, offset, exposure (within 0.5 s) and temperature (within 1.5 C). Without a match it says why and falls back to the hot-pixel list.
- **Rejects bad frames** (soft, elongated or star-poor ones) by comparing each with its neighbours in time, never more than 25% of a group.
- Removes the **sky gradient** per filter, conservatively: it will not model a bright region as sky.
- Optional **star/starless split**, deconvolution of the stars and denoising of the starless layer (`--deconvolve`, `--denoise`, `--remove-stars`).

Every stage keeps its Siril script (`.ssf`) and log beside its output, and a re-run skips stages whose inputs have not changed.

## Install

Requires Python 3.10 or newer and Siril 1.4 or newer (`brew install siril`; `siril-cli` on PATH, or `/Applications/Siril.app`).

```
python3.14 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/firstlight --help
```

There is also a desktop app, `firstlight-app` (`pip install -e '.[app]'`), with folder pickers, a live log and links to each result.

## Usage

```
firstlight <night-folder> -o <out> [--target NGC] [--filter H] [--dry-run] [--keep-intermediates]
           [--all-exposures] [--deconvolve] [--denoise] [--remove-stars]
           [--no-darks] [--no-hot-pixel-fix] [--no-reject-frames]
```

`--dry-run` writes and prints every generated script without running Siril. Input is **NINA output**: one FITS per frame with NINA's headers (`IMAGETYP`, `OBJECT`, `FILTER`, `EXPTIME`, `GAIN`, `OFFSET`, sensor temperature). Only the exposure time with the most integration per filter is stacked unless you pass `--all-exposures`.

`utils/unpack_sharpcap.py` converts old SharpCap captures (Siril FITSEQ files) into NINA-style frames, for test data.

## Status

Built: classification, dark library, hot pixels, frame rejection, stacking, cross-filter alignment, background removal, previews, report, widget, desktop app. Not built yet: GraXpert, plate solving, flats. Developed against an SVBONY SV605 mono with H, O and S filters.

`PLAN.md` is the original brief; `CLAUDE.md` describes the architecture and what was learned on real data.

## Tests

```
.venv/bin/pytest -q
```

No Siril is needed for the tests.
