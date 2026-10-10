# firstlight: one-command previews of a night's astrophotography

Working name, change freely. This is a build brief for a CLI tool, written to be handed to Claude Code.

## What it is

Point it at a folder of raw frames from one night of imaging and get back, with no further input:

1. A preview image of each target, good enough to judge whether the night worked.
2. Clean, linear, aligned per-filter stacks, ready for manual finishing in the Siril GUI.
3. A one-page HTML report of the night.

```
firstlight ~/astro/2026-10-03
```

The tool drives `siril-cli` (Siril's headless mode). It replaces the repetitive calibrate, register, and stack steps I currently click through by hand. It deliberately stops before final colour balance and stretching, which stay manual because they are taste decisions.

## Why it might interest non-astro people

A single raw frame of a nebula looks like black noise. A stack of a few hundred looks like a photograph. The README should lead with that before/after, and the HTML report should include it for every target (one raw sub next to the stacked preview). "Hundreds of noisy frames in, a picture out, one command" is the pitch.

## My setup

- Capture: NINA on a Windows mini PC, writing FITS files.
- Camera: SVBONY SV605 mono (IMX533), 5-position filter wheel, mostly narrowband (Ha, OIII, SII).
- Processing: Apple Silicon MacBook Air, Siril installed. Disk space is limited, so intermediate files matter.

## Pipeline

Each stage writes to its own output folder and is skipped on re-run if its outputs are current.

1. **Scan and classify.** Read FITS headers (IMAGETYP, FILTER, EXPTIME, OBJECT, GAIN, OFFSET, sensor temperature) to sort frames into lights, darks, flats, and bias or dark-flats, grouped by target and filter. Use headers, not folder names, so it survives changes to NINA's file naming. Write a manifest (JSON).
2. **Master calibration frames.** Master dark, master flat per filter, master bias or dark-flat. Cache masters in a library keyed by gain, offset, temperature, and exposure, so darks are not rebuilt every night.
3. **Per target, per filter.** Calibrate, register, stack with pixel rejection. Output: one linear 32-bit FITS per filter.
4. **Cross-filter alignment and crop.** Register the filter stacks to each other and crop to the common area so they overlay exactly.
5. **Background extraction.** Siril's own, with GraXpert CLI as an optional alternative if installed.
6. **Previews.** Auto-stretched JPG or PNG per filter, plus a colour composite (HOO if two filters, SHO if three). These are throwaway previews; the linear files from step 5 are the real output.
7. **Report.** Static HTML: previews, before/after, frames used and rejected per filter, total integration time, and whatever quality stats Siril exposes from registration (FWHM, star count, background).

Optional flags, off by default because good settings vary with the data: plate solving, denoise, deconvolution, star removal (StarNet).

## Constraints

- **Never modify the input folder.** Treat raw frames as read-only; work in a separate output directory using symlinks where possible.
- **Degrade gracefully.** Missing darks or flats should produce a warning and a preview, not a failure. A preview from uncalibrated data is still useful.
- **Clean up intermediates by default** (calibrated and registered sequences are many gigabytes), with a flag to keep them.
- **Be debuggable.** Save every generated Siril script and its log next to the outputs. Provide `--dry-run` that prints the scripts without running them.
- **Verify Siril command syntax against the installed version** before relying on it. Command names and options have changed between releases (for example, `preprocess` became `calibrate`). Check `siril-cli --version` and the command reference; start generated scripts with a `requires` line.
- Find `siril-cli` on PATH first, then inside the macOS app bundle.

## Not in scope

Final colour balance, palette mixing, and stretching. Capture control. Replacing the Siril GUI.

## Ask me before building

- A sample of NINA's output folder (I will provide a small real dataset for testing).
- Which calibration frames I actually take, and whether I use bias or dark-flats.
- Exactly which filters are in the wheel.
- Language. Python is the obvious choice (astropy reads FITS headers), but I mostly write Ruby, so say if there is a good reason either way.

## Done when

- The one command above, run on a real night's folder, produces previews, linear stacks, and the report with no manual steps.
- Re-running it does no unnecessary work.
- Frame classification has unit tests using synthetic FITS headers.
- The README opens with a real before/after image.