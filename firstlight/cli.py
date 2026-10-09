from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import siril as siril_mod
from .classify import build_manifest, scan
from . import enhance, hotpixels, quality, settings
from .finish import finish_target
from .stack import Masters, dominant_exposure, stack_filter


def reject_bad_frames(plan: dict, out: Path) -> dict:
    """Measure every frame to be stacked, judge each against its own group, drop clear outliers, and report."""
    import json

    from .stack import slug

    paths = [Path(f["path"]) for frames in plan.values() for f in frames]
    print(f"frame quality: measuring {len(paths)} frames (cached after the first run) ...")
    metrics = quality.measure_all(paths, out / ".frame_quality_cache.json")
    groups, meta = {}, {}
    for (target, flt), frames in plan.items():
        for f in frames:
            groups[f["path"]] = f"{target} | {flt} | {f['exptime']}"
            meta[f["path"]] = {"target": target, "filter": flt, "exposure": f["exptime"], "date_obs": f["date_obs"]}
    verdict: dict[str, list[str]] = {}
    for target in {t for t, _ in plan}:                       # thresholds can be overridden per target in settings.json
        override = settings.load(out / slug(target)).get("override", {})
        cfg = {k: override[k] for k in quality.DEFAULTS if k in override}
        sub = {p: g for p, g in groups.items() if meta[p]["target"] == target}
        order = {p: meta[p]["date_obs"] for p in sub}
        verdict.update(quality.cap(quality.judge(metrics, sub, cfg, order), sub, metrics))
    report = [{"path": p, **meta[p], **metrics[p], "rejected": bool(verdict[p]), "reasons": verdict[p]} for p in groups]
    (out / "frame_quality.json").write_text(json.dumps({"thresholds": quality.DEFAULTS, "frames": report}, indent=1))
    kept: dict = {}
    n_bad = 0
    for key, frames in plan.items():
        keep = [f for f in frames if not verdict[f["path"]]]
        for f in frames:
            if verdict[f["path"]]:
                n_bad += 1
                m = meta[f["path"]]
                print(f"rejected: {m['target']} / {m['filter']} at {m['date_obs'][11:19]}: {'; '.join(verdict[f['path']])}")
        kept[key] = keep
    print(f"frame quality: kept {sum(len(v) for v in kept.values())} of {len(paths)} frames, rejected {n_bad} "
          f"(details in {out / 'frame_quality.json'})")
    return kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="firstlight", description=__doc__)
    ap.add_argument("folder", type=Path, help="folder of one night's raw FITS frames (read-only)")
    ap.add_argument("-o", "--output", type=Path, help="output directory (default: <folder>-firstlight)")
    ap.add_argument("--target", help="only process targets whose name contains this text")
    ap.add_argument("--filter", dest="only_filter", help="only process this filter")
    ap.add_argument("--all-exposures", action="store_true", help="stack mixed exposure times together")
    grp = ap.add_argument_group("optional enhancement (off by default; applied to the linear output and previews)")
    for st in enhance.ORDER:
        grp.add_argument(st.flag, action="store_true", help=st.label)
    grp.add_argument("--star-splitter", choices=enhance.SPLITTERS, default="classical",
                     help="how stars are separated: built-in classical method, or StarNet if configured in Siril")
    grp.add_argument("--no-widget-enhance", action="store_true",
                     help="build the preview widget from the plain stacks instead of assuming the enhancements are on")
    ap.add_argument("--no-hot-pixel-fix", action="store_true",
                    help="do not correct the sensor's hot pixels (found from the lights, no darks needed)")
    ap.add_argument("--library", type=Path, help="dark library folder (default: $FIRSTLIGHT_LIBRARY or ~/.firstlight/library)")
    ap.add_argument("--no-darks", action="store_true", help="ignore the dark library and any darks in the folder")
    ap.add_argument("--no-reject-frames", action="store_true",
                    help="stack every frame, even soft, elongated or star-starved ones")
    ap.add_argument("--dry-run", action="store_true", help="write and print Siril scripts without running them")
    ap.add_argument("--keep-intermediates", action="store_true", help="keep calibrated/registered sequences")
    ap.add_argument("--siril", help="path to siril-cli (default: PATH, then /Applications/Siril.app)")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 2
    main_steps = tuple(st for st in enhance.ORDER if getattr(args, st.attr))
    why = enhance.unavailable(args.star_splitter)
    if why:
        print(f"error: --star-splitter {args.star_splitter}: {why}", file=sys.stderr)
        return 2
    if {"rl", "sx"} <= {st.key for st in main_steps}:
        print("note: --deconvolve has no effect with --remove-stars (the stars are discarded)")
    # The widget shows roughly what careful processing could give, so it assumes the enhancements are on.
    widget_steps = main_steps if args.no_widget_enhance else tuple(st for st in enhance.ORDER if st.in_widget)
    out = args.output or args.folder.with_name(args.folder.name + "-firstlight")
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(scan(args.folder))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for target, filters in manifest["targets"].items():
        parts = ", ".join(f"{flt}: {len(fr)}" for flt, fr in sorted(filters.items()))
        print(f"{target}: {parts}")
    missing = [k for k in ("dark", "flat") if not manifest["calibration"][k]]
    if missing:
        print(f"note: no {'/'.join(missing)} frames found; stacking without them")
    print(f"manifest: {out / 'manifest.json'}")

    siril_bin = None
    if not args.dry_run:
        try:
            siril_bin = siril_mod.find_siril(args.siril)
            print(f"siril: {siril_bin} {siril_mod.version(siril_bin)}")
        except siril_mod.SirilError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Darks: add this night's to the library, then pick the best matching master per filter below.
    library = None
    if not args.no_darks:
        from . import calibration
        library = calibration.Library(calibration.library_path(args.library))
        raw_darks = manifest["calibration"]["dark"]
        if raw_darks:
            try:
                calibration.build_masters(raw_darks, library, siril_bin, dry_run=args.dry_run)
            except siril_mod.SirilError as exc:
                print(f"warning: could not build master darks: {exc}")

    # Hot pixels belong to the sensor, so they are found across ALL fields (stars move, hot pixels do not).
    hot_list = None
    if not args.no_hot_pixel_fix:
        fields = {t: [(Path(f["path"]), f["exptime"]) for frames in filters.values() for f in frames]
                  for t, filters in manifest["targets"].items()}
        hot_list = hotpixels.build_list(fields, out, dry_run=args.dry_run)
        if hot_list is None:
            print("note: not enough frames to find hot pixels (need 3 at the longest exposure); skipping the fix")

    # Decide what to stack: the dominant exposure per filter, then drop clearly bad frames within each group.
    plan: dict[tuple[str, str], list[dict]] = {}
    for target, filters in manifest["targets"].items():
        if args.target and args.target.lower() not in target.lower():
            continue
        for flt, frames in sorted(filters.items()):
            if args.only_filter and args.only_filter != flt:
                continue
            if not args.all_exposures:
                frames, skipped = dominant_exposure(frames)
                if skipped:
                    exps = sorted({f["exptime"] for f in skipped})
                    print(f"warning: {target} / {flt}: ignoring {len(skipped)} frames with other "
                          f"exposures {exps} (use --all-exposures to include)")
            plan[(target, flt)] = frames
    if not args.no_reject_frames and not args.dry_run and plan:
        plan = reject_bad_frames(plan, out)

    failed = 0
    stacks: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    for (target, flt), frames in plan.items():
        try:
            masters = Masters()
            if library is not None:
                from . import calibration
                found, why = library.find_dark(calibration.light_conditions(frames), tol=settings.DEFAULTS["dark_temp_tolerance"])
                if found:
                    masters = Masters(dark=library.path_of(found))
                    print(f"{target} / {flt}: using master dark {found['file']} ({why})")
                elif library.masters():
                    print(f"{target} / {flt}: no matching dark: {why}; using the hot-pixel list from the lights")
            res = stack_filter(target, flt, frames, out, siril_bin, masters=masters, dry_run=args.dry_run,
                               keep=args.keep_intermediates, hot_list=hot_list)
            if res or args.dry_run:
                stacks.setdefault(target, {})[flt] = res or out / "dry-run" / flt
                raws.setdefault(target, {})[flt] = sorted(f["path"] for f in frames)[len(frames) // 2]
        except siril_mod.SirilError as exc:
            failed += 1
            print(f"error: {target} / {flt}: {exc}", file=sys.stderr)
    for target, st in stacks.items():
        try:
            finish_target(target, st, raws[target], out, siril_bin, dry_run=args.dry_run,
                          keep=args.keep_intermediates, main_steps=main_steps, widget_steps=widget_steps,
                          splitter=args.star_splitter)
        except siril_mod.SirilError as exc:
            failed += 1
            print(f"error: {target}: {exc}", file=sys.stderr)
    return 1 if failed else 0
