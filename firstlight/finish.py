"""Stages 4-6: cross-filter alignment + crop, background removal, previews."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

from . import siril
from .siril import MIN_VERSION
from .stack import slug, stamp

ROLES = {"h": "H", "ha": "H", "halpha": "H", "o": "O", "oiii": "O", "o3": "O",
         "s": "S", "sii": "S", "s2": "S",
         "r": "R", "red": "R", "g": "G", "green": "G", "b": "B", "blue": "B"}


def role(filter_name: str) -> Optional[str]:
    return ROLES.get(re.sub(r"[^a-z0-9]", "", filter_name.lower()))


def composite_channels(roles: set[str]) -> Optional[tuple[str, tuple[str, str, str]]]:
    """(palette name, (R, G, B) roles). SHO for three filters, HOO for H+O, RGB for R+G+B, else none."""
    if {"S", "H", "O"} <= roles:
        return "SHO", ("S", "H", "O")
    if {"H", "O"} <= roles:
        return "HOO", ("H", "O", "O")
    if {"R", "G", "B"} <= roles:
        return "RGB", ("R", "G", "B")
    return None


def align_script(names: list[str]) -> str:
    """names = filters in order; inputs are f_<index>.fit symlinks in cwd."""
    lines = [f"requires {MIN_VERSION}"]
    if len(names) > 1:
        lines += ["link f", "register f -2pass", "seqapplyreg f -framing=min"]
        loads = [f"r_f_{i + 1:05d}" for i in range(len(names))]
    else:
        loads = ["f_00001"]
        lines += ["convert f"]  # single filter: nothing to align, just normalise names
    for flt, src in zip(names, loads):
        lines += [f"load {src}", f"save aligned_{slug(flt)}"]    # sky gradient removal is a separate stage (background.py)
    lines.append("close")
    return "\n".join(lines) + "\n"


def preview_script(names: list[str], palette: Optional[tuple[str, tuple[str, str, str]]]) -> str:
    lines = [f"requires {MIN_VERSION}"]
    for n in names:
        s = slug(n)
        lines += [f"load raw_{s}", "autostretch", f"savejpg raw_{s} 92",
                  f"load linear_{s}", "autostretch", f"savejpg preview_{s} 92"]
    if palette:
        pname, (r, g, b) = palette
        by_role = {role(n): slug(n) for n in names}
        lines += [f"rgbcomp linear_{by_role[r]} linear_{by_role[g]} linear_{by_role[b]} -out=composite_{pname}",
                  f"load composite_{pname}", "autostretch", f"savejpg preview_{pname} 92"]
    lines.append("close")
    return "\n".join(lines) + "\n"


def _stage(d: Path, name: str, script: str, inputs: list[Path], setup: Callable[[Path], None],
           outputs: list[str], siril_bin: Optional[Path], dry_run: bool, keep: bool) -> bool:
    """Run one stage in d/work. Returns False if skipped. Outputs are moved up into d."""
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.ssf").write_text(script)
    if dry_run:
        print(f"--- {d.name}/{name}\n{script}")
        return False
    new = stamp([{"path": str(p)} for p in inputs], script)
    sp = d / f".{name}.stamp"
    if all((d / o).exists() for o in outputs) and sp.exists() and sp.read_text() == new:
        print(f"{d.parent.name} / {name}: up to date, skipping")
        return False
    work = d / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    setup(work)
    print(f"{d.parent.name} / {name}: running ...")
    try:
        siril.run_script(siril_bin, work, d / f"{name}.ssf", d / f"{name}.log")
        for o in outputs:
            if not (work / o).exists():
                raise siril.SirilError(f"siril produced no {o} (see {d / (name + '.log')})")
            shutil.move(str(work / o), d / o)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    sp.write_text(new)
    return True


def finish_target(target: str, stacks: dict[str, Path], raws: dict[str, str], out: Path,
                  siril_bin: Optional[Path], dry_run=False, keep=False,
                  main_steps=(), widget_steps=(), splitter="classical") -> None:
    """stacks: filter -> stack fits; raws: filter -> path of one raw sub (for the 'before' image).

    main_steps are the enhancement steps the user asked for; they feed the previews. widget_steps are
    applied only to the widget's data, so it can show roughly what careful processing would give.
    """
    from . import enhance
    names = sorted(stacks)
    tdir = out / slug(target)

    aligned, linear = tdir / "aligned", tdir / "linear"
    def link_stacks(work: Path):
        for i, n in enumerate(names):
            os.symlink(stacks[n], work / f"in_{i + 1:02d}.fit")
    _stage(aligned, "align", align_script(names), [stacks[n] for n in names], link_stacks,
           [f"aligned_{slug(n)}.fit" for n in names], siril_bin, dry_run, keep)
    from . import background, settings as settings_mod
    cur = settings_mod.current(tdir)
    background.stage(tdir, names, cur["background_degree"], dry_run)

    cfg = None
    if not dry_run:
        from . import settings
        cfg = settings.resolve(tdir, {n: linear / f"linear_{slug(n)}.fit" for n in names})
        print(f"{target}: star FWHM {cfg['fwhm']} px (measured on {cfg.get('fwhm_measured_on')}); settings in {tdir.name}/settings.json")

    main_dir = enhance.build_variant(tdir, names, main_steps, splitter, siril_bin, dry_run, keep, cfg)   # errors here are the user's problem
    widget_layers = None
    try:
        widget_layers = enhance.build_layers(tdir, names, widget_steps, splitter, siril_bin, dry_run, keep, cfg)
    except siril.SirilError as exc:
        print(f"warning: {target}: could not build the widget's layers, skipping the widget\n{exc}")

    prev = tdir / "preview"
    palette = composite_channels({role(n) for n in names if role(n)})
    outs = [f"{p}_{slug(n)}.jpg" for n in names for p in ("raw", "preview")]
    if palette:
        outs.append(f"preview_{palette[0]}.jpg")
    def link_inputs(work: Path):
        for n in names:
            os.symlink(main_dir / f"linear_{slug(n)}.fit", work / f"linear_{slug(n)}.fit")
            os.symlink(raws[n], work / f"raw_{slug(n)}.fit")
    if widget_layers is not None:
        if not dry_run:
            from .widget import write_widget
            write_widget(target, widget_layers, names, out, auto=cfg)
        else:
            print(f"--- {tdir.name}/widget: would write preview/widget.html from {widget_layers.name}")
    _stage(prev, "preview", preview_script(names, palette),
           [main_dir / f"linear_{slug(n)}.fit" for n in names] + [Path(raws[n]) for n in names],
           link_inputs, outs, siril_bin, dry_run, keep)

    if widget_layers is not None and not dry_run:
        try:
            from . import report
            pick = "H" if "H" in names else names[0]
            dest = report.write(out, target, names, cfg, widget_layers, prev / f"raw_{slug(pick)}.jpg")
            print(f"{target} / report: {dest}")
        except Exception as exc:  # the report is a convenience: it must never take the pipeline down
            print(f"warning: {target}: could not write the report ({type(exc).__name__}: {exc})")
