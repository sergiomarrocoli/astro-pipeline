"""The desktop app: a pywebview window around app_ui.html, driving the same pipeline as the command line.

`Backend` is plain Python (no window needed), so it can be tested; the window just exposes it to the page.
A run happens on a thread inside this process, with the pipeline's printed output collected line by line
and polled by the page. Only one run at a time.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from . import __version__, calibration, enhance
from .classify import build_manifest, scan
from .stack import slug


class _Lines(io.TextIOBase):
    """A write-only stream that keeps complete lines for polling."""
    def __init__(self, sink: list[str], lock: threading.Lock):
        self.sink, self.lock, self.partial = sink, lock, ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        with self.lock:
            self.partial += s.replace("\r", "\n")
            *done, self.partial = self.partial.split("\n")
            self.sink.extend(line for line in done if line.strip())
        return len(s)


def summarise(manifest: dict) -> dict:
    targets = [{"name": t, "filters": {f: len(fr) for f, fr in sorted(flts.items())},
                "seconds": sum((x.get("exptime") or 0) for fr in flts.values() for x in fr)}
               for t, flts in manifest["targets"].items()]
    return {"targets": targets, "calibration": {k: len(v) for k, v in manifest["calibration"].items()},
            "ignored": len(manifest["ignored"])}


def build_argv(folder: str, out: Optional[str], opts: dict) -> list[str]:
    """Command-line arguments for cli.main from the page's options."""
    argv = [folder]
    if out:
        argv += ["-o", out]
    if opts.get("target"):
        argv += ["--target", opts["target"]]
    if opts.get("filter"):
        argv += ["--filter", opts["filter"]]
    for flag in ("--all-exposures", "--no-hot-pixel-fix", "--no-reject-frames", "--no-darks",
                 "--no-widget-enhance", "--dry-run", *(s.flag for s in enhance.ORDER)):
        if opts.get(flag.lstrip("-")):
            argv.append(flag)
    if opts.get("library"):
        argv += ["--library", opts["library"]]
    return argv


def default_output(folder: str) -> str:
    p = Path(folder)
    return str(p.with_name(p.name + "-firstlight"))


def results(out: Path) -> list[dict]:
    """What a finished run produced, per target."""
    found = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()) if out.is_dir() else []:
        items = {k: d / v for k, v in (("widget", "preview/widget.html"), ("report", "report.html"), ("draft", "draft.jpg"))
                 if (d / v).exists()}
        if items:
            found.append({"target": d.name, **{k: str(v) for k, v in items.items()}})
    return found


class Backend:
    def __init__(self, window_getter=lambda: None):
        self._window = window_getter
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.cursor = 0
        self.state = "idle"          # idle | running | done | failed
        self.out: Optional[Path] = None
        self.thread: Optional[threading.Thread] = None

    # ---- what the page asks for -------------------------------------------------------
    def info(self) -> dict:
        return {"version": __version__,
                "steps": [{"flag": s.flag.lstrip("-"), "label": s.label} for s in enhance.ORDER],
                "library": str(calibration.library_path())}

    def pick_folder(self, start: Optional[str] = None) -> Optional[str]:
        import webview
        w = self._window()
        got = w.create_file_dialog(getattr(getattr(webview, 'FileDialog', None), 'FOLDER', None) or webview.FOLDER_DIALOG, directory=start or str(Path.home())) if w else None
        return got[0] if got else None

    def scan(self, folder: str) -> dict:
        p = Path(folder)
        if not p.is_dir():
            return {"error": f"{folder} is not a folder"}
        s = summarise(build_manifest(scan(p)))
        s["output"] = default_output(folder)
        if not s["targets"]:
            s["error"] = "no light frames found in this folder (looked at FITS headers, IMAGETYP = Light)"
        return s

    def library(self, folder: Optional[str] = None) -> list[dict]:
        lib = calibration.Library(calibration.library_path(folder))
        return [{"label": calibration.Conditions.of(m).label(), "temp": m.get("temp"), "frames": m["n_frames"],
                 "built": m["built"][:10]} for m in lib.masters()]

    def start(self, folder: str, out: Optional[str], opts: dict) -> dict:
        with self.lock:
            if self.state == "running":
                return {"error": "a run is already in progress"}
            self.log, self.cursor, self.state = [], 0, "running"
        out = out or default_output(folder)
        self.out = Path(out)
        self.thread = threading.Thread(target=self._run, args=(build_argv(folder, out, opts),), daemon=True)
        self.thread.start()
        return {"ok": True, "out": out}

    def poll(self) -> dict:
        with self.lock:
            lines, self.cursor = self.log[self.cursor:], len(self.log)
            return {"state": self.state, "lines": lines,
                    "results": results(self.out) if self.out and self.state in ("done", "failed") else []}

    def reveal(self, path: str) -> None:
        """Open a file with the default app (the widget and report open in the default browser) or show a folder."""
        subprocess.Popen(["open", path])

    # ---- the run ---------------------------------------------------------------------------
    def _run(self, argv: list[str]) -> None:
        from . import cli
        stream = _Lines(self.log, self.lock)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                code = cli.main(argv)
            stream.write("\n")
            end = "done" if code == 0 else "failed"
        except SystemExit as exc:            # argparse errors
            end = "done" if not exc.code else "failed"
        except Exception as exc:
            with self.lock:
                self.log.append(f"error: {type(exc).__name__}: {exc}")
            end = "failed"
        with self.lock:
            self.state = end


def selftest(folder: str) -> int:
    """Scan a folder and do a dry run through the backend, without a window (checks a packaged build)."""
    import importlib, pkgutil
    import firstlight
    for m in pkgutil.iter_modules(firstlight.__path__):
        importlib.import_module(f"firstlight.{m.name}")
    b = Backend()
    s = b.scan(folder)
    bad = [f for f in scan(Path(folder)) if f.kind == "other"]
    if bad:
        print("unreadable example:", bad[0].path, bad[0].imagetyp)
    print("targets:", [t["name"] for t in s.get("targets", [])], s.get("error", ""))
    b.start(folder, str(Path(folder).with_name("selftest-out")), {"dry-run": True})
    b.thread.join()
    p = b.poll()
    print("\n".join(p["lines"][-6:]))
    print("state:", p["state"])
    return 0 if p["state"] == "done" else 1


def main() -> int:
    import os
    if os.environ.get("FIRSTLIGHT_SELFTEST"):
        return selftest(os.environ["FIRSTLIGHT_SELFTEST"])
    import webview

    holder = {}
    backend = Backend(lambda: holder.get("w"))
    page = Path(__file__).with_name("app_ui.html")
    holder["w"] = webview.create_window(f"firstlight {__version__}", url=str(page), js_api=backend,
                                        width=980, height=820, min_size=(760, 560))
    webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
