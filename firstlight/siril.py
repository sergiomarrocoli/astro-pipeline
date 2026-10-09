"""Locate and drive siril-cli."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

APP_BUNDLE = Path("/Applications/Siril.app/Contents/MacOS/siril-cli")
NOISE = re.compile(r"CFITSIO")  # cosmetic header/library mismatch warning from brew builds
MIN_VERSION = "1.4.0"


class SirilError(RuntimeError):
    pass


def find_siril(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise SirilError(f"siril-cli not found at {explicit}")
    on_path = shutil.which("siril-cli")
    if on_path:
        return Path(on_path)
    # A GUI app launched from Finder has a minimal PATH, so also look where Homebrew puts it.
    for p in (Path("/opt/homebrew/bin/siril-cli"), Path("/usr/local/bin/siril-cli"), APP_BUNDLE):
        if p.exists():
            return p
    raise SirilError("siril-cli not found on PATH, in Homebrew's bin or in /Applications/Siril.app (brew install siril)")


def version(siril: Path) -> str:
    out = subprocess.run([str(siril), "--version"], capture_output=True, text=True).stdout
    m = re.search(r"\d+\.\d+\.\d+", out)
    if not m:
        raise SirilError(f"could not parse siril version from {out!r}")
    return m.group(0)


FAILURE = re.compile(r"Script execution failed|Error in line \d+|Command execution failed")


def failed(returncode: int, log_text: str) -> bool:
    """Siril can exit 0 when a command failed, so scan the log too. A bare "error:" is not enough:
    denoise prints "error: no suitable data in src fits" and then succeeds."""
    return returncode != 0 or bool(FAILURE.search(log_text))


def starnet_exe() -> Optional[Path]:
    """The StarNet executable configured in Siril's preferences, if it exists."""
    for cfg in sorted(Path("~/.config/siril").expanduser().glob("config.*.ini"), reverse=True):
        m = re.search(r"^starnet_exe=(.*)$", cfg.read_text(errors="replace"), re.M)
        if m and m.group(1).strip() and Path(m.group(1).strip()).exists():
            return Path(m.group(1).strip())
    return None


def run_script(siril: Path, workdir: Path, script: Path, log: Path) -> None:
    """Run a script with cwd=workdir; always keep the full log, raise on failure."""
    proc = subprocess.run(
        [str(siril), "-d", str(workdir), "-s", str(script)],
        capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"},
    )
    text = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if not NOISE.search(l))
    log.write_text(text + "\n")
    if failed(proc.returncode, text):
        tail = "\n".join(text.splitlines()[-8:])
        raise SirilError(f"siril failed (see {log}):\n{tail}")
