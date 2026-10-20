import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from firstlight import enhance, starnet

FAKE = """#!{py}
import sys
from astropy.io import fits
a = sys.argv
src, out = a[a.index("-i") + 1], a[a.index("-o") + 1]
assert "--linear" in a
d = fits.getdata(src)
fits.PrimaryHDU((d * 0.5).astype("float32")).writeto(out)
"""


def fake_exe(path: Path, body: str = FAKE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(starnet.ENV, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    starnet.set_exe(None)
    yield
    starnet.set_exe(None)


def test_found_in_tools_folder_and_none_otherwise(tmp_path):
    assert starnet.find_exe() is None
    exe = fake_exe(tmp_path / "tools" / "starnet2_macos-arm64_2.6.2" / "starnet2")
    assert starnet.find_exe() == exe


def test_explicit_path_wins_and_a_wrong_one_is_not_silently_replaced(tmp_path, monkeypatch):
    fake_exe(tmp_path / "tools" / "starnet2_x" / "starnet2")
    mine = fake_exe(tmp_path / "elsewhere" / "starnet2")
    monkeypatch.setenv(starnet.ENV, str(mine))
    assert starnet.find_exe() == mine
    starnet.set_exe(str(tmp_path / "missing"))
    assert starnet.find_exe() is None


def test_remove_stars_runs_the_tool_in_linear_mode_and_returns_its_output(tmp_path):
    exe = fake_exe(tmp_path / "bin" / "starnet2")
    src = tmp_path / "in.fit"
    fits.PrimaryHDU(np.full((8, 8), 0.4, np.float32)).writeto(src)
    out = starnet.remove_stars(exe, src, tmp_path)
    assert out.dtype == np.float32 and np.allclose(out, 0.2)
    assert os.listdir(tmp_path / "bin") == ["starnet2"] and not [p for p in tmp_path.iterdir() if p.name.startswith("tmp")]


def test_a_failing_tool_raises_with_its_message(tmp_path):
    exe = fake_exe(tmp_path / "starnet2", "#!/bin/sh\necho 'image too small' >&2\nexit 3\n")
    with pytest.raises(starnet.StarNetError, match="image too small"):
        starnet.remove_stars(exe, tmp_path / "x.fit", tmp_path)


def test_the_starnet_splitter_keeps_starless_plus_stars_exact(tmp_path):
    exe = fake_exe(tmp_path / "tools" / "starnet2_x" / "starnet2")
    rng = np.random.default_rng(1)
    lin = rng.uniform(0.1, 0.9, (16, 16)).astype(np.float32)
    (tmp_path / "t" / "linear").mkdir(parents=True)
    fits.PrimaryHDU(lin, header=fits.Header({"FILTER": "H"})).writeto(tmp_path / "t" / "linear" / "linear_H.fit")
    layers = enhance.split_layers(tmp_path / "t", ["H"], "starnet", None, False, False)
    sl, st, pos = (fits.getdata(layers / f"{k}_H.fit") for k in ("starless", "stars", "stars_pos"))
    assert np.allclose(sl, lin * 0.5) and np.allclose(sl + st, lin) and pos.min() >= 0
    assert fits.getheader(layers / "starless_H.fit")["FILTER"] == "H"
