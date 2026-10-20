import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from firstlight import cli, deepsnr, enhance, localtools, starnet
from firstlight.enhance import DENOISE, build_variant

FAKE = """#!{py}
import sys
from astropy.io import fits
a = sys.argv
src, out = a[a.index("-i") + 1], a[a.index("-o") + 1]
assert "--linear" in a
fits.PrimaryHDU((fits.getdata(src) * 0.25).astype("float32")).writeto(out)
"""


def fake_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for env in (deepsnr.ENV, starnet.ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    deepsnr.set_exe(None)
    starnet.set_exe(None)
    yield
    deepsnr.set_exe(None)
    starnet.set_exe(None)


def test_found_in_tools_folder_by_env_or_flag_and_not_mixed_up_with_starnet(tmp_path, monkeypatch):
    assert deepsnr.find_exe() is None
    exe = fake_exe(tmp_path / "tools" / "deepsnr_macos-arm64_1.3.2" / "deepsnr")
    assert deepsnr.find_exe() == exe and starnet.find_exe() is None
    mine = fake_exe(tmp_path / "elsewhere" / "deepsnr")
    monkeypatch.setenv(deepsnr.ENV, str(mine))
    assert deepsnr.find_exe() == mine
    deepsnr.set_exe(str(tmp_path / "missing"))
    assert deepsnr.find_exe() is None


def test_denoise_runs_the_tool_in_linear_mode(tmp_path):
    exe = fake_exe(tmp_path / "bin" / "deepsnr")
    src = tmp_path / "in.fit"
    fits.PrimaryHDU(np.full((8, 8), 0.4, np.float32)).writeto(src)
    out = deepsnr.denoise(exe, src, tmp_path)
    assert out.dtype == np.float32 and np.allclose(out, 0.1)


def test_a_failing_tool_raises_a_deepsnr_error_that_is_a_siril_error(tmp_path):
    exe = tmp_path / "deepsnr"
    exe.write_text("#!/bin/sh\necho 'image too small' >&2\nexit 3\n"); exe.chmod(0o755)
    with pytest.raises(deepsnr.DeepSNRError, match="image too small") as e:
        deepsnr.denoise(exe, tmp_path / "x.fit", tmp_path)
    assert isinstance(e.value, localtools.ToolError) and isinstance(e.value, enhance.siril.SirilError)


def test_availability_message_names_the_tool_and_the_flag(monkeypatch):
    assert "--deepsnr" in enhance.unavailable("deepsnr") and enhance.unavailable("siril") is None
    monkeypatch.setattr(deepsnr, "find_exe", lambda: Path("/bin/ls"))
    assert enhance.unavailable("deepsnr") is None


def test_the_deepsnr_denoiser_replaces_the_siril_stage_and_writes_starless_dn(tmp_path, monkeypatch):
    exe = fake_exe(tmp_path / "tools" / "deepsnr_x" / "deepsnr")
    lin = np.random.default_rng(2).uniform(0.1, 0.9, (16, 16)).astype(np.float32)
    layers = tmp_path / "t" / "layers"
    layers.mkdir(parents=True)
    fits.PrimaryHDU(lin, header=fits.Header({"FILTER": "H"})).writeto(layers / "starless_H.fit")
    siril_calls = []
    monkeypatch.setattr(enhance, "_stage", lambda *a, **k: siril_calls.append(a[1]))
    monkeypatch.setattr(enhance, "split_layers", lambda *a, **k: layers)
    enhance._process_layers(tmp_path / "t", ["H"], enhance.normalise([DENOISE]), "classical", None, False, False,
                            enhance.default_cfg(), "deepsnr")
    assert siril_calls == []                                  # no Siril denoise
    got = fits.getdata(layers / "starless_dn_H.fit")
    assert np.allclose(got, lin * 0.25) and fits.getheader(layers / "starless_dn_H.fit")["FILTER"] == "H"
    # a second run is cached
    before = (layers / "starless_dn_H.fit").stat().st_mtime_ns
    enhance._process_layers(tmp_path / "t", ["H"], enhance.normalise([DENOISE]), "classical", None, False, False,
                            enhance.default_cfg(), "deepsnr")
    assert (layers / "starless_dn_H.fit").stat().st_mtime_ns == before


def test_the_siril_denoiser_is_still_the_default(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(enhance, "_stage", lambda d, name, *a, **k: log.append(name))
    monkeypatch.setattr(enhance, "split_layers", lambda *a, **k: tmp_path)
    enhance._process_layers(tmp_path, ["H"], enhance.normalise([DENOISE]), "classical", None, False, False, enhance.default_cfg())
    assert log == ["dn_H"]


def test_cli_refuses_a_missing_denoiser_or_splitter_before_doing_any_work(tmp_path, capsys):
    assert cli.main([str(tmp_path), "--denoiser", "deepsnr"]) == 2
    assert "--denoiser deepsnr" in capsys.readouterr().err
    assert cli.main([str(tmp_path), "--star-splitter", "starnet"]) == 2
    assert "--star-splitter starnet" in capsys.readouterr().err
