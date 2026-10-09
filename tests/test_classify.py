import pytest

from firstlight.classify import (
    DARK, DARKFLAT, FLAT, BIAS, LIGHT, OTHER,
    build_manifest, classify_imagetyp, frame_from_header, scan,
)


def hdr(**kw):
    base = {"IMAGETYP": "LIGHT", "FILTER": "H", "OBJECT": "NGC 7000", "EXPTIME": 300.0,
            "GAIN": 120, "OFFSET": 50, "CCD-TEMP": -9.8, "DATE-OBS": "2026-10-03T22:53:04"}
    base.update(kw)
    return base


@pytest.mark.parametrize("value,kind", [
    ("LIGHT", LIGHT), ("Light Frame", LIGHT), ("DARK", DARK), ("Master Dark", OTHER),
    ("FLAT", FLAT), ("BIAS", BIAS), ("Zero", BIAS), ("DARKFLAT", DARKFLAT),
    ("Dark Flat", DARKFLAT), ("SNAPSHOT", OTHER), (None, OTHER), ("", OTHER),
])
def test_classify_imagetyp(value, kind):
    assert classify_imagetyp(value) == kind


def test_frame_fields():
    f = frame_from_header("a.fits", hdr())
    assert (f.kind, f.target, f.filter, f.exptime, f.gain, f.offset, f.temp) == (
        LIGHT, "NGC 7000", "H", 300.0, 120, 50, -9.8)


def test_missing_keys_do_not_fail():
    f = frame_from_header("a.fits", {"IMAGETYP": "DARK"})
    assert f.kind == DARK and f.exptime is None and f.filter is None


def test_exposure_fallback_and_string_gain():
    f = frame_from_header("a.fits", {"IMAGETYP": "LIGHT", "EXPOSURE": "60.0", "GAIN": "120.0"})
    assert f.exptime == 60.0 and f.gain == 120


def test_manifest_groups_by_target_and_filter():
    headers = {
        "1.fits": hdr(), "2.fits": hdr(), "3.fits": hdr(FILTER="O"),
        "4.fits": hdr(OBJECT="M31"), "5.fits": hdr(IMAGETYP="DARK", FILTER=""),
        "6.fits": hdr(IMAGETYP="SNAPSHOT"),
    }
    frames = [frame_from_header(k, v) for k, v in headers.items()]
    m = build_manifest(frames)
    assert {k: len(v) for k, v in m["targets"]["NGC 7000"].items()} == {"H": 2, "O": 1}
    assert len(m["targets"]["M31"]["H"]) == 1
    assert len(m["calibration"]["dark"]) == 1 and m["calibration"]["flat"] == []
    assert m["ignored"] == ["6.fits"]


def test_scan_uses_headers_not_folder_names(tmp_path):
    (tmp_path / "DARK").mkdir()
    (tmp_path / "DARK" / "x.fits").touch()  # folder says dark, header says light
    (tmp_path / "notes.txt").touch()
    frames = scan(tmp_path, header_reader=lambda p: hdr())
    assert [f.kind for f in frames] == [LIGHT]


def test_scan_survives_unreadable_file(tmp_path):
    (tmp_path / "bad.fits").touch()
    def boom(p): raise OSError("truncated")
    frames = scan(tmp_path, header_reader=boom)
    assert frames[0].kind == OTHER


def test_nina_snapshot_marked_light_is_ignored():
    f = frame_from_header("s.fits", hdr(OBJECT="Snapshot", FILTER="UV/IR cut", EXPTIME=3.0))
    assert f.kind == OTHER


def test_dominant_exposure_picks_most_integration():
    from firstlight.stack import dominant_exposure
    frames = [{"exptime": 300.0}] * 3 + [{"exptime": 120.0}] * 5 + [{"exptime": 60.0}]
    main, rest = dominant_exposure(frames)
    assert {f["exptime"] for f in main} == {300.0} and len(rest) == 6


def test_binning_and_instrument_are_read_for_matching_calibration_frames():
    f = frame_from_header("a.fits", {"IMAGETYP": "DARK", "XBINNING": 2, "INSTRUME": "SVBONY SV605MC", "EXPTIME": 300})
    assert (f.binning, f.instrument, f.kind) == (2, "SVBONY SV605MC", DARK)
    g = frame_from_header("b.fits", {"IMAGETYP": "LIGHT"})
    assert g.binning is None and g.instrument is None
