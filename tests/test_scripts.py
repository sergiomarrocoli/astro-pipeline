"""Generated Siril scripts and stage skipping. No Siril needed."""
from pathlib import Path

from firstlight.finish import align_script, composite_channels, preview_script, role
from firstlight.stack import Masters, build_script, slug, stamp


def lines(script):
    return script.strip().splitlines()


def test_requires_line_is_first_everywhere():
    assert lines(build_script(Masters(), "s"))[0].startswith("requires 1.4")
    assert lines(align_script(["H", "O"]))[0].startswith("requires 1.4")
    assert lines(preview_script(["H"], None))[0].startswith("requires 1.4")


def test_uncalibrated_stack_script():
    s = lines(build_script(Masters(), "stack_H"))
    assert "calibrate" not in " ".join(s)
    assert "register light -2pass" in s
    assert "seqapplyreg light -framing=min" in s  # common-area crop removes edge seams
    assert s[-2].startswith("stack r_light rej w 3 3") and s[-2].endswith("-out=stack_H")
    assert "-32b" in s[-2]


def test_calibrated_stack_script_uses_masters_and_pp_sequence():
    m = Masters(dark=Path("master_dark.fit"), flat=Path("master_flat.fit"))
    s = lines(build_script(m, "stack_H"))
    cal = next(l for l in s if l.startswith("calibrate"))
    assert "-dark=master_dark.fit" in cal and "-flat=master_flat.fit" in cal and "-cc=dark" in cal
    assert "register pp_light -2pass" in s and any(l.startswith("stack r_pp_light") for l in s)


def test_align_script_multi_filter_maps_each_registered_frame_to_its_filter():
    s = lines(align_script(["H", "O", "S"]))
    assert "link f" in s and "register f -2pass" in s and "seqapplyreg f -framing=min" in s
    # order matters: frame N of the registered sequence is filter N
    loads = [l for l in s if l.startswith("load")]
    saves = [l for l in s if l.startswith("save")]
    assert loads == ["load r_f_00001", "load r_f_00002", "load r_f_00003"]
    assert saves == ["save aligned_H", "save aligned_O", "save aligned_S"]
    assert not any("subsky" in l for l in s)             # gradient removal is its own stage


def test_align_script_single_filter_skips_registration():
    s = lines(align_script(["H"]))
    assert not any(l.startswith(("register", "link", "seqapplyreg")) for l in s)
    assert "load f_00001" in s and "save aligned_H" in s


def test_preview_script_composite_channel_order():
    pal = composite_channels({"S", "H", "O"})
    s = lines(preview_script(["H", "O", "S"], pal))
    assert "rgbcomp linear_S linear_H linear_O -out=composite_SHO" in s
    assert "savejpg preview_SHO 92" in s


def test_preview_script_rgb_composite_follows_roles_not_name_order():
    s = lines(preview_script(["B", "G", "R"], composite_channels({"B", "G", "R"})))
    assert "rgbcomp linear_R linear_G linear_B -out=composite_RGB" in s


def test_preview_script_without_palette_has_no_composite():
    assert "rgbcomp" not in preview_script(["H", "L"], None)


def test_composite_choice_and_filter_roles():
    assert composite_channels({"H", "O", "S"}) == ("SHO", ("S", "H", "O"))
    assert composite_channels({"H", "O"}) == ("HOO", ("H", "O", "O"))
    assert composite_channels({"H", "S"}) is None
    assert composite_channels({"R", "G", "B"}) == ("RGB", ("R", "G", "B"))
    assert [role(n) for n in ("H", "Ha", "OIII", "SII", "L")] == ["H", "H", "O", "S", None]
    assert [role(n) for n in ("R", "Red", "g", "Green", "B")] == ["R", "R", "G", "G", "B"]


def test_slug_is_filesystem_safe():
    assert slug("NGC 7635") == "NGC_7635" and slug("UV/IR cut") == "UV_IR_cut" and slug("///") == "x"


def test_stamp_changes_with_script_or_input(tmp_path):
    f = tmp_path / "a.fits"; f.write_text("x")
    frames = [{"path": str(f)}]
    a = stamp(frames, "script")
    assert a == stamp(frames, "script")
    assert a != stamp(frames, "script2")
    f.write_text("xy")  # size/mtime change = re-run
    assert a != stamp(frames, "script")


def test_a_single_frame_is_used_as_is_scaled_to_unit_range(tmp_path, capsys):
    import numpy as np
    from astropy.io import fits
    from firstlight.stack import stack_filter

    src = tmp_path / "raw.fits"
    counts = np.full((6, 8), 2000, dtype=np.uint16); counts[2, 3] = 65535
    fits.PrimaryHDU(counts).writeto(src)
    out = stack_filter("NGC 7000", "O", [{"path": str(src), "exptime": 300.0}], tmp_path / "out", None)  # no Siril needed
    data = fits.getdata(out)
    assert data.dtype.kind == "f" and data.dtype.itemsize == 4 and data.max() == 1.0 and abs(float(np.median(data)) - 2000 / 65535) < 1e-6
    assert "only 1 frame" in capsys.readouterr().out
    # re-running skips: the stamp matches
    assert stack_filter("NGC 7000", "O", [{"path": str(src), "exptime": 300.0}], tmp_path / "out", None) == out
    assert "up to date" in capsys.readouterr().out


def test_hot_pixel_correction_comes_after_convert_and_before_registration():
    s = lines(build_script(Masters(), "stack_H", hot=True))
    assert s.index("convert light") < s.index("seqcosme light hot.lst") < s.index("register cosme_light -2pass")
    assert "seqapplyreg cosme_light -framing=min" in s and any(l.startswith("stack r_cosme_light") for l in s)
    assert not any("seqcosme" in l for l in lines(build_script(Masters(), "stack_H")))   # off by default


def test_hot_pixel_correction_follows_calibration_when_masters_exist():
    s = lines(build_script(Masters(dark=Path("d.fit")), "stack_H", hot=True))
    assert s.index(next(l for l in s if l.startswith("calibrate"))) < s.index("seqcosme pp_light hot.lst")
    assert "register cosme_pp_light -2pass" in s


def test_single_frame_gets_its_hot_pixels_fixed_in_python(tmp_path):
    import numpy as np
    from astropy.io import fits
    from firstlight.stack import stack_filter

    counts = np.full((12, 12), 2000, dtype=np.uint16); counts[4, 6] = 60000
    h = fits.Header(); h["ROWORDER"] = "TOP-DOWN"
    fits.PrimaryHDU(counts, header=h).writeto(tmp_path / "raw.fits")
    lst = tmp_path / "hot.lst"; lst.write_text("P 6 7 H\n")          # array row 4 of 12, top-down -> y = 12 - 1 - 4 = 7
    out = stack_filter("T", "O", [{"path": str(tmp_path / "raw.fits"), "exptime": 300.0}], tmp_path / "out", None, hot_list=lst)
    data = fits.getdata(out)
    assert abs(float(data[4, 6]) - 2000 / 65535) < 1e-6 and data.max() < 0.05   # the spike is gone


def test_dry_run_works_when_the_hot_pixel_list_has_not_been_written_yet(tmp_path, capsys):
    from pathlib import Path
    from firstlight.stack import stack_filter
    p = tmp_path / "f.fits"; p.write_bytes(b"x")
    frames = [{"path": str(p), "exptime": 60.0}, {"path": str(p), "exptime": 60.0}]
    assert stack_filter("T", "H", frames, tmp_path / "o", None, dry_run=True, hot_list=tmp_path / "o" / "hot_pixels.lst") is None
