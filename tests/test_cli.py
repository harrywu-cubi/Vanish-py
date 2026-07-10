# tests/test_cli.py
import numpy as np
import pytest
from PIL import Image
from vanish import cli, io


def _write_img(path, h, w):
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    io.save_image(str(path), img)


def test_cli_shrink_width(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    cli.main(["shrink", str(src), str(dst), "--width", "9"])
    assert io.load_image(str(dst)).shape == (8, 9, 3)


def test_cli_enlarge_dw(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 10)
    cli.main(["enlarge", str(src), str(dst), "--dw", "3"])
    assert io.load_image(str(dst)).shape == (8, 13, 3)


def test_cli_remove_with_mask(tmp_path):
    src, dst, mpath = tmp_path / "in.png", tmp_path / "out.png", tmp_path / "m.png"
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    io.save_image(str(src), img)
    m = np.zeros((12, 16), dtype=np.uint8)
    m[:, 7:9] = 255
    Image.fromarray(m, mode="L").save(mpath)
    cli.main(["remove", str(src), str(dst), "--mask", str(mpath), "--shrink"])
    assert io.load_image(str(dst)).shape == (12, 14, 3)


def test_cli_shrink_rejects_width_not_smaller(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    with pytest.raises(SystemExit):
        cli.main(["shrink", str(src), str(dst), "--width", "20"])


def test_cli_shrink_dw(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    cli.main(["shrink", str(src), str(dst), "--dw", "-3"])
    assert io.load_image(str(dst)).shape == (8, 9, 3)


def test_cli_shrink_height(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 12, 10)
    cli.main(["shrink", str(src), str(dst), "--height", "9"])
    assert io.load_image(str(dst)).shape == (9, 10, 3)


def test_cli_shrink_rejects_equal_and_zero(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    with pytest.raises(SystemExit):
        cli.main(["shrink", str(src), str(dst), "--dw", "0"])     # target == current
    with pytest.raises(SystemExit):
        cli.main(["shrink", str(src), str(dst), "--dw", "3"])     # positive dw on shrink


def test_cli_enlarge_rejects_negative_dw(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    with pytest.raises(SystemExit):
        cli.main(["enlarge", str(src), str(dst), "--dw", "-3"])


def test_cli_energy_basic(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    cli.main(["energy", str(src), str(dst)])
    out = io.load_image(str(dst))
    assert out.shape == (8, 12, 3)
    assert out.dtype == np.uint8


def test_cli_energy_uniform_no_crash(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    io.save_image(str(src), np.full((8, 12, 3), 128, dtype=np.uint8))
    cli.main(["energy", str(src), str(dst)])
    out = io.load_image(str(dst))
    assert out.shape == (8, 12, 3)
    assert out.max() == 0   # uniform image => zero energy everywhere


def test_cli_seams_draws_red_without_mutating_input(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    original = io.load_image(str(src)).copy()
    cli.main(["seams", str(src), str(dst), "--count", "2"])
    out = io.load_image(str(dst))
    assert out.shape == (8, 12, 3)
    red = np.all(out == [255, 0, 0], axis=-1)
    assert red.sum() >= 8                       # at least one full-height seam
    assert np.array_equal(io.load_image(str(src)), original)   # source untouched


def test_cli_seams_rejects_bad_count(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    with pytest.raises(SystemExit):
        cli.main(["seams", str(src), str(dst), "--count", "0"])
    with pytest.raises(SystemExit):
        cli.main(["seams", str(src), str(dst), "--count", "12"])   # >= width
