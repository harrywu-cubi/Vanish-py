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
