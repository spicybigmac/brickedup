from pathlib import Path

from app.manual import _build_steps, build_pdf


def test_pdf_is_created(tmp_path: Path):
    bricks = [
        {
            "x": 0,
            "y": 0,
            "z": 0,
            "width": 2,
            "depth": 2,
            "height": 1,
            "color": "#C84832",
        }
    ]
    copy = {
        "title": "Test Build",
        "intro": "Build it.",
        "safety": "Small parts.",
        "tips": ["Sort.", "Align.", "Press."],
    }
    output = tmp_path / "manual.pdf"
    build_pdf(output, bricks, {"2x2": 1}, copy)
    assert output.read_bytes().startswith(b"%PDF")


def test_dense_layers_are_split_into_readable_complete_steps():
    bricks = [
        {
            "x": index,
            "y": 0,
            "z": 0,
            "width": 1,
            "depth": 1,
            "color": f"#{index:02X}3366",
        }
        for index in range(18)
    ]
    steps = _build_steps({0: bricks})
    assert sum(len(batch) for _z, batch in steps) == len(bricks)
    assert all(len(batch) <= 12 for _z, batch in steps)
    assert all(len({(b["width"], b["depth"], b["color"]) for b in batch}) <= 6 for _z, batch in steps)
