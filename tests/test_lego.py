from app.lego import dimensions, greedy_pack


def test_greedy_pack_covers_each_voxel_exactly_once():
    voxels = [
        {"x": x, "y": y, "z": z, "color": "#C84832"}
        for z in range(2)
        for y in range(4)
        for x in range(8)
    ]
    bricks, bom = greedy_pack(voxels)
    covered = set()
    for brick in bricks:
        for dx in range(brick["width"]):
            for dy in range(brick["depth"]):
                cell = (brick["x"] + dx, brick["y"] + dy, brick["z"])
                assert cell not in covered
                covered.add(cell)
    expected = {(v["x"], v["y"], v["z"]) for v in voxels}
    assert covered == expected
    assert sum(bom.values()) == len(bricks)
    assert len(bricks) < len(voxels)


def test_dimensions_handles_voxels_and_bricks():
    assert dimensions([]) == {"width": 0, "depth": 0, "height": 0}
    assert dimensions([{"x": 2, "y": 3, "z": 4}]) == {
        "width": 3,
        "depth": 4,
        "height": 5,
    }
    assert dimensions([{"x": 2, "y": 3, "z": 4, "width": 4, "depth": 2}]) == {
        "width": 6,
        "depth": 5,
        "height": 5,
    }


def test_greedy_pack_uses_six_stud_bricks_without_losing_detail():
    voxels = [
        {"x": x, "y": y, "z": 0, "color": "#F5C542"}
        for y in range(2)
        for x in range(12)
    ]

    bricks, bom = greedy_pack(voxels)

    assert len(bricks) == 2
    assert bom == {"2x6": 2}
    assert {brick["part_id"] for brick in bricks} == {"2456"}
