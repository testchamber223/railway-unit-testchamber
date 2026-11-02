"""Case 0 mesh generator (ansys/case0_mesh.py): geometry and face handedness.
No Fluent needed."""

import pytest

from ansys.case0_mesh import build_cavity_mesh, check_mesh, write_msh


@pytest.mark.parametrize("n", [2, 40, 160])
def test_mesh_is_closed_and_right_handed(n):
    m = build_cavity_mesh(0.05, n)
    chk = check_mesh(m)
    assert chk["wrong_handedness"] == 0
    assert chk["max_closure_error"] < 1e-15
    assert chk["min_faces_per_cell"] == chk["max_faces_per_cell"] == 4
    assert chk["n_faces"] == 2 * n * (n + 1)          # 2n(n-1) interior + 4n boundary
    assert m.n_cells == n * n and m.h == pytest.approx(1 / n)
    assert len(m.nodes) == (n + 1) ** 2


def test_walls_cover_the_whole_boundary():
    n, L = 8, 0.03
    m = build_cavity_mesh(L, n)
    xs = {zone: {round(m.nodes[a - 1][0], 12) for a, b, *_ in f} |
                {round(m.nodes[b - 1][0], 12) for a, b, *_ in f}
          for zone, f in m.faces.items()}
    assert xs["wall_hot"] == {0.0} and xs["wall_cold"] == {L}
    for zone in ("wall_hot", "wall_cold", "wall_top", "wall_bottom"):
        assert len(m.faces[zone]) == n


def test_msh_file_sections(tmp_path):
    p = write_msh(build_cavity_mesh(0.05, 4), tmp_path / "c.msh")
    text = p.read_text()
    assert "(2 2)" in text
    assert "(12 (2 1 10 1 3))" in text                 # 16 quad cells, hex
    for name in ("wall_hot", "wall_cold", "wall_top", "wall_bottom", "interior-fluid"):
        assert name in text


def test_cells_have_positive_signed_area():
    """Fluent convention: c0 lies left of n0 -> n1, so walking each face with c0 on
    the left goes counter-clockwise -> positive area. (The reversed order made
    Fluent 2026 R1 report left-handed faces and negative volumes.)"""
    n, L = 5, 0.04
    m = build_cavity_mesh(L, n)
    area = [0.0] * (m.n_cells + 1)
    for faces in m.faces.values():
        for n0, n1, c0, c1 in faces:
            (x0, y0), (x1, y1) = m.nodes[n0 - 1], m.nodes[n1 - 1]
            a = 0.5 * (x0 * y1 - x1 * y0)
            area[c0] += a
            if c1:
                area[c1] -= a
    for a in area[1:]:
        assert a == pytest.approx((L / n) ** 2, rel=1e-12)
