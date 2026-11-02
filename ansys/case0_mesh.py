"""Case 0 mesh: uniform N x N quadrilateral mesh of the square cavity, written
directly in Fluent's ASCII mesh format (.msh), so no GUI meshing is needed.

Zones (names used by ansys/run_case0.py):
    fluid (cell zone 2), interior-fluid (3), wall_hot x=0 (4), wall_cold x=L (5),
    wall_top y=L (6), wall_bottom y=0 (7).

Format: Ansys Fluent User's Guide, "Fluent mesh file format" (sections 0, 2,
10 nodes, 12 cells, 13 faces, 45 zone names; integers in hexadecimal). Face
convention (confirmed by Fluent 2026 R1's mesh check): for a face n0 -> n1 the
vector (n1 - n0) x k points from cell c0 towards cell c1, i.e. c0 lies on the
LEFT of n0 -> n1; boundary faces have c1 = 0, so their normal points out of the
domain. The opposite order gives "left-handed faces" and negative volumes.
check_mesh() below verifies the convention independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ZONES = {"fluid": 2, "interior-fluid": 3, "wall_hot": 4, "wall_cold": 5,
         "wall_top": 6, "wall_bottom": 7}


@dataclass
class CavityMesh:
    L: float
    n: int
    nodes: np.ndarray                     # (n_nodes, 2) coordinates, m
    faces: dict[str, list[tuple[int, int, int, int]]]   # zone -> (n0, n1, c0, c1), 1-based

    @property
    def n_cells(self) -> int:
        return self.n * self.n

    @property
    def h(self) -> float:
        """Representative cell size normalised by L (as in case.yaml)."""
        return 1.0 / self.n


def build_cavity_mesh(L: float, n: int) -> CavityMesh:
    if n < 2 or L <= 0:
        raise ValueError("need n >= 2 and L > 0")
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs)            # row j = y index, column i = x index
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nd(i, j):
        return 1 + i + j * (n + 1)

    def cl(i, j):
        return 1 + i + j * n

    interior, hot, cold, top, bottom = [], [], [], [], []
    # (n1 - n0) x k must point from c0 to c1 (outwards on boundaries)
    for j in range(n):                    # vertical faces: edge -y -> normal -x, c0 at +x
        for i in range(1, n):
            interior.append((nd(i, j + 1), nd(i, j), cl(i, j), cl(i - 1, j)))
    for j in range(1, n):                 # horizontal faces: edge -x -> normal +y, c0 below
        for i in range(n):
            interior.append((nd(i + 1, j), nd(i, j), cl(i, j - 1), cl(i, j)))
    for j in range(n):
        hot.append((nd(0, j + 1), nd(0, j), cl(0, j), 0))               # normal -x (out)
        cold.append((nd(n, j), nd(n, j + 1), cl(n - 1, j), 0))          # normal +x (out)
    for i in range(n):
        bottom.append((nd(i, 0), nd(i + 1, 0), cl(i, 0), 0))            # normal -y (out)
        top.append((nd(i + 1, n), nd(i, n), cl(i, n - 1), 0))          # normal +y (out)
    return CavityMesh(L, n, nodes, {"interior-fluid": interior, "wall_hot": hot,
                                    "wall_cold": cold, "wall_top": top,
                                    "wall_bottom": bottom})


def check_mesh(m: CavityMesh) -> dict[str, float]:
    """Independent checks (Fluent convention): every face normal (n1-n0) x k
    points away from c0 (towards c1, or out of the domain), and each cell is
    closed (sum of its outward normals = 0)."""
    centres = np.zeros((m.n_cells + 1, 2))
    for j in range(m.n):
        for i in range(m.n):
            centres[1 + i + j * m.n] = ((i + 0.5) * m.L / m.n, (j + 0.5) * m.L / m.n)
    closure = np.zeros((m.n_cells + 1, 2))
    wrong = 0
    count = np.zeros(m.n_cells + 1, dtype=int)
    for zone, faces in m.faces.items():
        for n0, n1, c0, c1 in faces:
            p0, p1 = m.nodes[n0 - 1], m.nodes[n1 - 1]
            e = p1 - p0
            normal = np.array([e[1], -e[0]])          # e x k
            mid = 0.5 * (p0 + p1)
            if np.dot(centres[c0] - mid, normal) >= 0:
                wrong += 1
            closure[c0] += normal                       # outward of c0 is +normal
            count[c0] += 1
            if c1:
                closure[c1] -= normal
                count[c1] += 1
    return dict(
        wrong_handedness=wrong,
        max_closure_error=float(np.abs(closure[1:]).max()),
        min_faces_per_cell=int(count[1:].min()),
        max_faces_per_cell=int(count[1:].max()),
        n_faces=sum(len(f) for f in m.faces.values()),
    )


def write_msh(m: CavityMesh, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nn, nc = len(m.nodes), m.n_cells
    nf = sum(len(f) for f in m.faces.values())
    out = [f'(0 "railway-hvac-sim Case 0 square cavity, L = {m.L!r} m, {m.n} x {m.n} quads")',
           "(2 2)",
           f"(10 (0 1 {nn:x} 0 2))",
           f"(12 (0 1 {nc:x} 0))",
           f"(13 (0 1 {nf:x} 0))",
           f"(10 (1 1 {nn:x} 1 2)("]
    out += [f"{x:.17g} {y:.17g}" for x, y in m.nodes]
    out.append("))")
    out.append(f"(12 ({ZONES['fluid']:x} 1 {nc:x} 1 3))")
    first = 1
    for zone, faces in m.faces.items():
        bc = 2 if zone.startswith("interior") else 3
        last = first + len(faces) - 1
        out.append(f"(13 ({ZONES[zone]:x} {first:x} {last:x} {bc:x} 2)(")
        out += [f"{a:x} {b:x} {c:x} {d:x}" for a, b, c, d in faces]
        out.append("))")
        first = last + 1
    out.append(f"(45 ({ZONES['fluid']} fluid fluid)())")
    for zone in m.faces:
        kind = "interior" if zone.startswith("interior") else "wall"
        out.append(f"(45 ({ZONES[zone]} {kind} {zone})())")
    path.write_text("\n".join(out) + "\n", encoding="ascii")
    return path


if __name__ == "__main__":
    import sys
    L, n = float(sys.argv[1]), int(sys.argv[2])
    mesh = build_cavity_mesh(L, n)
    print(check_mesh(mesh))
    print(write_msh(mesh, Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"cavity_{n}.msh")))
