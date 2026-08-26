# === ALG2 MOD (aux-selection): structural auxiliary evaluators. Pure functions. ===
# Each aux is a pair (function, direction):
#   direction = +1  -> higher is believed better (favor top half of the pool)
#   direction = -1  -> lower  is believed better (favor bottom half)
# The direction is a PRIOR supplied with the aux; the bandit can override it in effect
# by down-weighting an arm whose advantage on the oracle turns out negative.
#
# These NEVER touch the oracle and are NEVER shown to the LLM (scores stay private).
# They are used only to build alternative parent-SELECTION distributions.
#
# Deliberately NOT rewarded: equal radii, symmetry, total covered area, raw contact
# count. The SOTA n=26 packing is asymmetric, uses several smaller circles, and gains
# value from heterogeneous local geometry rather than regularity.
#
# Input contract (matches evaluate.py / run_packing):
#   centers : np.ndarray (n, 2) of (x, y)
#   radii   : np.ndarray (n,)
#   unit square is [0,1] x [0,1]

import numpy as np

# Two circles (or a circle and a wall) count as touching when the gap is below this.
# Optimizer-produced packings have near-exact tangency; 1e-3 is generous enough to
# catch structurally load-bearing near-contacts without inventing spurious ones.
CONTACT_TOL = 1e-3


def _contact_directions(centers, radii, i, tol=CONTACT_TOL):
    """Unit directions from circle i toward each of its contacts (circles + walls)."""
    c_i, r_i = centers[i], radii[i]
    dirs = []

    # circle-circle contacts
    d = np.linalg.norm(centers - c_i, axis=1)
    touching = np.abs(d - (radii + r_i)) < tol
    touching[i] = False
    for j in np.where(touching)[0]:
        v = centers[j] - c_i
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            dirs.append(v / nv)

    # circle-wall contacts (contact point lies on that wall)
    x, y = c_i
    if abs(x - r_i) < tol:          dirs.append(np.array([-1.0, 0.0]))  # left
    if abs((1.0 - x) - r_i) < tol:  dirs.append(np.array([1.0, 0.0]))   # right
    if abs(y - r_i) < tol:          dirs.append(np.array([0.0, -1.0]))  # bottom
    if abs((1.0 - y) - r_i) < tol:  dirs.append(np.array([0.0, 1.0]))   # top
    return dirs


def a_caging(centers, radii, tol=CONTACT_TOL) -> float:
    """a1: Directional caging -- mean caging MARGIN over circles.

    A circle is 'caged' if its contact directions leave no angular gap larger than pi
    (an open half-plane means it could drift that way). Rather than a binary
    caged/not-caged count -- which saturates at 1.0 among good packings and then gives
    no selection signal -- we score the *margin*:
        margin_i = clip((pi - largest_gap_i) / pi, 0, 1)
    A loose circle (gap >= pi) scores 0; a tightly surrounded one approaches 1. The mean
    over circles keeps support DISTRIBUTED (a few heavily-contacted circles can't carry
    the score) while remaining continuous among already-caged configurations.

    direction = +1 (maximize).
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(radii)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        dirs = _contact_directions(centers, radii, i, tol)
        if len(dirs) < 2:
            continue  # 0 or 1 contact -> free to move, margin 0
        ang = np.sort(np.array([np.arctan2(v[1], v[0]) for v in dirs]))
        gaps = np.diff(ang)
        wrap = 2.0 * np.pi - (ang[-1] - ang[0])
        largest_gap = max(gaps.max() if gaps.size else 0.0, wrap)
        total += float(np.clip((np.pi - largest_gap) / np.pi, 0.0, 1.0))
    return total / n


def a_residual_hole(centers, radii, grid: int = 160) -> float:
    """a2: Largest residual hole -- radius of the biggest empty disk in the square.

    A large open region signals the layout could be reorganized to fit bigger circles.
    max over grid points p of min(dist to wall, min_i(|p - c_i| - r_i)): the largest
    disk centred at p touching neither a circle nor the boundary.

    direction = -1 (minimize).
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) == 0:
        return 0.0
    g = np.linspace(0.0, 1.0, grid)
    px, py = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1)                   # (M,2)

    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)  # (M,n)
    clearance = (d - radii[None, :]).min(axis=1)                       # (M,)
    wall = np.minimum.reduce([pts[:, 0], pts[:, 1], 1.0 - pts[:, 0], 1.0 - pts[:, 1]])

    return float(max(0.0, np.minimum(clearance, wall).max()))


def _contact_adjacency(centers, radii, tol=CONTACT_TOL):
    """Adjacency matrix of the contact graph: n circles + one 'boundary' node (index n)."""
    n = len(radii)
    A = np.zeros((n + 1, n + 1))
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (abs(x - r) < tol or abs((1.0 - x) - r) < tol
                or abs(y - r) < tol or abs((1.0 - y) - r) < tol):
            A[i, n] = A[n, i] = 1.0
        for j in range(i + 1, n):
            if abs(float(np.linalg.norm(centers[i] - centers[j])) - (r + radii[j])) < tol:
                A[i, j] = A[j, i] = 1.0
    return A


def a_boundary_connectivity(centers, radii, tol=CONTACT_TOL) -> float:
    """a3: Boundary-anchored network strength -- algebraic connectivity (Fiedler value).

    Contact graph = circles + one 'boundary' super-node; edges = contacts. We use the
    second-smallest eigenvalue of the graph Laplacian (algebraic connectivity) rather
    than "fraction connected to a wall": the latter saturates at 1.0 for essentially
    every decent packing and therefore carries NO selection signal among good programs.
    The Fiedler value stays continuous -- it measures how *well-knit* and hard-to-sever
    the boundary-anchored load-bearing network is, and is 0 for a disconnected graph.

    direction = +1 (maximize).
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(radii)
    if n == 0:
        return 0.0
    A = _contact_adjacency(centers, radii, tol)
    L = np.diag(A.sum(axis=1)) - A
    ev = np.linalg.eigvalsh(L)          # symmetric -> real, ascending
    return float(max(0.0, ev[1]))       # second smallest = algebraic connectivity


# Registry: name -> (function, direction prior).
AUX_SPECS = {
    "caging":  (a_caging, +1),
    "hole":    (a_residual_hole, -1),
    "connect": (a_boundary_connectivity, +1),
}
AUX_NAMES = list(AUX_SPECS.keys())


def compute_all(centers, radii) -> dict:
    """Compute every aux for one configuration -> {name: float}."""
    return {name: float(fn(centers, radii)) for name, (fn, _) in AUX_SPECS.items()}
# === END ALG2 MOD (aux-selection) ===
