# === ALG2 MOD (aux-selection): structural auxiliary evaluators for 2048. Pure fns. ===
# Each aux is a pair (function, direction):
#   direction = +1  -> higher is believed better;  -1 -> lower is believed better.
# The direction is a PRIOR; the bandit down-weights any arm whose advantage on the
# oracle turns out negative, so the experiment can tell GOOD auxes from BAD ones by
# which arms earn weight (q_aux) over the search.
#
# These NEVER touch the oracle (max tile reached / #moves) and are NEVER shown to the
# LLM (scores stay private). They only build alternative parent-SELECTION distributions.
#
# All five are the classic 2048 board heuristics -- distinct from the sparse oracle,
# and DENSE (defined at every move), which is exactly where aux-shaped selection should
# help most. Each is computed per board then averaged over the whole trajectory, so the
# aux measures the *style* of play the strategy produces, not just where it ended up.
#
# Input contract (matches evaluate.py / play_2048):
#   boards : sequence of (4, 4) int arrays of EXPONENTS
#            (0 = empty; value v means a tile worth 2**v; 11 = the 2048 tile)
import numpy as np

SIZE = 4


def _as_boards(boards) -> np.ndarray:
    b = np.asarray(boards)
    if b.ndim == 2:            # a single board -> trajectory of length 1
        b = b[None, ...]
    return b.astype(float)


def a_empty(boards) -> float:
    """a1: Mean fraction of EMPTY cells across the trajectory.

    Room to maneuver -- boards with more open space survive longer and keep options
    open. direction = +1 (maximize).
    """
    b = _as_boards(boards)
    if b.size == 0:
        return 0.0
    return float((b == 0).mean())          # mean over cells and time -> [0, 1]


def _line_monotonic(line: np.ndarray) -> float:
    """Fraction of adjacent pairs consistent with the line's better-ordered direction."""
    d = np.diff(line)
    if d.size == 0:
        return 1.0
    non_inc = float(np.mean(d <= 0))       # sorted high -> low
    non_dec = float(np.mean(d >= 0))       # sorted low  -> high
    return max(non_inc, non_dec)


def a_monotonic(boards) -> float:
    """a2: Mean MONOTONICITY of rows and columns (the 'staircase' good play keeps).

    For every row and column, the fraction of adjacent pairs ordered consistently in the
    line's stronger direction; averaged over the 8 lines and over time. 1.0 = every line
    perfectly ordered. direction = +1 (maximize).
    """
    b = _as_boards(boards)
    if b.size == 0:
        return 0.0
    total = 0.0
    for board in b:
        lines = [board[i, :] for i in range(SIZE)] + [board[:, j] for j in range(SIZE)]
        total += float(np.mean([_line_monotonic(ln) for ln in lines]))
    return total / len(b)


def a_smooth(boards) -> float:
    """a3: Mean SMOOTHNESS -- neighbouring occupied tiles have similar values.

    Small differences between adjacent (occupied) tiles mean they can soon merge. We
    average |exponent difference| over adjacent occupied pairs, then map to higher-is-
    better via 1/(1+mean_diff) so the aux is in (0, 1]. direction = +1 (maximize).
    """
    b = _as_boards(boards)
    if b.size == 0:
        return 0.0
    total = 0.0
    for board in b:
        diffs = []
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i, j] == 0:
                    continue
                if j + 1 < SIZE and board[i, j + 1] != 0:
                    diffs.append(abs(board[i, j] - board[i, j + 1]))
                if i + 1 < SIZE and board[i + 1, j] != 0:
                    diffs.append(abs(board[i, j] - board[i + 1, j]))
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        total += 1.0 / (1.0 + mean_diff)
    return total / len(b)


def a_corner(boards) -> float:
    """a4: Fraction of the trajectory where the MAX tile sits in a corner.

    The cornerstone strategy: anchoring the largest tile in a corner is the single
    strongest human/AI 2048 heuristic. direction = +1 (maximize).
    """
    b = _as_boards(boards)
    if b.size == 0:
        return 0.0
    corners = [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)]
    hits = 0.0
    for board in b:
        m = board.max()
        if m <= 0:
            continue
        if any(board[r, c] == m for r, c in corners):
            hits += 1.0
    return hits / len(b)


def a_merge(boards) -> float:
    """a5: Mean MERGE POTENTIAL -- fraction of adjacent pairs that are equal & nonzero.

    Adjacent equal tiles can be merged immediately; a board rich in them keeps the tile
    count down and builds bigger tiles faster. Normalised by the number of adjacent
    pairs. direction = +1 (maximize).
    """
    b = _as_boards(boards)
    if b.size == 0:
        return 0.0
    n_pairs = 2 * SIZE * (SIZE - 1)         # horizontal + vertical adjacencies
    total = 0.0
    for board in b:
        c = 0
        for i in range(SIZE):
            for j in range(SIZE):
                v = board[i, j]
                if v == 0:
                    continue
                if j + 1 < SIZE and board[i, j + 1] == v:
                    c += 1
                if i + 1 < SIZE and board[i + 1, j] == v:
                    c += 1
        total += c / n_pairs
    return total / len(b)


# Registry: name -> (function, direction prior). All +1: we WANT to learn which of these
# five structural signals actually predict good play (earn q_aux) and which do not.
AUX_SPECS = {
    "empty":     (a_empty, +1),
    "monotonic": (a_monotonic, +1),
    "smooth":    (a_smooth, +1),
    "corner":    (a_corner, +1),
    "merge":     (a_merge, +1),
}
AUX_NAMES = list(AUX_SPECS.keys())


def compute_all(boards) -> dict:
    """Compute every aux for one trajectory -> {name: float}."""
    return {name: float(fn(boards)) for name, (fn, _) in AUX_SPECS.items()}
# === END ALG2 MOD (aux-selection) ===
