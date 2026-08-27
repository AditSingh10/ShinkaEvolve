# Rectangle Strip Packing Example

This example evolves constructive heuristics for two-dimensional rectangle
strip packing. The strip has a fixed width and unbounded height; lower valid
packing height is better.

Each candidate receives an instance dictionary and a deterministic seed:

```python
def run_packing(instance, seed):
    return {
        "placements": [
            {"id": 0, "x": 0.0, "y": 0.0, "rotated": False},
        ]
    }
```

Every input rectangle ID must occur exactly once. A placement contains only
`id`, `x`, `y`, and the boolean `rotated` flag. Rotation is legal only when the
instance enables it. Coordinates must be finite and nonnegative; rectangles
must stay within the strip and may touch, but not overlap with positive area.

The evaluator never trusts candidate-reported dimensions or metrics. It
independently verifies the original input, detects candidate input mutation,
derives placed dimensions, checks all geometry, computes used height and lower
bounds, then assigns a score in `[0, 1]`. Invalid candidates receive score
zero.

Run the baseline evaluator from this directory:

```bash
python evaluate.py --program_path initial.py --results_dir results/manual
```

Run evolution with:

```bash
python run_evo.py
```

The baseline uses a deterministic rotation-aware shelf heuristic. Evolution
can improve it with bottom-left, skyline, maximal-rectangle, guillotine,
local-search, or hybrid strategies without external optimization libraries.
