# Vehicle Routing with Time Windows Example

This example evolves constructive and improvement heuristics for the Vehicle
Routing Problem with Time Windows (VRPTW). Valid solutions first minimize the
number of vehicles and then minimize total travel distance.

Each candidate receives an instance dictionary and deterministic seed:

```python
def run_vrptw(instance, seed):
    return {
        "routes": [
            [0, 3, 5, 1, 0],
            [0, 2, 4, 0],
        ]
    }
```

Every route starts and ends at depot `0`. Every customer must appear exactly
once, depot `0` cannot appear inside a route, and the number of routes cannot
exceed `max_vehicles`. Capacity, travel time, waiting, service time, customer
time windows, and an optional depot return deadline are all enforced.

The evaluator deep-copies each input instance and rejects mutation. It also
recomputes route loads, arrival and service-start times, return times, travel
distance, travel time, vehicle count, and score from the original instance.
Candidate-reported metrics are ignored. Invalid candidates receive score zero.

Run the baseline evaluator from this directory:

```bash
python evaluate.py --program_path initial.py --results_dir results/manual
```

Run evolution with:

```bash
python run_evo.py
```

The baseline is a deterministic feasibility-first greedy constructor. No
external routing solver is required. Evolution has room to discover insertion,
savings, clustering, local-search, large-neighborhood, small-subproblem, and
hybrid approaches.
