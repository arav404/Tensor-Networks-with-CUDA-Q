import cudaq
import pandas as pd
from qaoa_research.src.bench import run_experiment

# -----------------------------
# Experiment configuration
# -----------------------------
BACKENDS = ["qpp-cpu", "tensornet-mps"]
GRAPH_FAMILY = "k_regular"
K = 3
P = 2
SIZES = [6, 8, 10]
REPS = 3

# -----------------------------
# Run experiments
# -----------------------------
all_results = []

for backend in BACKENDS:
    print(f"\nRunning backend: {backend}")
    cudaq.set_target(backend)

    df = run_experiment(
        graph_family=GRAPH_FAMILY,
        sizes=SIZES,
        p=P,
        reps=REPS,
        backend=backend,
        k=K
    )

    df["backend"] = backend
    all_results.append(df)

# -----------------------------
# Save results
# -----------------------------
results = pd.concat(all_results, ignore_index=True)

outfile = f"results/{GRAPH_FAMILY}_p{P}.csv"
results.to_csv(outfile, index=False)

print(f"\nSaved results to {outfile}")
