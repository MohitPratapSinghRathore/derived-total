"""Extend the scale ladder, then run the full diagnostic on every rung.

The central vulnerability of a small-model study is the reviewer's question of
whether the phenomenon is an artifact of undertrained models. Two points cannot
answer it. This adds larger rungs so that horizon against scale is estimated
rather than bracketed, and, more importantly, so we can test whether the
COUPLING between state loss and behavioural failure is scale-invariant even
where the horizon itself moves.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
RES = os.path.join(HERE, "..", "results")

# rungs beyond the main grid, ordered cheapest first so partial completion is useful
LADDER = [
    ("attn_6L192_s0",  ["--layers", "6",  "--width", "192", "--heads", "6"]),
    ("attn_12L384_s0", ["--layers", "12", "--width", "384", "--heads", "8"]),
    ("attn_12L512_s0", ["--layers", "12", "--width", "512", "--heads", "8"]),
]
GRID_EXPECTED = 14


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def main():
    py = sys.executable
    # wait for the main grid to free the GPU
    while len([f for f in os.listdir(RUNS) if f.endswith(".pt")]) < GRID_EXPECTED:
        time.sleep(180)
    print("main grid complete; starting scale ladder", flush=True)

    for name, extra in LADDER:
        if not os.path.exists(os.path.join(RUNS, f"{name}.pt")):
            sh([py, "train.py", "--name", name, "--steps", "12000"] + extra)
        if not os.path.exists(os.path.join(RES, f"{name}.json")):
            sh([py, "eval_all.py", "--name", name, "--probe_games", "3000",
                "--eval_games", "3000", "--epochs", "2"])
        if not os.path.exists(os.path.join(RES, f"{name}_coupling.json")):
            sh([py, "coupling.py", "--name", name, "--games", "1500"])
    print("SCALE LADDER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
