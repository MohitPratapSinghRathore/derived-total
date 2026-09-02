"""Train seeds 1 and 2 for every scale-ladder rung, then re-measure.

The ladder was single seed, which forced the paper to report the direction of the
scale trend rather than its slope. This removes that caveat. One rung at a time,
cheapest first, so partial completion still improves the table.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
RES = os.path.join(HERE, "..", "results")
LOCK = os.path.join(HERE, ".ladder_seeds.lock")

RUNGS = [
    ("attn_6L192",  ["--layers", "6",  "--width", "192", "--heads", "6"]),
    ("attn_12L256", ["--layers", "12", "--width", "256", "--heads", "8"]),
    ("attn_8L384",  ["--layers", "8",  "--width", "384", "--heads", "8"]),
    ("attn_12L384", ["--layers", "12", "--width", "384", "--heads", "8"]),
    ("attn_12L512", ["--layers", "12", "--width", "512", "--heads", "8"]),
]
SEEDS = [1, 2]


def sh(cmd, label):
    print(f"\n=== {label}  [{time.strftime('%H:%M')}]", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  ! {label} returned {r.returncode}", flush=True)
    return r.returncode


def main():
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3600:
        print("another instance is running; refusing to start")
        return
    open(LOCK, "w").write(str(os.getpid()))
    py = sys.executable
    try:
        for base, arch in RUNGS:
            for sd in SEEDS:
                name = f"{base}_s{sd}"
                if not os.path.exists(os.path.join(RUNS, f"{name}.pt")):
                    sh([py, "train.py", "--name", name, "--steps", "12000",
                        "--seed", str(sd)] + arch, f"train {name}")
                if not os.path.exists(os.path.join(RES, f"{name}.json")):
                    sh([py, "eval_all.py", "--name", name, "--probe_games", "3000",
                        "--eval_games", "3000", "--epochs", "2"], f"eval {name}")
                if not os.path.exists(os.path.join(RES, f"{name}_belief.json")):
                    sh([py, "belief.py", "--name", name, "--games", "1200"],
                       f"belief {name}")
                if not os.path.exists(os.path.join(RES, f"{name}_auc.json")):
                    sh([py, "auc.py", "--name", name, "--games", "1500"],
                       f"auc {name}")
        for step in ("make_macros.py", "make_tables.py", "figures.py",
                     "build_paper.py", "qa_gates.py"):
            sh([py, step], step)
        print("\nLADDER SEEDS COMPLETE", flush=True)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()
