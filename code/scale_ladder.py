"""Extend the scale ladder, measure every rung, then rebuild the paper.

The central vulnerability of a small-model study is the reviewer's question of
whether the phenomenon is an artifact of undertrained models. Two points cannot
answer it. This adds rungs so the horizon against scale is estimated rather than
bracketed and, more importantly, so we can ask whether the COUPLING between state
loss and behaviour is scale-invariant even where the horizon itself moves.

Runs last, after the main grid and its analysis, and finishes by regenerating
macros, tables, figures, both PDFs, and the QA gates so the delivered paper
includes every rung.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
RES = os.path.join(HERE, "..", "results")

LADDER = [
    ("attn_6L192_s0",  ["--layers", "6",  "--width", "192", "--heads", "6"]),
    ("attn_12L384_s0", ["--layers", "12", "--width", "384", "--heads", "8"]),
    ("attn_12L512_s0", ["--layers", "12", "--width", "512", "--heads", "8"]),
]
GRID_EXPECTED = 14
PATCH_ALPHA = "0.5"


def sh(cmd, label=""):
    print(f"+ {label}", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  ! returncode {r.returncode}", flush=True)
    return r.returncode


def have(n):
    return os.path.exists(os.path.join(RES, f"{n}.json"))


def main():
    py = sys.executable
    while len([f for f in os.listdir(RUNS) if f.endswith(".pt")]) < GRID_EXPECTED:
        time.sleep(240)
    # let the main analysis have the GPU first
    while not os.path.exists(os.path.join(RES, "attn_8L256_s0.json")):
        time.sleep(240)
    print("starting scale ladder", flush=True)

    for name, extra in LADDER:
        if not os.path.exists(os.path.join(RUNS, f"{name}.pt")):
            sh([py, "train.py", "--name", name, "--steps", "12000"] + extra,
               f"train {name}")
        if not have(name):
            sh([py, "eval_all.py", "--name", name, "--probe_games", "3000",
                "--eval_games", "3000", "--epochs", "2"], f"eval {name}")
        if not have(f"{name}_coupling"):
            sh([py, "coupling.py", "--name", name, "--games", "1500"],
               f"coupling {name}")
        if not have(f"{name}_belief"):
            sh([py, "belief.py", "--name", name, "--games", "1200"],
               f"belief {name}")
        if not have(f"{name}_patch"):
            sh([py, "patching.py", "--name", name, "--n", "600",
                "--alpha", PATCH_ALPHA], f"patch {name}")

    for step in ("make_macros.py", "make_tables.py", "figures.py",
                 "build_paper.py", "qa_gates.py"):
        sh([py, step], step)
    print("SCALE LADDER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
