"""Full unattended analysis: wait for the grid, measure everything, build the paper.

Stages, each skipped if its output already exists so the driver is restartable:
  1. per-model diagnostic (probes, controls, illegal rate, horizons)
  2. test-time ablation of the recurrent read-out
  3. coupling: aggregate vs action-relevant state error
  4. belief consistency, with mismatched and random controls
  5. causal patching, at the calibrated edit strength
  6. E_plan via the engine, conditioned on verified state
  7. macros, tables, figures, PDF builds, QA gates
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
RES = os.path.join(HERE, "..", "results")

CORE = ["attn_8L256", "attnpm_8L256", "alsb_l0_8L256", "alsb_l1_8L256"]
GRID = [f"{c}_s{s}" for s in (0, 1, 2) for c in CORE] + \
       ["attn_12L256_s0", "attn_8L384_s0"]
LADDER = ["attn_6L192_s0", "attn_12L384_s0", "attn_12L512_s0"]
PATCH_ALPHA = "0.5"      # calibrated; see results/*_patch_sweep.json


def sh(cmd, label=""):
    print(f"+ {label or ' '.join(str(c) for c in cmd[1:4])}", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  ! returncode {r.returncode}", flush=True)
    return r.returncode


def have(n):
    return os.path.exists(os.path.join(RES, f"{n}.json"))


def main():
    py = sys.executable

    while len([f for f in os.listdir(RUNS) if f.endswith(".pt")]) < len(GRID):
        n = len([f for f in os.listdir(RUNS) if f.endswith(".pt")])
        print(f"waiting for grid: {n}/{len(GRID)}", flush=True)
        time.sleep(180)
    print("grid complete", flush=True)

    # 1. diagnostics. Full controls on seed 0 only; they triple probe cost.
    for n in GRID:
        if not os.path.exists(os.path.join(RUNS, f"{n}.pt")) or have(n):
            continue
        extra = ["--controls"] if n.endswith("_s0") else []
        sh([py, "eval_all.py", "--name", n, "--probe_games", "3000",
            "--eval_games", "3000", "--epochs", "2"] + extra, f"eval {n}")

    # 2. read-out ablation
    for s in (0, 1, 2):
        n = f"alsb_l1_8L256_s{s}"
        if os.path.exists(os.path.join(RUNS, f"{n}.pt")) and not have(f"{n}_zablate"):
            sh([py, "eval_all.py", "--name", n, "--ablate_z", "--probe_games", "3000",
                "--eval_games", "3000", "--epochs", "2"], f"ablate {n}")

    # 3-5. mechanism, on seed 0 of every condition and every ladder rung
    for n in [f"{c}_s0" for c in CORE] + ["attn_12L256_s0", "attn_8L384_s0"] + LADDER:
        if not have(n):
            continue
        if not have(f"{n}_coupling"):
            sh([py, "coupling.py", "--name", n, "--games", "1500"], f"coupling {n}")
        if not have(f"{n}_belief"):
            sh([py, "belief.py", "--name", n, "--games", "1200"], f"belief {n}")
        if not have(f"{n}_patch"):
            sh([py, "patching.py", "--name", n, "--n", "600",
                "--alpha", PATCH_ALPHA], f"patch {n}")

    # 6. planning error via the engine
    for c in CORE:
        n = f"{c}_s0"
        if have(n) and not have(f"{n}_eplan"):
            sh([py, "eplan.py", "--name", n, "--per_bucket", "120"], f"eplan {n}")

    # 7. paper
    sh([py, "make_macros.py"], "macros")
    sh([py, "make_tables.py"], "tables")
    sh([py, "figures.py"], "figures")
    sh([py, "build_paper.py"], "build")
    sh([py, "qa_gates.py"], "qa")
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
