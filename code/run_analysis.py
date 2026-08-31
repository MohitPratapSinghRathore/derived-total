"""Driver: wait for the grid, then run the full diagnostic on every checkpoint."""
import os, sys, glob, time, subprocess

here = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(here, "..", "runs")
RES = os.path.join(here, "..", "results")

CORE = ["attn_8L256", "attnpm_8L256", "alsb_l0_8L256", "alsb_l1_8L256"]
EXPECTED = [f"{c}_s{s}" for s in (0, 1, 2) for c in CORE] + \
           ["attn_12L256_s0", "attn_8L384_s0"]


def sh(cmd):
    print("+", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=here).returncode


def main():
    # wait for the grid
    while True:
        have = [n for n in EXPECTED if os.path.exists(os.path.join(RUNS, f"{n}.pt"))]
        if len(have) == len(EXPECTED):
            break
        print(f"waiting: {len(have)}/{len(EXPECTED)} checkpoints", flush=True)
        time.sleep(120)
    time.sleep(30)

    py = sys.executable
    for n in EXPECTED:
        if os.path.exists(os.path.join(RES, f"{n}.json")):
            print("[skip eval]", n, flush=True)
            continue
        # full controls only on seed 0 (they triple probe-fitting cost)
        extra = ["--controls"] if n.endswith("_s0") else []
        sh([py, "eval_all.py", "--name", n, "--probe_games", "3000",
            "--eval_games", "3000", "--epochs", "2"] + extra)

    # test-time ablation of the ALSB read-out
    for s in (0, 1, 2):
        n = f"alsb_l1_8L256_s{s}"
        if os.path.exists(os.path.join(RUNS, f"{n}.pt")) and \
           not os.path.exists(os.path.join(RES, f"{n}_zablate.json")):
            sh([py, "eval_all.py", "--name", n, "--ablate_z",
                "--probe_games", "3000", "--eval_games", "3000", "--epochs", "2"])

    # E_plan (Stockfish) and causal patching on seed 0 of each core condition
    for c in CORE:
        n = f"{c}_s0"
        if not os.path.exists(os.path.join(RES, f"{n}_eplan.json")):
            sh([py, "eplan.py", "--name", n, "--per_bucket", "120"])
        if not os.path.exists(os.path.join(RES, f"{n}_patch.json")):
            sh([py, "patching.py", "--name", n, "--n", "300"])

    sh([py, "aggregate.py"])
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
