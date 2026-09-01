"""Single sequential driver for all remaining work.

Replaces several overlapping drivers. Earlier launches used pkill, which does not
match these processes on Windows, so duplicates accumulated: two scale ladders
trained the same models concurrently on a 4 GB card and domain 2 never ran. This
runs one stage at a time, in one process, and refuses to start if another
instance is already alive.

Order:
  1. domain 2, the cross-domain check
  2. the last scale-ladder rung, with its mechanism tests
  3. planning error, rescored on action-relevant state
  4. macros, tables, figures, both PDF builds, QA gates
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RUNS = os.path.join(HERE, "..", "runs")
LOCK = os.path.join(HERE, ".final_queue.lock")
CORE = ["attn_8L256", "attnpm_8L256", "alsb_l0_8L256", "alsb_l1_8L256"]


def sh(cmd, label):
    print(f"\n=== {label}  [{time.strftime('%H:%M')}]", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  ! {label} returned {r.returncode}", flush=True)
    return r.returncode


def have(n):
    return os.path.exists(os.path.join(RES, f"{n}.json"))


def main():
    if os.path.exists(LOCK):
        age = time.time() - os.path.getmtime(LOCK)
        if age < 3600:
            print("another instance appears to be running; refusing to start")
            return
    open(LOCK, "w").write(str(os.getpid()))
    py = sys.executable

    try:
        # 1. domain 2
        sh([py, "boxes_run.py", "--conds", "attn,alsb_l1", "--seeds", "0,1,2",
            "--steps", "8000", "--n_train", "120000", "--n_probe", "4000",
            "--n_eval", "4000"], "domain 2")

        # 2. the remaining ladder rung
        rung = "attn_12L512_s0"
        if not os.path.exists(os.path.join(RUNS, f"{rung}.pt")):
            sh([py, "train.py", "--name", rung, "--steps", "12000",
                "--layers", "12", "--width", "512", "--heads", "8"], f"train {rung}")
        if not have(rung):
            sh([py, "eval_all.py", "--name", rung, "--probe_games", "3000",
                "--eval_games", "3000", "--epochs", "2"], f"eval {rung}")
        for tool, args in (("coupling.py", ["--games", "1500"]),
                           ("belief.py", ["--games", "1200"]),
                           ("patching.py", ["--n", "600", "--alpha", "0.5"])):
            tag = tool.split(".")[0]
            if not have(f"{rung}_{tag}"):
                sh([py, tool, "--name", rung] + args, f"{tag} {rung}")

        # 3. planning error on action-relevant state
        for c in CORE:
            n = f"{c}_s0"
            p = os.path.join(RES, f"{n}_eplan.json")
            if os.path.exists(p):
                d = open(p).read()
                if "squares the move touches" in d:
                    continue          # already the corrected version
                os.remove(p)
            sh([py, "eplan.py", "--name", n, "--per_bucket", "150"], f"eplan {n}")

        # 4. paper
        for step in ("make_macros.py", "make_tables.py", "figures.py",
                     "build_paper.py", "qa_gates.py"):
            sh([py, step], step)
        print("\nQUEUE COMPLETE", flush=True)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()
