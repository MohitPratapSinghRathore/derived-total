"""Serialised queue for everything that remains.

The GPU has 4 GB and a single job fills most of it, so concurrent stages cause
out-of-memory failures and severe thrashing. Everything below runs one at a time,
in dependency order, and each stage skips work that already exists.

  1. wait for the chess analysis to finish
  2. domain 2 across all conditions and seeds
  3. the scale ladder, with the mechanism tests at every rung
  4. macros, tables, figures, both PDF builds, QA gates
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ANALYSIS_LOG = "/tmp/analysis.log"


def sh(cmd, label):
    print(f"\n=== {label}", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  ! {label} returned {r.returncode}", flush=True)
    return r.returncode


def wait_for_analysis():
    while True:
        try:
            if "ANALYSIS COMPLETE" in open(ANALYSIS_LOG, errors="ignore").read():
                print("chess analysis complete", flush=True)
                return
        except FileNotFoundError:
            pass
        time.sleep(120)


def main():
    py = sys.executable
    wait_for_analysis()

    # 2. domain 2, now that it reaches a measurable regime
    # Only the two informative conditions. Domain 2 exists to test whether the
    # MECHANISM generalises, not to repeat the architecture comparison, which the
    # chess arm settled with three seeds. attn asks whether decay, coupling and
    # belief consistency replicate; alsb_l1 asks whether availability without use
    # replicates. attnpm and alsb_l0 would cost hours to re-answer a settled
    # question.
    sh([py, "boxes_run.py", "--conds", "attn,alsb_l1",
        "--seeds", "0,1,2", "--steps", "8000", "--n_train", "120000",
        "--n_probe", "4000", "--n_eval", "4000"], "domain 2")

    # 3. scale ladder
    sh([py, "scale_ladder.py"], "scale ladder")

    # 4. paper
    for step in ("make_macros.py", "make_tables.py", "figures.py",
                 "build_paper.py", "qa_gates.py"):
        sh([py, step], step)
    print("\nFINISH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
