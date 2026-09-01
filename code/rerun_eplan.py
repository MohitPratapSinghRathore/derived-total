"""Re-run planning error with action-relevant conditioning, after the queue clears.

Waits for the serialised queue so the 4 GB GPU is not shared, then rescores the
four core conditions and rebuilds the paper.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CORE = ["attn_8L256", "attnpm_8L256", "alsb_l0_8L256", "alsb_l1_8L256"]


def sh(cmd, label):
    print(f"\n=== {label}", flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def main():
    py = sys.executable
    while True:
        try:
            if "FINISH COMPLETE" in open("/tmp/finish.log", errors="ignore").read():
                break
        except FileNotFoundError:
            pass
        time.sleep(120)
    print("queue clear; rescoring planning error", flush=True)

    for c in CORE:
        n = f"{c}_s0"
        p = os.path.join(RES, f"{n}_eplan.json")
        if os.path.exists(p):
            os.remove(p)          # supersede the degenerate exact-state version
        sh([py, "eplan.py", "--name", n, "--per_bucket", "150"], f"eplan {n}")

    for step in ("make_macros.py", "make_tables.py", "figures.py",
                 "build_paper.py", "qa_gates.py"):
        sh([py, step], step)
    print("\nEPLAN RERUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
