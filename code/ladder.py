"""Replicate the validated results across all six rungs and three seeds.

Everything here is inference against probes that were already fitted for Paper 1,
so no training and no probe refitting is needed. The three results worth
replicating are the ones that survived scrutiny:

  S1 repair       correct repair against wrong-target and against irrelevant,
                  entropy-matched, which is the causal claim
  structure       what kind of wrong the failures are, which named the mechanism
  natural         divergence and the policy share, which set the ceiling

The whole-board repair arm is deliberately NOT replicated. Amendment A3 withdrew
it because the edit never installs a board, and replicating a broken measurement
eighteen times would only make it look authoritative.

A lockfile guards the GPU. During Paper 1 two drivers ran concurrently on this
machine, trained the same models twice and silently corrupted a ladder, and the
Windows process names did not match what was being killed. One driver at a time.
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
LOCK = os.path.join(ROOT, ".ladder.lock")
RUNGS = ["6L192", "8L256", "12L256", "8L384", "12L384", "12L512"]
SEEDS = [0, 1, 2]

JOBS = [
    ("repair", ["--n", "600", "--games", "2500"], "_repair.json"),
    ("structure", ["--n", "900", "--games", "2500"], "_structure.json"),
    ("natural_divergence", ["--eval_games", "600"], "_natdiv.json"),
]


def main():
    if os.path.exists(LOCK):
        age = time.time() - os.path.getmtime(LOCK)
        print(f"lockfile present ({age/60:.1f} min old): {LOCK}")
        print("another ladder driver is running. refusing to start a second.")
        sys.exit(1)
    open(LOCK, "w").write(f"{os.getpid()}\n{time.time()}\n")
    t0 = time.time()
    done, failed, skipped = [], [], []
    try:
        names = [f"attn_{r}_s{s}" for r in RUNGS for s in SEEDS]
        total = len(names) * len(JOBS)
        i = 0
        for name in names:
            for script, extra, suffix in JOBS:
                i += 1
                out = os.path.join(RES, f"{name}{suffix}")
                if os.path.exists(out):
                    skipped.append(f"{name}/{script}")
                    print(f"[{i}/{total}] skip {name} {script} (exists)",
                          flush=True)
                    continue
                cmd = [sys.executable, os.path.join(HERE, f"{script}.py"),
                       "--name", name] + extra
                el = time.time() - t0
                print(f"[{i}/{total}] {name} {script}  "
                      f"(elapsed {el/60:.0f} min)", flush=True)
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                   text=True)
                if r.returncode != 0 or not os.path.exists(out):
                    failed.append(f"{name}/{script}")
                    print(f"    FAILED rc={r.returncode}", flush=True)
                    print("    " + (r.stderr or "")[-600:], flush=True)
                else:
                    done.append(f"{name}/{script}")
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)

    print(f"\nladder finished in {(time.time()-t0)/60:.0f} min")
    print(f"  completed {len(done)}, skipped {len(skipped)}, "
          f"failed {len(failed)}")
    if failed:
        print("  failures: " + ", ".join(failed))
    json.dump({"done": done, "failed": failed, "skipped": skipped,
               "minutes": (time.time() - t0) / 60},
              open(os.path.join(RES, "ladder_status.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
