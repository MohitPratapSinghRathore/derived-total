"""Driver: run the model grid sequentially (single 4GB GPU)."""
import subprocess, sys, os, time

STEPS = 12000
CORE_SEEDS = [0, 1, 2]      # >=3 seeds for the core comparison (protocol sec.6)

RUNS = []
for s in CORE_SEEDS:
    RUNS += [
        (f"attn_8L256_s{s}",    ["--layers", "8", "--width", "256", "--seed", str(s)]),
        # parameter-matched attention-only control (wider FFN absorbs ALSB params)
        (f"attnpm_8L256_s{s}",  ["--layers", "8", "--width", "256", "--ffn_mult", "4.156",
                                 "--seed", str(s)]),
        (f"alsb_l0_8L256_s{s}", ["--layers", "8", "--width", "256", "--alsb",
                                 "--d_s", "64", "--k", "4", "--lam", "0", "--seed", str(s)]),
        (f"alsb_l1_8L256_s{s}", ["--layers", "8", "--width", "256", "--alsb",
                                 "--d_s", "64", "--k", "4", "--lam", "1.0", "--seed", str(s)]),
    ]
# scaling axes, single seed
RUNS += [
    ("attn_12L256_s0", ["--layers", "12", "--width", "256", "--seed", "0"]),
    ("attn_8L384_s0",  ["--layers", "8", "--width", "384", "--heads", "8", "--seed", "0"]),
]

here = os.path.dirname(os.path.abspath(__file__))
for name, extra in RUNS:
    ck = os.path.join(here, "..", "runs", f"{name}.pt")
    if os.path.exists(ck):
        print(f"[skip] {name}", flush=True)
        continue
    print(f"[run ] {name}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(here, "train.py"),
                        "--name", name, "--steps", str(STEPS)] + extra,
                       cwd=here)
    print(f"[done] {name} rc={r.returncode} {(time.time()-t)/60:.1f} min", flush=True)
print("GRID COMPLETE", flush=True)
