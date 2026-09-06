"""Can S5 detect anything, given what the memory fraction actually is?

S5 was designed to evaluate four training arms by their effect on the memory
fraction: three sizes, three seeds, 36 runs. That design was specified when the
memory fraction was assumed near 0.34. It measured 0.114, with the state-repair
component that S5 targets smaller still, so the design has to be re-checked
against the quantity it will actually move.

Two sources of noise, and both are estimated from data rather than assumed.

  within-run   sampling error of a proportion over roughly 700 failures that are
               clustered in ~58 games. Taken from the S2 interval directly, since
               that interval is already game-clustered.

  between-seed spread of the same kind of quantity across seeds at fixed
               architecture. Estimated from the 18-run ladder, where three seeds
               per rung give a direct read on how much a fraction moves when only
               the seed changes.

The question is the minimum effect on the memory fraction that S5 could detect,
and whether an intervention is plausibly that large.

The answer contradicted the expectation this file was written with. An earlier
version printed a conclusion that the design was underpowered -- written before
the numbers existed, which is the bias the pre-registration discipline exists to
prevent. It is not underpowered: seed spread is small and the design detects a
fifth of the memory fraction. The objection to S5 is importance rather than
power, since the component it targets is a small share of failures. The
pre-written verdict was replaced by one that follows from the output.
"""
import os, json, glob
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RUNGS = ["6L192", "8L256", "12L256", "8L384", "12L384", "12L512"]
SEEDS = [0, 1, 2]


def load(n, suf):
    p = os.path.join(RES, f"{n}{suf}")
    return json.load(open(p)) if os.path.exists(p) else None


def between_seed_sd():
    """How much does a failure-composition fraction move with the seed alone?"""
    out = {}
    for key, get in (
        ("policy_share",
         lambda n: (load(n, "_natdiv.json") or {}).get("policy_share", {}).get("mean")),
        ("leaves_check_share",
         lambda n: ((load(n, "_structure.json") or {}).get("overall", {})
                    .get("leaves_check", {}) or {}).get("mean")),
        ("from_empty_share",
         lambda n: ((load(n, "_structure.json") or {}).get("overall", {})
                    .get("from_empty", {}) or {}).get("mean")),
        ("illegal_rate",
         lambda n: (load(n, "_natdiv.json") or {}).get("illegal_rate")),
    ):
        sds, means = [], []
        for r in RUNGS:
            v = [get(f"attn_{r}_s{s}") for s in SEEDS]
            v = [x for x in v if x is not None]
            if len(v) == 3:
                sds.append(np.std(v, ddof=1))
                means.append(np.mean(v))
        if sds:
            out[key] = {"sd": float(np.mean(sds)),
                        "mean": float(np.mean(means)),
                        "rel": float(np.mean(sds) / max(np.mean(means), 1e-9))}
    return out


def mde(sigma, n_per_arm, alpha=0.05, power=0.80):
    """Minimum detectable difference between two arms, two-sided."""
    from math import sqrt
    z_a, z_b = 1.959964, 0.841621        # normal quantiles
    return (z_a + z_b) * sigma * sqrt(2.0 / n_per_arm)


def main():
    d = load("attn_8L256_s0", "_decompose.json")
    mem = d["shares"]["memory"]["mean"]
    mem_lo, mem_hi = d["shares"]["memory"]["ci_lo"], d["shares"]["memory"]["ci_hi"]
    memmix = d["memory_plus_mixed"]
    within = (mem_hi - mem_lo) / 3.92

    print("what S5's metric actually is")
    print(f"  memory fraction            {mem:.4f} [{mem_lo:.4f}, {mem_hi:.4f}]")
    print(f"  memory + mixed             {memmix:.4f}")
    print(f"  within-run SE (clustered)  {within:.4f}")

    bs = between_seed_sd()
    print(f"\nbetween-seed spread at fixed architecture, from the 18-run ladder")
    print(f"{'quantity':>22} {'mean':>9} {'seed SD':>9} {'relative':>9}")
    for k, v in bs.items():
        print(f"{k:>22} {v['mean']:9.4f} {v['sd']:9.4f} {v['rel']:9.3f}")
    rel = float(np.median([v["rel"] for v in bs.values()]))
    between = rel * mem
    print(f"\n  median relative seed spread {rel:.3f}")
    print(f"  implied between-seed SD of the memory fraction "
          f"{between:.4f}")

    sigma = float(np.sqrt(within ** 2 + between ** 2))
    print(f"  total per-run SD           {sigma:.4f}")

    print(f"\nminimum detectable difference between two arms "
          f"(alpha 0.05, power 0.80)")
    print(f"{'design':>34} {'n/arm':>6} {'MDE':>9} {'as % of memory':>16}")
    for label, n in (("per size, 3 seeds", 3),
                     ("pooled over 3 sizes, 9 runs", 9),
                     ("if seeds doubled to 6, per size", 6),
                     ("pooled, 6 seeds x 3 sizes", 18)):
        m = mde(sigma, n)
        print(f"{label:>34} {n:6d} {m:9.4f} {100*m/mem:15.0f}%")

    m9 = mde(sigma, 9)
    ill = 0.1950                # illegal top-1 rate at ply >= 20, 8L256_s0
    abs_now, abs_mde = mem * ill, m9 * ill
    print(f"\nreading")
    print(f"  Power is adequate. Pooling 9 runs per arm detects a shift of")
    print(f"  {m9:.4f} in the memory fraction, {100*m9/mem:.0f}% of the fraction "
          f"itself,")
    print(f"  which is a reasonable bar for a targeted intervention. Between-seed")
    print(f"  spread is small, relative SD {rel:.3f}, so noise is dominated by")
    print(f"  within-run sampling: more scored positions buy more than more seeds.")
    print(f"\n  The objection is importance, not power. The memory fraction is")
    print(f"  {mem:.4f} of failures and failures are {ill:.4f} of positions, so")
    print(f"  state-attributable failures are {abs_now:.4f} of all positions. A")
    print(f"  just-detectable change moves the illegal rate by {abs_mde:.4f}, from")
    print(f"  {ill:.4f} to {ill - abs_mde:.4f}. Removing the memory bucket entirely")
    print(f"  would move it to {ill - abs_now:.4f}.")
    print(f"\n  So S5 can measure what it set out to measure. What it cannot show")
    print(f"  is a large behavioural gain, because the component these arms")
    print(f"  target is a small share of what goes wrong.")

    json.dump({"memory_fraction": mem, "within_run_se": within,
               "between_seed_sd": between, "total_sd": sigma,
               "mde_per_size_3seeds": mde(sigma, 3),
               "mde_pooled_9": m9,
               "relative_seed_spread": rel,
               "between_seed_detail": bs},
              open(os.path.join(RES, "s5_power.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
