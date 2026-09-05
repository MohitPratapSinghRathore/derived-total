"""Generate paper2's tables from the second paper's result files."""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper2")


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def rollout_table():
    a = load("attn_8L256_s0_trans_rollout")
    b = load("attn_8L256_s0_trans12_rollout")
    if not (a and b):
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Is a rolled-forward state better than the model's current",
         "one? Board accuracy on occupied squares, decoded from the state",
         "subspace at the same positions. The one-step operator compounds error",
         "and lands below even the unrolled stale state. Training on rollouts",
         "recovers most of that, and still does not exceed the state it was",
         "meant to repair.}",
         "\\label{tab:rollout}", "\\begin{tabular}{lcccccc}", "\\toprule",
         "& \\multicolumn{3}{c}{one-step operator}"
         " & \\multicolumn{3}{c}{multi-step operator} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Ply & current & rolled & stale & current & rolled & stale \\\\",
         "\\midrule"]
    for i, (lo, hi) in enumerate(a["buckets"]):
        L.append(f"{lo}--{hi} & "
                 f"\\textbf{{{a['occ_acc']['now'][i]:.3f}}} & "
                 f"{a['occ_acc']['rolled'][i]:.3f} & {a['occ_acc']['stale'][i]:.3f} & "
                 f"\\textbf{{{b['occ_acc']['now'][i]:.3f}}} & "
                 f"{b['occ_acc']['rolled'][i]:.3f} & {b['occ_acc']['stale'][i]:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def repair_table():
    d = load("attn_8L256_s0_refresh")
    if not d:
        return []
    modes = [("noop", "no-op"), ("repair", "repair"), ("stale", "stale"),
             ("random", "random")]
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Test-time injection of the repaired state. Illegal top-1",
         "move rate by depth, with no-op giving the base rate, stale injecting",
         "the earlier state without rolling it forward, and random a",
         "norm-matched direction. Nothing separates them.}",
         "\\label{tab:repair}", "\\begin{tabular}{lcccc}", "\\toprule",
         "Ply & " + " & ".join(m[1] for m in modes) + " \\\\", "\\midrule"]
    for i, (lo, hi) in enumerate(d["buckets"]):
        L.append(f"{lo}--{hi} & "
                 + " & ".join(f"{d['illegal'][m[0]][i]:.4f}" for m in modes)
                 + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def detect_table():
    d = load("attn_8L256_s0_drift_detect")
    if not d:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Oracle-free drift detection. Illegal-move rate among",
         "positions retained at each coverage, on held-out games, with the",
         "combination fitted on training games only. Consistency error is the",
         "weaker signal alone and is not redundant with the model's own",
         "confidence.}",
         "\\label{tab:detect}", "\\begin{tabular}{lccc}", "\\toprule",
         "Coverage & model confidence & consistency error & combined \\\\",
         "\\midrule"]
    for cov, row in d["coverage"].items():
        pct = int(round(float(cov) * 100))
        L.append(f"{pct}\\% & {row['model_confidence']:.4f} & "
                 f"{row['consistency_error']:.4f} & "
                 f"\\textbf{{{row['combined']:.4f}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def main():
    for name, fn in (("rollout", rollout_table), ("repair", repair_table),
                     ("detect", detect_table)):
        block = fn()
        with open(os.path.join(OUT, f"tab_{name}.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(block) if block else "")
    print("wrote paper2 tables")


if __name__ == "__main__":
    main()
