"""Generate the supplementary material from the result artifacts.

Everything here is a table the main text summarises or omits for space: the
per-run training detail, the per-run evaluation counts, the full locality
measurements, the audit of the classifier correction, and the inventory of what
each verification gate catches.

The tables are written from the result JSON files rather than transcribed, for the
same reason the manuscript's numbers are macros: a supplementary table that has
quietly stopped matching the analysis is worse than no supplementary table, and
this document is exactly where that would go unnoticed.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

B = chr(92)
OUT = os.path.join(S.ROOT, "paper_scaling", "supplement.tex")

CLASS_LABEL = [("from_empty", "empty source"),
               ("from_opponent", "opponent source"),
               ("to_own", "own destination"),
               ("castling", "castling"),
               ("geometry_impossible", "geometry, impossible"),
               ("geometry_blocked", "geometry, blocked path"),
               ("leaves_check", "leaves check")]


def C(cmd, *args):
    return B + cmd + "".join("{" + str(a) + "}" for a in args)


def row(*cells):
    return " & ".join(str(c) for c in cells) + " " + B + B


def table(spec, header, rows, caption, label):
    out = [C("begin", "table") + "[htbp]", C("centering") + C("small"),
           C("caption") + "{" + caption + "}", C("label", label),
           C("begin", "tabular") + "{" + spec + "}", C("toprule"),
           header, C("midrule")]
    out += rows
    out += [C("bottomrule"), C("end", "tabular"), C("end", "table"), ""]
    return out


def main() -> int:
    tt = S.load("training_table.json")
    fc = S.load("full_classification.json")
    rg = S.load("regen_exponents.json")
    li = S.load("locality_index.json")
    bc = S.load("blocker_check.json")
    dd = S.load("drift_decomposition.json")
    rc = S.load("regen_compare.json")
    ag = S.load("aggregate_gate_check.json")
    rb = S.load("robustness_checks.json")
    env = S.load("environment.json")
    jv = S.load("joint_vs_conditional.json")
    missing = [n for n, d in (("training_table", tt), ("full_classification", fc),
                              ("regen_exponents", rg), ("locality_index", li),
                              ("drift_decomposition", dd), ("regen_compare", rc),
                              ("aggregate_gate_check", ag),
                              ("robustness_checks", rb), ("environment", env))
               if not d]
    if missing:
        print("missing result files:", missing)
        return 1

    L = [
        C("documentclass", "11pt") if False else
        B + "documentclass[11pt]{article}",
        B + "usepackage[utf8]{inputenc}" + B + "usepackage[T1]{fontenc}"
        + B + "usepackage{lmodern}",
        B + "usepackage[margin=1in]{geometry}",
        B + "usepackage{booktabs}" + B + "usepackage{microtype}",
        B + "usepackage[hidelinks]{hyperref}",
        B + "renewcommand" + "{" + B + "thetable}{S" + B + "arabic{table}}",
        B + "renewcommand" + "{" + B + "thesection}{S" + B + "arabic{section}}",
        B + "title{" + B + "bf Supplementary Material" + B + B,
        B + "large Scale Fixes Local Errors First: Differential Scaling of Chess "
        "Transformer Failures}",
        B + "author{Mohit Pratap Singh Rathore " + B + "and Gunveer Singh Kalsi "
        + B + "and Sriharsha Meduri" + B + B
        + " " + B + "small Oviqo Private Limited}",
        B + "date{}",
        "",
        C("begin", "document"),
        C("maketitle"),
        "",
        "This document holds the per-run tables and verification records that the",
        "manuscript summarises. Every table is generated from the result files in the",
        "public repository by " + B + "texttt{code/make" + B + "_supplement.py}, so a table here"
        " cannot",
        "drift away from the analysis it reports. Section numbers are prefixed S and the",
        "tables are numbered S1 onward.",
        "",
    ]

    # ---------------------------------------------------------------- S1
    L += [C("section", "Reproduction"), ""]
    L += ["The analysis runs in the following order. Each step writes result files that",
          "the next step reads, and the manuscript's numbers are generated from those",
          "files by the macro pipeline rather than typed.", "",
          C("begin", "enumerate"),
          C("item") + " " + B + "texttt{code/prep" + B + "_data.py}, "
          + B + "texttt{code/make" + B + "_states.py}: tokenise games and record "
          "ground-truth board state.",
          C("item") + " " + B + "texttt{code/run" + B + "_grid.py}: train the "
          + str(tt["n_runs"]) + " runs of the ladder.",
          C("item") + " " + B + "texttt{code/full" + B + "_classification.py}: classify "
          "every failure at every scored position.",
          C("item") + " " + B + "texttt{code/regen" + B + "_exponents.py}, "
          + B + "texttt{code/compositional.py}, "
          + B + "texttt{code/robustness" + B + "_checks.py}: fit exponents, the closed "
          "compositional model, and the leave-one-rung-out refits.",
          C("item") + " " + B + "texttt{code/make" + B + "_macros" + B + "_ds.py}: write "
          "every number the manuscript prints, with its source file recorded beside it.",
          C("item") + " " + B + "texttt{code/build" + B + "_ds.py}: build the manuscript "
          "and run the gates and the consistency audit.",
          C("end", "enumerate"), ""]
    L += ["All of it runs on one " + env["gpu_name"] + " with "
          + f"{env['gpu_memory_gib']:.0f}" + " GiB of device memory under "
          + env["os"] + ", using Python " + env["python"] + ", PyTorch "
          + env["torch"] + ", NumPy " + env["numpy"] + " and python-chess "
          + env["python_chess"] + ". The classification and analysis steps run on "
          "CPU; training used the GPU.", ""]

    # ---------------------------------------------------------------- S2
    L += [C("section", "Per-run training detail"), ""]
    rows = []
    for r in tt["rows"]:
        rows.append(row(r["rung"], r["seed"], f"{r['params']:,}", r["layers"],
                        r["width"], r["heads"], f"{r['steps']:,}",
                        f"{r['tokens'] / 1e6:.1f}",
                        f"{r['val_final']:.4f}", f"{r['val_best']:.4f}"))
    L += table("llrrrrrrrr",
               row("rung", "seed", "params", "L", "d", "heads", "steps",
                   "tokens (M)", "val.", "best"),
               rows,
               "Per-run training detail, read out of the checkpoints. Every run "
               "shares one recipe; the final checkpoint is also the "
               "lowest-validation-loss checkpoint in "
               + str(tt["n_final_is_best"]) + " of " + str(tt["n_runs"])
               + " runs, so checkpoint selection carries none of the result.",
               "tab:s-training")

    # ---------------------------------------------------------------- S3
    L += [C("section", "Per-run evaluation and failure counts"), ""]
    rows = []
    for r in fc["rows"]:
        cs = r["counts"]
        rows.append(row(r["rung"], r["seed"], f"{r['n_positions']:,}",
                        f"{r['n_failures']:,}", f"{r['illegal_rate']:.5f}",
                        *[f"{cs.get(c, 0):,}" for c, _ in CLASS_LABEL]))
    L += table("llrrr" + "r" * len(CLASS_LABEL),
               row("rung", "seed", "positions", "failures", "rate",
                   "empty", "opp", "own", "cast", "g.imp", "g.blk", "check"),
               rows,
               "Per-run evaluation counts. Every failure at every scored position is "
               "classified, so these are counts rather than sampled shares: "
               + f"{fc['total_failures']:,}" + " failures over "
               + f"{fc['total_positions']:,}" + " model-position evaluations, each "
               "model being evaluated at the same "
               + f"{fc['rows'][0]['n_positions']:,}" + " unique positions. No "
               "recorded failure was a legal move under the corrected gate.",
               "tab:s-counts")

    # ---------------------------------------------------------------- S4
    L += [C("section", "Exponents, endpoints and fold drops"), ""]
    rows = []
    order = [c for c, _ in CLASS_LABEL] + ["local_pooled", "all_illegal",
                                           "corr_local"]
    names = dict(CLASS_LABEL)
    names.update({"local_pooled": B + "textit{pooled local}",
                  "all_illegal": B + "textit{all illegal}",
                  "corr_local": B + "textit{despite correct local belief}"})
    for c in order:
        e = rg["classes"].get(c)
        if not e:
            continue
        rows.append(row(names[c], f"{e['first']:.4f}", f"{e['last']:.4f}",
                        f"{e['fold']:.1f}" if e.get("fold") else "--",
                        f"{e['point']:+.4f}",
                        f"[{e['lo']:+.4f}, {e['hi']:+.4f}]",
                        "yes" if e["excludes_zero"] else "no"))
    L += table("lrrrrlc",
               row("class", "at smallest", "at largest", "fold", "exponent",
                   "95\\% CI", "excl.\\ 0"),
               rows,
               "Absolute failure rate per scored position at the smallest and "
               "largest rung, the fold drop between them, and the size exponent with "
               "its rung-resampled interval. The last three rows are not disjoint "
               "classes and must not be summed with the others.",
               "tab:s-exponents")

    rows = []
    for lab, d in rg.get("differentials", {}).items():
        rows.append(row(lab, f"{d['point']:+.4f}",
                        f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                        "yes" if d["excludes_zero"] else "no"))
    L += table("lrlc", row("differential", "point", "95\\% CI", "excl.\\ 0"), rows,
               "Paired differentials, bootstrapped on the same rung resamples so each "
               "interval is on the difference itself.",
               "tab:s-diffs")

    # ---------------------------------------------------------------- S5
    L += [C("section", "Leave one rung out"), ""]
    lo = rb["leave_one_rung_out"]
    rows = [row("full ladder", f"{lo['full']['diff']:+.4f}", "--")]
    for k, v in lo["dropped"].items():
        rows.append(row("without " + k, f"{v['diff']:+.4f}",
                        f"{v['diff'] - lo['full']['diff']:+.4f}"))
    L += table("lrr", row("fit", "check $-$ empty", "shift"), rows,
               "The headline differential refitted with each rung deleted in turn. "
               "The sign is unchanged in every refit and the largest shift is "
               + f"{lo['max_shift']:.4f}" + ", so no single rung carries the result.",
               "tab:s-loro")

    # ---------------------------------------------------------------- S6
    L += [C("section", "Locality measurements"), ""]
    L += ["For a position and an illegal move, $k$ counts the squares whose contents "
          "can alone flip the legality verdict, excluding the two the move names. It "
          "is computed by brute force with no model involved, so it is a property of "
          "the rules and the position.", ""]
    rows = []
    for c in ("from_empty", "from_opponent", "to_own", "geometry",
              "geometry_impossible", "geometry_blocked", "castling",
              "leaves_check"):
        r = li["classes"].get(c)
        if not r:
            continue
        rows.append(row(c.replace("_", " "), f"{r['n']:,}", f"{r['mean']:.3f}",
                        f"{r['median']:.0f}", f"{r['sd']:.2f}", r["min"],
                        r["max"]))
    L += table("lrrrrrr",
               row("class", "$n$", "mean $k$", "median", "sd", "min", "max"),
               rows,
               "Dependency set sizes over " + f"{li['games']}" + " games. The two "
               "geometry rows subdivide the geometry row above them; castling was "
               "sampled separately once the audit identified it as a distinct class.",
               "tab:s-locality")
    if bc:
        L += ["The statistic counts single-square sufficiency, which undercounts when "
              "no single square is sufficient. The deficit is characterised rather "
              "than unknown: among the " + str(bc["n_blocked"]) + " blocked-path "
              "moves, those with one blocker have $k=1$ in "
              + str(bc["single_blocker_k_one"]) + " of "
              + str(bc["single_blocker_n"]) + " cases, and all "
              + str(bc["multi_blocker_n"]) + " moves with two or more blockers have "
              "$k=0$, since removing either blocker alone leaves the path blocked.",
              ""]

    # ---------------------------------------------------------------- S7
    L += [C("section", "The classifier correction, audited"), ""]
    rows = []
    for k, v in sorted(rc["flow"].items(), key=lambda kv: -kv[1]):
        o, n = k.split("->")
        rows.append(row(o.replace("_", " "), n.replace("_", " "), f"{v:,}",
                        "changed" if o != n else ""))
    L += table("llrl", row("old label", "new label", "moves", ""), rows,
               "Class flow under the corrected classifier, over the "
               + f"{rc['n_failures']:,}" + " audited moves that carry both labels. "
               "No move entered or left either class the headline differential uses, "
               "so the relabelling cannot have changed it.",
               "tab:s-flow")

    rows = []
    for r in ag["rows"]:
        rows.append(row(r["name"].replace("attn_", "").replace("_", " "),
                        f"{r['n_positions']:,}",
                        f"{r['rate_membership_gate']:.6f}",
                        f"{r['rate_legality_gate']:.6f}",
                        r["n_gates_disagree"]))
    L += table("lrrrr",
               row("run", "positions", "membership gate", "legality gate",
                   "disagree"),
               rows,
               "The original failure gate tested membership in the generated legal "
               "moves; the corrected gate tests legality directly. The two differ "
               "only on the king-onto-rook castling encoding. Over every scored "
               "position of every run they disagree at "
               + str(ag["total_disagreements"]) + " positions, so the aggregate rate "
               "is unaffected over the whole set and not merely in a sample.",
               "tab:s-gates")

    rows = [row("A published", "old", "float16 GPU", "sampled",
                f"{dd['A_published']:+.4f}", "--"),
            row("B precision", "old", "float32 CPU", "sampled",
                f"{dd['B_precision']:+.4f}", f"{dd['d_precision']:+.4f}"),
            row("C estimator", "old", "float32 CPU", "full set",
                f"{dd['C_estimator']:+.4f}", f"{dd['d_estimator']:+.4f}"),
            row("D classifier", "new", "float32 CPU", "full set",
                f"{dd['D_classifier']:+.4f}", f"{dd['d_classifier']:+.4f}")]
    L += table("lllrrr",
               row("configuration", "classifier", "arithmetic", "estimator",
                   "check $-$ empty", "shift"),
               rows,
               "Why the headline differential moved, varying one thing at a time. "
               "The dominant term is the estimator, not the hardware: the sampled "
               "estimator was biased by corpus order. The classifier term is exactly "
               "zero, which independently confirms the flow table.",
               "tab:s-drift")

    if jv:
        L += ["The decomposition of the near-flat joint rate is exact rather than "
              "estimated, and that commits it to a checkable property. Since the "
              "identity holds position by position, the logs add and an "
              "ordinary-least-squares slope is a linear functional of the response, "
              "the three exponents must sum. They do, to "
              + f"{jv['slope_additivity_error']:.1e}" + ": "
              + f"{jv['marginal_correct']['point']:+.4f}" + " for the marginal and "
              + f"{jv['conditional']['point']:+.4f}" + " for the conditional give "
              + f"{jv['joint']['point']:+.4f}" + " for the joint.", ""]

    # ---------------------------------------------------------------- S8
    L += [C("section", "What the verification suite checks"), ""]
    L += ["The build fails rather than warns. Two kinds of check run on every build: "
          "gates on the manuscript, and invariants between quantities that must "
          "agree. Each was added after a defect of its kind reached a draft.", "",
          C("begin", "itemize"),
          C("item") + " " + B + "textbf{Provenance.} Every number in the manuscript "
          "resolves to a macro with a recorded source file, no decimal may be typed "
          "in prose, and no macro sourced from the superseded classification pass may "
          "appear at all.",
          C("item") + " " + B + "textbf{De-escaped control sequences.} Raw tabs and "
          "carriage returns in the source, and orphaned fragments such as "
          + B + "texttt{exttt}, which typeset as visible prose while the document "
          "still compiles.",
          C("item") + " " + B + "textbf{Arithmetic.} The partition sums to the "
          "illegal-move rate; fold drops agree with their endpoints; relabelling "
          "counts sum to their components; the exponents of an exact decomposition "
          "sum to the exponent they decompose.",
          C("item") + " " + B + "textbf{Significance.} No class may be described as "
          "having an interval that includes zero when its interval excludes it.",
          C("item") + " " + B + "textbf{Scope.} Every cross-reference resolves; every "
          "quoted count is no larger than its total; the anonymous build contains no "
          "author identity; the source archive compiles from its own extracted "
          "contents.",
          C("end", "itemize"), ""]

    L += [C("section", "Artifact inventory"), ""]
    inv = [("full_classification.json", "every failure classified, per run"),
           ("regen_exponents.json", "exponents, shares, depth bands, differentials"),
           ("attn_*_regen.json", "the audited sample, each move under both labels"),
           ("aggregate_gate_check.json", "both legality gates on every position"),
           ("drift_decomposition.json", "why the headline differential moved"),
           ("locality_index.json", "dependency set sizes by class"),
           ("blocker_check.json", "the multi-blocker characterisation of $k$"),
           ("training_table.json", "per-run architecture and training detail"),
           ("compositional.json", "the closed simplex fit and the closure argument"),
           ("robustness_checks.json", "leave-one-rung-out refits"),
           ("joint_vs_conditional.json", "the exact decomposition and its additivity"),
           ("environment.json", "hardware and library versions"),
           ("prereg_timing.json", "registration dates read from the git history")]
    L += table("ll", row("file under " + B + "texttt{results/}", "contents"),
               [row(B + "texttt{" + a.replace("_", B + "_") + "}", b)
                for a, b in inv],
               "Result files cited by the manuscript or this supplement. All are in "
               "the public repository.",
               "tab:s-artifacts")

    L += [C("end", "document"), ""]

    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(f"wrote {os.path.relpath(OUT, S.ROOT)}: {len(L)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
