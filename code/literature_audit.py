"""Run the admissibility check against published error decompositions.

Limitation L6 said we had audited none. This audits what we could obtain with the
numbers printed rather than estimated, which turned out to be the binding
constraint and is itself the finding.

What we looked for: a published table giving the rates of categories that
partition an error total, at three or more scales, together with an aggregate the
authors fitted or reported separately. What we found is that the first two
conditions are met often and the third almost never, for a reason that vindicates
the criticism this project makes.

Speech recognition is the mature case. Word error rate is defined as the sum of
substitutions, deletions and insertions, so the aggregate is the sum of the parts
by construction and no independent fit exists to conflict with it. Closure holds
exactly, at every row, in every table we checked. The convention that protects
these tables is precisely the repair we recommend.

The decompositions that carry an independently estimated aggregate are the ones
where the aggregate is the headline quantity and the categories are an afterthought,
which is the pattern in scaling analyses rather than in error analysis. Our own
companion manuscript is an instance, which is how this began.

Numbers below are transcribed from the cited tables. Each source records where the
figures came from, and the check verifies the partition identity as printed, which
also catches transcription error: a row whose parts do not sum to its stated total
is either a typo in our transcription or a rounding disclosure in theirs.
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S

# ---------------------------------------------------------------------------
# Source 1. Loquacious supplementary analysis, Table 9: detailed WER in terms of
# substitutions, deletions and insertions, LibriSpeech and Yodas dev sets, models
# trained on train.small. Rows vary ARCHITECTURE, not scale, so no exponent can be
# fitted; what it establishes is that the aggregate is the sum of the parts.
LOQUACIOUS_T9 = {
    "source": "Loquacious supplementary analysis, Table 9 (arXiv:2512.17915)",
    "scale_variable": None,
    "rows": [
        # label,                 sub,  del,  ins,  stated total
        ("AED BPE 1K / LS",      10.9,  1.1,  1.7, 13.7),
        ("CTC BPE 128 / LS",      8.9,  1.3,  1.1, 11.3),
        ("CTC phonemes 80 / LS",  8.4,  1.0,  1.3, 10.6),
        ("RNN-T BPE 128 / LS",   11.3,  1.3,  1.3, 13.9),
        ("mRNN-T / LS",           8.4,  1.3,  1.0, 10.7),
        ("RNN-T phonemes 80/LS",  8.3,  1.3,  1.2, 10.8),
        ("AED BPE 1K / Yodas",    9.9,  4.5,  3.9, 18.3),
        ("CTC BPE 128 / Yodas",   8.3,  6.4,  2.1, 16.8),
        ("CTC phonemes 80/Yodas", 8.1,  5.1,  2.4, 15.6),
        ("RNN-T BPE 128 / Yodas", 9.6,  5.4,  3.5, 18.5),
        ("mRNN-T / Yodas",        7.0,  6.7,  2.2, 15.9),
    ],
}


def check_identity(table, decimals=1):
    """Do the printed parts sum to the printed total, row by row?

    The tolerance has to account for every printed quantity being rounded
    independently, not just one. With k parts and a total each rounded to d
    decimals, the sum of the parts can differ from the printed total by up to
    (k+1)/2 units in the last place. Using half a unit instead, as we first did,
    flags ordinary rounding as a closure failure: it reported a row of
    Loquacious Table 9 as broken when 8.4+1.0+1.3 against a stated 10.6 is
    exactly what rounding three numbers to one decimal can produce.
    """
    out = []
    for label, *parts_and_total in table["rows"]:
        *parts, total = parts_and_total
        tol = (len(parts) + 1) / 2 * 10 ** -decimals
        s = sum(parts)
        out.append({"row": label, "parts_sum": round(s, 4), "stated": total,
                    "residual": round(s - total, 4), "tol": round(tol, 4),
                    "ok": abs(s - total) <= tol + 1e-9})
    return out


def main():
    report = {"sources": []}

    rows = check_identity(LOQUACIOUS_T9)
    n_ok = sum(r["ok"] for r in rows)
    worst = max(abs(r["residual"]) for r in rows)
    report["sources"].append({
        "source": LOQUACIOUS_T9["source"],
        "n_rows": len(rows),
        "n_closed": n_ok,
        "max_abs_residual": worst,
        "aggregate_independently_fitted": False,
        "admissible": bool(n_ok == len(rows)),
        "note": "WER is defined as the sum of its parts, so closure is exact by "
                "construction and no independent aggregate exists to conflict "
                "with it. Rows vary architecture, not scale, so no exponent is "
                "fitted and no breakdown scale applies.",
        "detail": rows,
    })

    # ------------------------------------------------------------------ ours
    ours = S.ladder_rows()
    import admissibility as A
    report["sources"].append({
        "source": "This project's ladder (companion manuscript)",
        "n_rows": len(ours),
        "n_closed": 0,
        "aggregate_independently_fitted": True,
        "admissible": False,
        "note": "The aggregate illegal-move rate was fitted as its own power law "
                "alongside the category curves, which is the configuration that "
                "fails. Breakdown scale reported in admissibility.json.",
    })

    report["summary"] = {
        "sources_examined": len(report["sources"]),
        "with_independent_aggregate": sum(
            s["aggregate_independently_fitted"] for s in report["sources"]),
        "inadmissible": sum(not s["admissible"] for s in report["sources"]),
    }

    json.dump(report, open(os.path.join(S.RES, "literature_audit.json"), "w"),
              indent=1)

    for s in report["sources"]:
        print(f"\n{s['source']}")
        print(f"  rows {s['n_rows']}  aggregate fitted independently: "
              f"{s['aggregate_independently_fitted']}  admissible: {s['admissible']}")
        if "max_abs_residual" in s:
            print(f"  partition identity holds in {s['n_closed']}/{s['n_rows']} "
                  f"rows, worst residual {s['max_abs_residual']:.3f} WER points")
    print(f"\n{report['summary']}")


if __name__ == "__main__":
    main()
