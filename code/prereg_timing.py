"""Record what the repository can actually evidence about registration order.

A hash fixes a document's content. It says nothing about when the document was
written, so it cannot on its own show that a plan preceded the results it governs.
The review is right to press on this, and the honest answer is not a stronger
assurance but a precise one: the ordering evidence here is the commit history,
which is self-maintained rather than third-party timestamped, and it shows a
staged design rather than a single registration before all analysis.

So the dates are read out of git rather than asserted in prose. If the history is
the evidence, the manuscript should quote the history.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

ROOT = S.ROOT


def git(*args) -> str:
    return subprocess.run(["git", "-C", ROOT, *args],
                          capture_output=True, text=True).stdout.strip()


def first_commit(pathspec: str):
    """Oldest commit that added anything matching pathspec."""
    out = git("log", "--diff-filter=A", "--format=%H|%ad", "--date=iso",
              "--", pathspec)
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return None
    h, d = lines[-1].split("|")
    return {"commit": h[:9], "date": d}


def main() -> int:
    pre = first_commit("PREREGISTRATION.sha256")
    rec = {
        "prereg_freeze": pre,
        "first_structure": first_commit("results/attn_*_structure.json"),
        "first_natdiv": first_commit("results/attn_*_natdiv.json"),
        "timestamp_authority": "git history in a self-hosted repository",
        "third_party_timestamp": False,
    }
    if pre and rec["first_structure"]:
        rec["structure_after_freeze"] = (
            rec["first_structure"]["date"] > pre["date"])
    if pre and rec["first_natdiv"]:
        rec["natdiv_after_freeze"] = (
            rec["first_natdiv"]["date"] > pre["date"])
    for k, v in rec.items():
        print(f"  {k}: {v}")
    json.dump(rec, open(os.path.join(S.RES, "prereg_timing.json"), "w"),
              indent=1)
    print("wrote results/prereg_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
