"""Check the corpus before spending any compute on it.

The preregistration says: verify on 100 rows that pile_set_name exists and lists
the expected sources, and if it does not, stop and report rather than substituting
a different dataset. This does that and nothing else. It loads no model.
"""
import os, sys, json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VAL_URL = "hf://datasets/monology/pile-uncopyrighted/val.jsonl.zst"

# The sources the full Pile contains. The uncopyrighted mirror is expected to be
# missing several; which ones are missing is recorded rather than worked around.
FULL_PILE = [
    "ArXiv", "BookCorpus2", "Books3", "DM Mathematics", "Enron Emails",
    "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)", "HackerNews",
    "NIH ExPorter", "OpenSubtitles", "OpenWebText2", "PhilPapers",
    "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    "USPTO Backgrounds", "Ubuntu IRC", "Wikipedia (en)", "YoutubeSubtitles",
]


def main(n_rows=100):
    from datasets import load_dataset

    # The mirror ships the validation data as val.jsonl.zst at the repo root and
    # does not register it as a named split, so the builder reports only "train".
    # Pointing at the file is the same data the preregistration named, streamed the
    # same way; it is not a substitution and the path is recorded in the output.
    ds = load_dataset("json", data_files=VAL_URL, split="train", streaming=True)
    seen, missing_field, examples = Counter(), 0, {}
    for i, rec in enumerate(ds):
        if i >= n_rows:
            break
        meta = rec.get("meta")
        if not isinstance(meta, dict) or "pile_set_name" not in meta:
            missing_field += 1
            if missing_field == 1:
                print(f"  first record without pile_set_name: keys={list(rec)} "
                      f"meta={meta!r}")
            continue
        src = meta["pile_set_name"]
        seen[src] += 1
        examples.setdefault(src, rec["text"][:70].replace("\n", " "))

    print(f"inspected {n_rows} rows; {missing_field} lacked pile_set_name")
    if missing_field:
        print("\nSTOP: the source label is not present on every record. The "
              "preregistration forbids substituting a dataset, so the run does "
              "not proceed.")
        return 1

    print(f"\n{len(seen)} sources in this sample:")
    for src, c in seen.most_common():
        print(f"  {src:24s} {c:4d}   {examples[src]}")

    absent = [s for s in FULL_PILE if s not in seen]
    print(f"\npresent in the full Pile but not in this sample ({len(absent)}):")
    print("  " + ", ".join(absent))

    out = {"rows_inspected": n_rows, "records_without_label": missing_field,
           "sources_seen": dict(seen), "absent_from_sample": absent,
           "corpus": VAL_URL}
    json.dump(out, open(os.path.join(RES, "pile_verification.json"), "w"), indent=1)
    print("\nwrote results/pile_verification.json")
    print("NOTE: a 100-row sample cannot distinguish a source that the mirror "
          "removed from one that is merely rare. The full run records the "
          "sources it actually encounters, and that list is what the "
          "limitations cite.")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100))
