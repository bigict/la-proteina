"""Smoke test for reusing an offline train/val/test assignment.

Exercises the real ``PDBDataSplitter`` and ``split_dataframe`` with real pandas.
Needs the training environment (torch and the repo imports), but **no DATA_PATH
and no PDB files**: the fixture frame is built in memory and also written to a
CSV so the on-disk round trip is covered.

Run:
    PYTHONPATH=. python script_utils/smoke_offline_split.py

Exits non-zero on any failing expectation, so it can gate a merge.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from proteinfoundation.datasets.pdb_data import PDBDataSplitter
from proteinfoundation.utils.cluster_utils import split_dataframe

SPLITS = ["train", "val", "test"]
RATIOS = [0.8, 0.15, 0.05]
# Deliberately not proportional to RATIOS, so a re-split cannot coincide with it.
ASSIGN = ["train"] * 5 + ["val"] * 4 + ["test"] * 3
WANT = {s: ASSIGN.count(s) for s in SPLITS}

_failures = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        _failures.append(label)


def fixture(assign=ASSIGN):
    return pd.DataFrame(
        {
            "pdb": [f"p{i:03d}" for i in range(len(assign))],
            "chain": ["A"] * len(assign),
            "split": assign,
        }
    )


def run_split(df, overwrite):
    """split_data returns (dfs_splits, clusterid_to_seqid_mappings), not a dict."""
    sp = PDBDataSplitter(
        df_data=df,
        data_dir=".",
        train_val_test=RATIOS,
        split_type="random",
        overwrite_split_dataframe=overwrite,
    )
    dfs, mappings = sp.split_data(df, "smoke")
    return sp, dfs, mappings


def main():
    df = fixture()
    print(f"fixture: {len(df)} rows, offline assignment {WANT}")

    print("\n[1] CSV round trip keeps the split column")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "offline.csv"
        df.to_csv(path, index=False)
        back = pd.read_csv(path)
        check("split" in back.columns, "read_csv preserves the split column")
        check(back["split"].tolist() == ASSIGN, "labels survive the round trip")
        df_rt = back

    print("\n[2] overwrite=False reuses the assignment via the splitter")
    sp, out, mappings = run_split(df_rt, overwrite=False)
    check(isinstance(out, dict), f"dfs_splits is a dict, got {type(out).__name__}")
    check(set(out) == set(SPLITS), f"keys are the split names: {sorted(out)}")
    sizes = {s: len(out[s]) for s in SPLITS}
    check(sizes == WANT, f"sizes match the offline file: {sizes} vs {WANT}")
    check(out["train"]["pdb"].tolist() == df_rt["pdb"].tolist()[:5],
          "row order preserved, not reshuffled")
    check(mappings is None, "random route leaves cluster mappings unset")
    check(set(sp.dfs_splits) == set(SPLITS), "splitter.dfs_splits is keyed too")

    print("\n[3] consumer access patterns work on the result")
    check(len(out["train"]) == 5, "d['train'] indexes by split name")
    items = list(out.items())
    check(all(isinstance(k, str) and isinstance(v, pd.DataFrame) for k, v in items),
          ".items() yields (split name, DataFrame)")
    check(out["train"]["pdb"].tolist() == [f"p{i:03d}" for i in range(5)],
          "the pdb column reads back as expected downstream")

    print("\n[4] overwrite=True still re-splits (backward compatible)")
    _, out2, _ = run_split(df_rt, overwrite=True)
    check(isinstance(out2, dict), "still a dict")
    check(sum(len(out2[s]) for s in SPLITS) == len(df_rt), "no rows lost")
    check({s: len(out2[s]) for s in SPLITS} != WANT,
          "sizes follow RATIOS, not the offline labels")

    print("\n[5] no split column falls back to a normal split")
    plain = df_rt.drop(columns=["split"])
    _, out3, _ = run_split(plain, overwrite=False)
    check(isinstance(out3, dict) and sum(len(out3[s]) for s in SPLITS) == len(plain),
          "overwrite=False without the column still splits")

    print("\n[6] validation rejects an unrecognized label")
    bad = fixture(["train"] * 5 + ["val"] * 4 + ["oops"] * 3)
    try:
        run_split(bad, overwrite=False)
        check(False, "expected ValueError for an unknown label")
    except ValueError as exc:
        check("oops" in str(exc), f"error names the offending value: {exc}")

    print("\n[7] validation rejects a missing split")
    missing = fixture(["train"] * 8 + ["val"] * 4)
    try:
        run_split(missing, overwrite=False)
        check(False, "expected ValueError for an empty test split")
    except ValueError as exc:
        check("test" in str(exc), f"error names the missing split: {exc}")

    print("\n[8] split_dataframe called directly agrees with the splitter")
    direct = split_dataframe(df_rt, SPLITS, RATIOS, overwrite=False)
    check({s: len(direct[s]) for s in SPLITS} == WANT, "direct call matches")

    print("\n" + ("ALL PASSED" if not _failures else f"{len(_failures)} FAILED"))
    for f in _failures:
        print("  - " + f)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
