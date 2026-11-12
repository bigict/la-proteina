#!/usr/bin/env python3
"""Measure how often the OP1/OP2 canonicalization rule relabels a dataset.

Background: OP1 and OP2 are chemically equivalent phosphate oxygens and their
naming in PDB entries is arbitrary. Before the rule was wired in, every feature
that distinguished them carried dataset-level label noise: the alpha surrogate
in chi_angles_atoms[NA][0], all five channels of
RelativeResidueOrientationPairFeat, two of four channels of
BackbonePairDistancesNanometerPairFeat, and theta_x/theta_y of
BackboneBondAnglesSeqFeat. This script quantifies that noise.

Flip detection calls the production `canonicalize_phosphate_oxygens` and diffs
the OP1 slot before/after, rather than reimplementing the sign rule, so the
statistic stays coupled to whatever the pipeline actually does. The signed
volume is recomputed only for diagnostics (separating "already canonical" from
"rule declined to decide"); it never drives the decision.

Two levels of reporting:

1) Global and per-restype flip rate. Near 50% means the original labelling was
   uncorrelated with geometry.

2) Per-entry flip-rate distribution. A global 50% has two very different causes
   that must be told apart:
   (a) Consistent within an entry, different between entries -- each entry's
       rate sits near 0% or 100% and the distribution is bimodal. OP1/OP2 is
       then a systematic per-entry property, plausibly correlated with
       deposition software or date, i.e. a *learnable confound*, which is worse
       than random noise because the model can fit it.
   (b) Random within an entry -- rates cluster around 50%, i.e. plain variance.
   The strength of the claim, and the value of the fix, differ between the two.

The per-entry denominator is the number of *decidable* residues (missing-atom
and near-planar residues are excluded); counting them would dilute every rate
toward 0% and fabricate within-entry consistency. --min-na-res additionally
drops entries with too few nucleic-acid residues: an entry with a single
nucleotide necessarily has a rate of 0% or 100%, inflating the consistent
fraction.

Usage:
  python script_utils/stat_op1_op2_flip.py $DATA_PATH/pdb_train/processed
  python script_utils/stat_op1_op2_flip.py <dir_or_pt> [<dir_or_pt> ...] \
      --limit 2000 --workers 8 --csv op1_op2_flip.csv \
      --by-entry-csv op1_op2_by_entry.csv --min-na-res 4

Arguments may be .pt files or directories (searched recursively for *.pt). The
script reads graph.coords / graph.coord_mask / graph.residue_type from the
preprocessed cache, i.e. the labelling *before* canonicalization. Do not use
pdb_utils.from_pdb_string to measure the original confusion -- it canonicalizes
on load.
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from openfold.np import residue_constants as rc
from proteinfoundation.utils.coors_utils import canonicalize_phosphate_oxygens

# NB: do not invert rc.resname_to_idx. It is built from rc.resnames, whose last
# element is mistakenly the tuple ("UNK", PROT) rather than the string "UNK",
# so build the index -> name map forward from restypes instead.
IDX_TO_NAME = {
    i: rc.restype_1to3[(one, rc.moltype(i))] for i, one in enumerate(rc.restypes)
}

I_OP1 = rc.atom_order["OP1"]
I_OP2 = rc.atom_order["OP2"]
I_P = rc.atom_order["P"]
I_O5P = rc.atom_order["O5'"]

N_SV_BUCKETS = 3  # |sv| < 1e-3, < 1e-2, < 1e-1
SV_THRESHOLDS = (1e-3, 1e-2, 1e-1)

ENTRY_BUCKET_LABELS = (
    "= 0%", "(0%,5%)", "[5%,45%)", "[45%,55%]", "(55%,95%)", "[95%,100%)", "= 100%",
)


def _entry_bucket(pct):
    """Map a per-entry flip rate to one of 7 disjoint, exhaustive buckets."""
    if pct == 0.0:
        return 0
    if pct < 5.0:
        return 1
    if pct < 45.0:
        return 2
    if pct <= 55.0:
        return 3
    if pct < 95.0:
        return 4
    if pct < 100.0:
        return 5
    return 6


def _bincount(x, minlength):
    """torch.bincount on a possibly-empty selection."""
    if x.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long)
    return torch.bincount(x, minlength=minlength)


def _scan_one(path_str, eps):
    """Scan one .pt; return per-restype counts plus this entry's totals.

    Runs in a worker process.
    """
    path = Path(path_str)
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # one bad file must not abort the whole sweep
        return {"ok": False, "err": f"{path.name}: {type(exc).__name__}: {exc}"}

    missing = [k for k in ("coords", "coord_mask", "residue_type") if k not in graph]
    if missing:
        return {"ok": False, "err": f"{path.name}: missing {'/'.join(missing)}"}

    n_slots = rc.restype_num + 1
    coords = graph.coords.float()
    mask = graph.coord_mask.bool()
    rtype = graph.residue_type.long()

    is_na = (
        ((rtype >= rc.dna_from_idx) & (rtype <= rc.dna_to_idx))
        | ((rtype >= rc.rna_from_idx) & (rtype <= rc.rna_to_idx))
    )
    if not bool(is_na.any()):
        return {"ok": True, "rows": {}, "sv": None, "entry": None}

    sel = torch.nonzero(is_na, as_tuple=True)[0]
    c, m, rt = coords[sel], mask[sel], rtype[sel]

    # Which residues the production rule actually swaps.
    canon = canonicalize_phosphate_oxygens(c, m, eps=eps)
    flipped = torch.any(canon[..., I_OP1, :] != c[..., I_OP1, :], dim=-1)

    # Diagnostics only below this point.
    present = m[..., I_OP1] & m[..., I_OP2] & m[..., I_P] & m[..., I_O5P]
    sv = torch.linalg.vecdot(
        torch.linalg.cross(c[..., I_OP1, :] - c[..., I_P, :],
                           c[..., I_O5P, :] - c[..., I_P, :], dim=-1),
        c[..., I_OP2, :] - c[..., I_P, :],
        dim=-1,
    )
    degenerate = present & (sv.abs() <= eps)

    rows = {}
    n_rt = _bincount(rt, n_slots)
    flip_rt = _bincount(rt[flipped], n_slots)
    miss_rt = _bincount(rt[~present], n_slots)
    degen_rt = _bincount(rt[degenerate], n_slots)
    for i in torch.nonzero(n_rt, as_tuple=True)[0].tolist():
        rows[IDX_TO_NAME.get(i, f"idx{i}")] = (
            int(n_rt[i]), int(flip_rt[i]), int(miss_rt[i]), int(degen_rt[i])
        )

    entry = {
        "name": path.stem,
        "n_na": int(n_rt.sum()),
        "n_flip": int(flip_rt.sum()),
        "n_skip": int(miss_rt.sum() + degen_rt.sum()),
    }

    av = sv[present & ~degenerate].abs()
    sv_stat = None
    if av.numel():
        sv_stat = {
            "n": int(av.numel()),
            "min": float(av.min()),
            "max": float(av.max()),
            "buckets": [int((av < t).sum()) for t in SV_THRESHOLDS],
        }
    return {"ok": True, "rows": rows, "sv": sv_stat, "entry": entry}


def _collect(paths, workers, eps, quiet):
    total = len(paths)
    if total == 0:
        sys.exit("ERROR: no .pt files found")

    rows = defaultdict(lambda: [0, 0, 0, 0])
    sv = {"n": 0, "min": float("inf"), "max": 0.0, "buckets": [0] * N_SV_BUCKETS}
    entries = []
    errors, done, n_na_files = [], 0, 0

    def _merge(res):
        nonlocal n_na_files
        if not res["ok"]:
            errors.append(res["err"])
            return
        if res["rows"]:
            n_na_files += 1
        for name, (n, f, mi, dg) in res["rows"].items():
            acc = rows[name]
            acc[0] += n
            acc[1] += f
            acc[2] += mi
            acc[3] += dg
        if res.get("entry"):
            entries.append(res["entry"])
        s = res["sv"]
        if s:
            sv["n"] += s["n"]
            sv["min"] = min(sv["min"], s["min"])
            sv["max"] = max(sv["max"], s["max"])
            for i in range(N_SV_BUCKETS):
                sv["buckets"][i] += s["buckets"][i]

    if workers <= 1:
        for p in paths:
            _merge(_scan_one(str(p), eps))
            done += 1
            if not quiet and done % 500 == 0:
                print(f"  ... {done}/{total}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_scan_one, str(p), eps) for p in paths]
            for fut in as_completed(futs):
                _merge(fut.result())
                done += 1
                if not quiet and done % 500 == 0:
                    print(f"  ... {done}/{total}", file=sys.stderr)

    return rows, sv, errors, total, n_na_files, entries


DNA_NAMES = ("DA", "DC", "DG", "DT", "DX")
RNA_NAMES = ("A", "C", "G", "U", "X")


def _group(rows, names):
    acc = [0, 0, 0, 0]
    for nm in names:
        for i, v in enumerate(rows.get(nm, (0, 0, 0, 0))):
            acc[i] += v
    return acc


def _fmt_row(label, r):
    n, f, mi, dg = r
    pct = f"{100.0 * f / n:>7.2f}%" if n else f"{'-':>8s}"
    return f"{label:<10s} {n:>12,d} {f:>12,d} {pct:>9s} {mi:>10,d} {dg:>12,d}"


HEADER = (
    f"{'restype':<10s} {'n_res':>12s} {'n_flipped':>12s} {'flip%':>9s} "
    f"{'missing':>10s} {'degenerate':>12s}"
)


def _select_entries(entries, min_na_res):
    """Filter entries by decidable residue count; return (kept, n_small, n_undecided).

    kept elements are (name, n_decided, n_flip, pct). The denominator is the
    decidable count, not all nucleic-acid residues, so that missing-atom and
    near-planar residues cannot dilute the rate toward 0%.
    """
    kept, n_small, n_undecided = [], 0, 0
    for e in entries:
        n_dec = e["n_na"] - e["n_skip"]
        if n_dec <= 0:
            n_undecided += 1
            continue
        if n_dec < min_na_res:
            n_small += 1
            continue
        kept.append((e["name"], n_dec, e["n_flip"], 100.0 * e["n_flip"] / n_dec))
    return kept, n_small, n_undecided


def _report_entries(entries, min_na_res):
    """Print the per-entry distribution and classify within-entry vs between-entry."""
    if not entries:
        return
    kept, n_small, n_undecided = _select_entries(entries, min_na_res)

    print()
    print("=" * 72)
    print("Per-entry flip-rate distribution")
    print("=" * 72)
    print(f"Entries containing nucleic acid: {len(entries):,d}")
    if n_undecided:
        print(f"  excluded {n_undecided:,d}: no decidable residue at all "
              f"(missing atoms / near-planar)")
    if n_small:
        print(f"  excluded {n_small:,d}: fewer than {min_na_res} decidable residues "
              f"(too few residues force the rate to 0%/100%, inflating consistency)")
    if not kept:
        print(f"  No entry meets --min-na-res={min_na_res}; no distribution to report.")
        return
    print(f"  included: {len(kept):,d}")

    counts = [0] * len(ENTRY_BUCKET_LABELS)
    for _, _, _, pct in kept:
        counts[_entry_bucket(pct)] += 1

    n_kept = len(kept)
    print()
    print(f"{'flip-rate bin':<14s} {'entries':>10s} {'share':>9s}  histogram")
    print("-" * 72)
    for label, c in zip(ENTRY_BUCKET_LABELS, counts):
        frac = c / n_kept
        bar = "#" * int(round(frac * 40))
        print(f"{label:<14s} {c:>10,d} {frac * 100:>8.2f}%  {bar}")

    n_res = sum(k[1] for k in kept)
    n_flip = sum(k[2] for k in kept)
    agg = 100.0 * n_flip / n_res
    cons = sum(1 for k in kept if k[3] < 5.0 or k[3] >= 95.0)
    mixed = sum(1 for k in kept if 45.0 <= k[3] <= 55.0)
    cons_f, mixed_f = cons / n_kept, mixed / n_kept

    print()
    print(f"Aggregate flip rate over included entries (residue-weighted): "
          f"{n_flip:,d}/{n_res:,d} = {agg:.2f}%")
    print(f"Within-entry consistent (<5% or >=95%): "
          f"{cons:,d}/{n_kept:,d} = {cons_f * 100:.2f}%")
    print(f"Within-entry mixed (45%-55%)         : "
          f"{mixed:,d}/{n_kept:,d} = {mixed_f * 100:.2f}%")

    print()
    if cons_f >= 0.8 and 40.0 <= agg <= 60.0:
        print("=> Case (a): consistent within entries, different conventions between")
        print("   them (bimodal). OP1/OP2 is a systematic per-entry property, likely")
        print("   correlated with deposition software or date, i.e. a LEARNABLE")
        print("   CONFOUND -- worse than random noise, because the model can fit it.")
        print("   Export --by-entry-csv and cross-tabulate against deposition date,")
        print("   resolution and experimental method to confirm the correlation.")
    elif mixed_f >= 0.5:
        print("=> Case (b): random within entries, i.e. plain variance, not a")
        print("   learnable confound.")
    elif cons_f >= 0.8:
        print("=> Consistent within entries and the aggregate rate is far from 50%:")
        print("   the original labelling is already largely consistent, so")
        print("   canonicalization is mainly defensive (cross-source agreement)")
        print("   and the noise concern was overstated.")
    else:
        print("=> Mixed: some entries consistent, some random within. Export the")
        print("   per-entry CSV and inspect by group; one global number is not")
        print("   enough to describe this.")


def _report(rows, sv, errors, n_files, n_na_files, eps):
    print()
    print(HEADER)
    print("-" * len(HEADER))

    present_names = [n for n in DNA_NAMES + RNA_NAMES if n in rows]
    other = sorted(set(rows) - set(present_names))
    for nm in present_names + other:
        print(_fmt_row(nm, rows[nm]))

    print("-" * len(HEADER))
    dna, rna = _group(rows, DNA_NAMES), _group(rows, RNA_NAMES)
    allr = [dna[i] + rna[i] for i in range(4)]
    for name in other:
        for i, v in enumerate(rows[name]):
            allr[i] += v
    print(_fmt_row("DNA", dna))
    print(_fmt_row("RNA", rna))
    print(_fmt_row("ALL", allr))

    n, f = allr[0], allr[1]
    print()
    print(f"Scanned {n_files:,d} files ({n_na_files:,d} containing nucleic acid), "
          f"{n:,d} nucleic-acid residues")
    if n:
        pct = 100.0 * f / n
        print(f"OP1/OP2 swapped: {f:,d} = {pct:.2f}%")
        if 40.0 <= pct <= 60.0:
            print("  => Near 50%: the original labelling is essentially uncorrelated")
            print("     with geometry. See the per-entry distribution below to tell")
            print("     'different between entries' from 'random within an entry'.")
        elif pct < 5.0 or pct > 95.0:
            print("  => Far from 50%: the original labelling is already highly")
            print("     consistent, so the label noise is much smaller than feared")
            print("     and canonicalization is mainly defensive.")
        else:
            print("  => In between: some sources consistent, some not; mixed noise.")

    if allr[2]:
        print(f"Note: {allr[2]:,d} residues lack OP1/OP2/P/O5' and cannot be "
              f"decided; the rule leaves them untouched by design.")
    if allr[3]:
        print(f"Note: {allr[3]:,d} residues have a near-planar phosphate "
              f"(|sv| <= {eps:g}); the rule declines to decide.")

    if sv["n"]:
        print()
        print(f"|signed volume| over the {sv['n']:,d} decided residues (A^3):")
        print(f"  min = {sv['min']:.4g}    max = {sv['max']:.4g}")
        for t, c in zip(SV_THRESHOLDS, sv["buckets"]):
            print(f"  |sv| < {t:<6g} : {c:,d}")
        margin = sv["min"] / eps if eps else float("inf")
        verdict = "eps is safe" if margin > 100 else "eps too large, lower it"
        print(f"  min / eps = {margin:.3g}  ({verdict})")

    if errors:
        print()
        print(f"{len(errors)} files failed to load; first 10:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restype", "n_res", "n_flipped", "flip_pct",
                    "n_missing_atom", "n_degenerate"])
        for nm in sorted(rows, key=lambda k: (k not in DNA_NAMES + RNA_NAMES, k)):
            n, f, mi, dg = rows[nm]
            w.writerow([nm, n, f, f"{100.0 * f / n:.4f}" if n else "", mi, dg])
    print(f"\nCSV written to {path}")


def _write_entry_csv(path, entries, min_na_res):
    """Export every entry so rates can be cross-tabulated against PDB metadata."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "n_na", "n_decided", "n_flipped", "flip_pct",
                    "n_skipped", "included", "bucket"])
        for e in sorted(entries, key=lambda x: x["name"]):
            n_dec = e["n_na"] - e["n_skip"]
            included = n_dec >= min_na_res and n_dec > 0
            pct = (100.0 * e["n_flip"] / n_dec) if n_dec > 0 else None
            w.writerow([
                e["name"], e["n_na"], n_dec, e["n_flip"],
                f"{pct:.4f}" if pct is not None else "",
                e["n_skip"], int(included),
                ENTRY_BUCKET_LABELS[_entry_bucket(pct)] if included else "",
            ])
    print(f"Per-entry CSV written to {path}")


def _resolve(inputs):
    out = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pt")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"WARN: path does not exist, skipping: {p}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Measure the OP1/OP2 canonicalization flip rate "
                    "(global and per-entry distribution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+",
                    help=".pt files or directories (searched recursively for *.pt)")
    ap.add_argument("--limit", type=int, default=None,
                    help="scan only this many files, drawn as a seeded random "
                         "sample rather than the first N sorted paths (the latter "
                         "is biased if the cache is laid out by molecule type)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --limit")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="number of worker processes (1 = serial, easier to debug)")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="near-planar threshold; must match "
                         "canonicalize_phosphate_oxygens")
    ap.add_argument("--min-na-res", type=int, default=4,
                    help="per-entry distribution only includes entries with at "
                         "least this many decidable residues (default 4; too few "
                         "forces the rate to 0%%/100%%, inflating consistency)")
    ap.add_argument("--csv", default=None, help="also write a per-restype CSV")
    ap.add_argument("--by-entry-csv", default=None,
                    help="also write a per-entry CSV (includes excluded entries, "
                         "for cross-tabulation)")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = ap.parse_args()

    paths = _resolve(args.paths)
    if args.limit is not None:
        k = min(args.limit, len(paths))
        paths = random.Random(args.seed).sample(paths, k)
        if not args.quiet:
            print(f"--limit: random sample of {k} files (seed={args.seed})")
    if not args.quiet:
        print(f"Scanning {len(paths)} .pt files, workers={args.workers}, "
              f"eps={args.eps:g}")

    rows, sv, errors, n_files, n_na_files, entries = _collect(
        paths, args.workers, args.eps, args.quiet
    )
    if not rows:
        sys.exit("ERROR: no nucleic-acid residues found -- check that the path "
                 "points at the preprocessed cache and that restype_list "
                 "enables DNA/RNA")

    _report(rows, sv, errors, n_files, n_na_files, args.eps)
    _report_entries(entries, args.min_na_res)
    if args.csv:
        _write_csv(args.csv, rows)
    if args.by_entry_csv:
        _write_entry_csv(args.by_entry_csv, entries, args.min_na_res)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
