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

# Covalent P-O is 1.48-1.61 A. These bounds are deliberately generous so that
# only genuinely broken coordinates are flagged, not poorly refined ones.
P_O_MIN, P_O_MAX = 1.2, 2.0

# protein_to_pyg fills absent atoms with this value, and coord_mask only tests
# the x component of the comparison. Any component equal to it means the atom is
# a placeholder, not a measurement.
FILL_VALUE_COORDS = 1e-5

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


def _scan_one(path_str, eps, na_only=False):
    """Scan one .pt; return per-restype counts plus this entry's totals.

    Runs in a worker process. With na_only, entries containing any protein
    residue are skipped entirely (protein-nucleic acid complexes).
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

    if na_only:
        # A complex is excluded as a whole entry, not residue by residue.
        is_prot = (rtype >= rc.prot_from_idx) & (rtype <= rc.prot_to_idx)
        n_prot = int(is_prot.sum())
        if n_prot:
            return {"ok": True, "rows": {}, "sv": None, "entry": None,
                    "complex": True, "n_prot": n_prot,
                    "n_res": int(rtype.numel())}

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

    # Diagnostics only below this point. `present` mirrors what the production
    # rule sees (coord_mask); `valid` additionally rejects placeholder coords so
    # that a fill value cannot masquerade as an impossible bond length.
    present = m[..., I_OP1] & m[..., I_OP2] & m[..., I_P] & m[..., I_O5P]
    four = torch.stack([c[..., i, :] for i in (I_OP1, I_OP2, I_P, I_O5P)], dim=-2)
    is_fill = torch.any(four == FILL_VALUE_COORDS, dim=(-1, -2))
    valid = present & ~is_fill
    n_fill_leak = int((present & is_fill).sum())

    sv = torch.linalg.vecdot(
        torch.linalg.cross(c[..., I_OP1, :] - c[..., I_P, :],
                           c[..., I_O5P, :] - c[..., I_P, :], dim=-1),
        c[..., I_OP2, :] - c[..., I_P, :],
        dim=-1,
    )
    degenerate = valid & (sv.abs() <= eps)

    rows = {}
    n_rt = _bincount(rt, n_slots)
    flip_rt = _bincount(rt[flipped], n_slots)
    # "missing" is everything the diagnostics cannot use, placeholders included,
    # so that n_res - missing - degenerate equals the |sv| denominator exactly.
    miss_rt = _bincount(rt[~valid], n_slots)
    degen_rt = _bincount(rt[degenerate], n_slots)
    for i in torch.nonzero(n_rt, as_tuple=True)[0].tolist():
        rows[IDX_TO_NAME.get(i, f"idx{i}")] = (
            int(n_rt[i]), int(flip_rt[i]), int(miss_rt[i]), int(degen_rt[i])
        )

    decided = valid & ~degenerate

    # Data quality: a signed volume far outside the tetrahedral range implies
    # impossible P-O bond lengths. Check the distances directly -- they are the
    # interpretable quantity, and they also tell us whether the flips cluster
    # in broken geometry rather than in mislabelled but sound geometry.
    d_op1 = torch.linalg.norm(c[..., I_OP1, :] - c[..., I_P, :], dim=-1)
    d_op2 = torch.linalg.norm(c[..., I_OP2, :] - c[..., I_P, :], dim=-1)
    d_o5p = torch.linalg.norm(c[..., I_O5P, :] - c[..., I_P, :], dim=-1)
    bad_geom = valid & (
        (d_op1 < P_O_MIN) | (d_op1 > P_O_MAX)
        | (d_op2 < P_O_MIN) | (d_op2 > P_O_MAX)
        | (d_o5p < P_O_MIN) | (d_o5p > P_O_MAX)
    )
    geom = None
    if bool(valid.any()):
        dp = torch.stack([d_op1, d_op2, d_o5p], dim=-1)[valid]
        geom = {
            "n_valid": int(valid.sum()),
            "n_bad": int(bad_geom.sum()),
            "n_bad_decided": int((bad_geom & decided).sum()),
            "n_flip_bad": int((flipped & bad_geom).sum()),
            "n_flip_good": int((flipped & decided & ~bad_geom).sum()),
            "n_flip_fill": int((flipped & present & is_fill).sum()),
            "d_min": float(dp.min()),
            "d_max": float(dp.max()),
        }

    entry = {
        "name": path.stem,
        "n_na": int(n_rt.sum()),
        "n_flip": int(flip_rt.sum()),
        "n_skip": int(miss_rt.sum() + degen_rt.sum()),
        "n_bad": int(bad_geom.sum()),
        "n_fill_leak": n_fill_leak,
    }

    av = sv[decided].abs()
    sv_stat = None
    if av.numel():
        sv_stat = {
            "n": int(av.numel()),
            "min": float(av.min()),
            "max": float(av.max()),
            "buckets": [int((av < t).sum()) for t in SV_THRESHOLDS],
        }
    return {"ok": True, "rows": rows, "sv": sv_stat, "entry": entry,
            "geom": geom, "n_fill_leak": n_fill_leak}


def _collect(paths, workers, eps, quiet, na_only=False):
    total = len(paths)
    if total == 0:
        sys.exit("ERROR: no .pt files found")

    rows = defaultdict(lambda: [0, 0, 0, 0])
    sv = {"n": 0, "min": float("inf"), "max": 0.0, "buckets": [0] * N_SV_BUCKETS}
    entries = []
    errors, done, n_na_files = [], 0, 0
    cx = {"n": 0, "n_res": 0, "n_prot": 0}
    gm = {"n_valid": 0, "n_fill_leak": 0, "n_bad": 0, "n_bad_decided": 0,
          "n_flip_bad": 0, "n_flip_good": 0, "n_flip_fill": 0,
          "d_min": float("inf"), "d_max": 0.0}

    def _merge(res):
        nonlocal n_na_files
        if not res["ok"]:
            errors.append(res["err"])
            return
        if res.get("complex"):
            cx["n"] += 1
            cx["n_res"] += res["n_res"]
            cx["n_prot"] += res["n_prot"]
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
        gm["n_fill_leak"] += res.get("n_fill_leak", 0)
        g = res.get("geom")
        if g:
            for k in ("n_valid", "n_bad", "n_bad_decided",
                      "n_flip_bad", "n_flip_good", "n_flip_fill"):
                gm[k] += g[k]
            gm["d_min"] = min(gm["d_min"], g["d_min"])
            gm["d_max"] = max(gm["d_max"], g["d_max"])

    if workers <= 1:
        for p in paths:
            _merge(_scan_one(str(p), eps, na_only))
            done += 1
            if not quiet and done % 500 == 0:
                print(f"  ... {done}/{total}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_scan_one, str(p), eps, na_only) for p in paths]
            for fut in as_completed(futs):
                _merge(fut.result())
                done += 1
                if not quiet and done % 500 == 0:
                    print(f"  ... {done}/{total}", file=sys.stderr)

    return rows, sv, errors, total, n_na_files, entries, cx, gm


DNA_NAMES = ("DA", "DC", "DG", "DT", "DX")
RNA_NAMES = ("A", "C", "G", "U", "X")


def _group(rows, names):
    acc = [0, 0, 0, 0]
    for nm in names:
        for i, v in enumerate(rows.get(nm, (0, 0, 0, 0))):
            acc[i] += v
    return acc


def _fmt_row(label, r):
    # flip% uses the decidable count, the same denominator as the per-entry
    # distribution; otherwise missing/degenerate residues dilute the rate and
    # the two tables disagree (badly so for X, which is ~46% missing).
    n, f, mi, dg = r
    nd = n - mi - dg
    pct = f"{100.0 * f / nd:>7.2f}%" if nd else f"{'-':>8s}"
    return (f"{label:<10s} {n:>12,d} {nd:>12,d} {f:>12,d} {pct:>9s} "
            f"{mi:>10,d} {dg:>12,d}")


HEADER = (
    f"{'restype':<10s} {'n_res':>12s} {'n_decided':>12s} {'n_flipped':>12s} "
    f"{'flip%':>9s} {'missing':>10s} {'degenerate':>12s}"
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


def _report(rows, sv, errors, n_files, n_na_files, eps, cx=None, gm=None):
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
    nd = n - allr[2] - allr[3]
    print()
    print(f"Scanned {n_files:,d} files ({n_na_files:,d} containing nucleic acid), "
          f"{n:,d} nucleic-acid residues")
    if cx and cx["n"]:
        print(f"--na-only: excluded {cx['n']:,d} protein-nucleic acid complexes "
              f"({cx['n_res']:,d} residues total, {cx['n_prot']:,d} of them protein)")
    if nd:
        pct = 100.0 * f / nd
        print(f"OP1/OP2 swapped: {f:,d}/{nd:,d} decidable = {pct:.2f}%"
              f"   ({100.0 * f / n:.2f}% of all {n:,d} NA residues)")
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

    if gm and gm["n_valid"]:
        nv = gm["n_valid"]
        print()
        print("Data quality (P-O bond lengths, A):")
        print(f"  observed {gm['d_min']:.4g} - {gm['d_max']:.4g}"
              f"    covalent P-O is 1.48-1.61, flag window [{P_O_MIN}, {P_O_MAX}]")
        if gm["n_fill_leak"]:
            print(f"  WARNING: {gm['n_fill_leak']:,d} residues pass coord_mask yet "
                  f"hold a coordinate == {FILL_VALUE_COORDS:g}.")
            print("           coord_mask tests only the x component, so placeholder")
            print("           atoms leak through and the production rule decides")
            print("           their handedness.")
            print(f"           {gm['n_flip_fill']:,d} of them were actually swapped. "
                  f"They count as 'missing'")
            print("           in the table above, so those swaps appear in n_flipped")
            print("           but fall outside the flip% denominator.")
        nb = gm["n_bad"]
        if nb:
            print(f"  {nb:,d} residues ({100.0 * nb / nv:.3f}% of {nv:,d} usable) "
                  f"have a P-O distance outside [{P_O_MIN}, {P_O_MAX}]")
            print("  -> broken coordinates, not refinement noise.")
            nbd = gm["n_bad_decided"]
            ngd = max(0, sv["n"] - nbd)
            fb, fg = gm["n_flip_bad"], gm["n_flip_good"]
            rb = f"{100.0 * fb / nbd:.2f}%" if nbd else "n/a"
            rg = f"{100.0 * fg / ngd:.2f}%" if ngd else "n/a"
            print(f"  flip rate, broken geometry: {fb:,d}/{nbd:,d} = {rb}")
            print(f"  flip rate, sound geometry : {fg:,d}/{ngd:,d} = {rg}")
            print("  A much higher rate in broken geometry means the flips are a")
            print("  data-quality artifact, not a competing labelling convention.")
        else:
            print(f"  No P-O distance outside [{P_O_MIN}, {P_O_MAX}] among "
                  f"{nv:,d} usable residues.")

    if sv["n"]:
        print()
        print(f"|signed volume| over the {sv['n']:,d} decided residues (A^3):")
        print(f"  min = {sv['min']:.4g}    max = {sv['max']:.4g}")
        for t, c in zip(SV_THRESHOLDS, sv["buckets"]):
            print(f"  |sv| < {t:<6g} : {c:,d}")
        if eps:
            print(f"  min / eps = {sv['min'] / eps:.3g}")
        # A small margin does not mean eps is too large -- the degenerate count
        # above shows whether anything was actually skipped. It means some
        # residues are decided on knife-edge geometry, so the remedy is to RAISE
        # eps; lowering it would make those decisions more confident, not safer.
        fragile = [(t, c) for t, c in zip(SV_THRESHOLDS, sv["buckets"]) if c]
        if fragile:
            t, c = fragile[0]
            print(f"  WARNING: {c:,d} decided residues have |sv| < {t:g}. Their")
            print("           handedness is numerically fragile -- float32 rounding")
            print(f"           can flip the sign. Consider --eps {t:g} to skip them.")
            print("           Lowering eps would make these decisions *more*")
            print("           confident, which is the opposite of what is wanted.")
        else:
            print(f"  All decided residues have |sv| >= {SV_THRESHOLDS[-1]:g}; "
                  f"eps={eps:g} sits well below the data.")

    if errors:
        print()
        print(f"{len(errors)} files failed to load; first 10:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restype", "n_res", "n_decided", "n_flipped", "flip_pct",
                    "n_missing_atom", "n_degenerate"])
        for nm in sorted(rows, key=lambda k: (k not in DNA_NAMES + RNA_NAMES, k)):
            n, f, mi, dg = rows[nm]
            nd = n - mi - dg
            w.writerow([nm, n, nd, f, f"{100.0 * f / nd:.4f}" if nd else "", mi, dg])
    print(f"\nCSV written to {path}")


def _write_entry_csv(path, entries, min_na_res):
    """Export every entry so rates can be cross-tabulated against PDB metadata."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "n_na", "n_decided", "n_flipped", "flip_pct",
                    "n_skipped", "n_bad_geom", "n_fill_leak", "included", "bucket"])
        for e in sorted(entries, key=lambda x: x["name"]):
            n_dec = e["n_na"] - e["n_skip"]
            included = n_dec >= min_na_res and n_dec > 0
            pct = (100.0 * e["n_flip"] / n_dec) if n_dec > 0 else None
            w.writerow([
                e["name"], e["n_na"], n_dec, e["n_flip"],
                f"{pct:.4f}" if pct is not None else "",
                e["n_skip"], e.get("n_bad", 0), e.get("n_fill_leak", 0),
                int(included),
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
    ap.add_argument("--na-only", action="store_true",
                    help="keep only pure nucleic-acid entries: skip any entry "
                         "containing a protein residue (protein-NA complexes). "
                         "Note this cannot distinguish free NA from a NA chain "
                         "that was extracted per-chain from a complex")
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

    rows, sv, errors, n_files, n_na_files, entries, cx, gm = _collect(
        paths, args.workers, args.eps, args.quiet, args.na_only
    )
    if not rows:
        hint = (" With --na-only every entry may have been a protein-NA "
                "complex; rerun without the flag to check." if args.na_only else "")
        sys.exit("ERROR: no nucleic-acid residues found -- check that the path "
                 "points at the preprocessed cache and that restype_list "
                 "enables DNA/RNA." + hint)

    _report(rows, sv, errors, n_files, n_na_files, args.eps, cx, gm)
    _report_entries(entries, args.min_na_res)
    if args.csv:
        _write_csv(args.csv, rows)
    if args.by_entry_csv:
        _write_entry_csv(args.by_entry_csv, entries, args.min_na_res)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
