#!/usr/bin/env python3
"""Derive torsion-independent distance restraints from the preprocessed cache.

Why not just use the ideal-geometry table: OpenFold's NA rigid groups are mostly
single atoms (DA has 8 groups for 21 atoms, only 4 of them multi-atom). Intra-group
distances therefore cover the phosphate and the base but miss the sugar ring and
the glycosidic bond, which are exactly the cross-group linkages. Measured on DA:
44 intra-group pairs, only 12 of them actual bonds, out of >=20 needed.

So the table is derived from the data instead. Real PDB structures are refined
against restraints, so the per-pair mean distance is the ideal value.

Which pairs may be constrained: only those at bond-graph distance 1 (bond lengths)
or 2 (1-3 distances, which fix bond angles via the law of cosines). Both are
independent of every torsion. Graph distance >= 3 depends on conformation and must
stay free, or the restraint would fight the model's legitimate degrees of freedom.

Note that distance restraints do not fix chirality: a mirror image has identical
pairwise distances. A separate signed-volume term is needed for that; this script
only reports the stereocentres it finds.

Output:
  - an .npz with target / mask / tol / graph_dist tables in atom65 layout
  - a text report: per-restype bond counts, spread, and a cross-check of the
    derived bond lengths against rigid_group_atom_positions
  - a CSV of every constrained pair, for inspection

Usage:
  python script_utils/build_geometry_restraints.py $DATA_PATH/pdb_train/processed \
      --out geometry_restraints.npz --csv geometry_restraints.csv
  python script_utils/build_geometry_restraints.py <dir> --limit 500 --workers 1
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from openfold.np import residue_constants as rc

FILL_VALUE_COORDS = 1e-5  # protein_to_pyg placeholder for absent atoms
BOND_CUTOFF = 1.8  # A; covalent bonds involving C/N/O/P are 1.2-1.7
PAIR_CUTOFF = 4.0  # A; generous enough to contain every 1-3 distance
MIN_COUNT = 50  # absolute floor; meaningless alone at 270k residues per type
# A real covalent bond occurs in essentially every residue of its type, so the
# binding constraint is coverage relative to that type's residue count. At full
# scale real bonds sit at 96-98% coverage while spurious pairs sit at 0.04-0.5%,
# so any threshold in between separates them cleanly.
MIN_FRAC = 0.2
# Unknown bases have inconsistent atom sets (residue_atoms["DX"] lists N1/C2 but
# rigid_group_atom_positions["DX"] does not), so their graphs fracture.
UNKNOWN_NAMES = ("DX", "X")
# Independent cycles: sugar ring 1 + purine 2 / pyrimidine 1. Acceptance gate is
# bonds == atoms - 1 + cycles for a connected graph.
EXPECTED_CYCLES = {"DA": 3, "DC": 2, "DG": 3, "DT": 2,
                   "A": 3, "C": 2, "G": 3, "U": 2}

NATOM = rc.atom_type_num
PAIR_I, PAIR_J = torch.triu_indices(NATOM, NATOM, offset=1)
NPAIR = PAIR_I.numel()


def _restype_range(which):
    if which == "na":
        return rc.dna_from_idx, rc.rna_to_idx
    if which == "dna":
        return rc.dna_from_idx, rc.dna_to_idx
    if which == "rna":
        return rc.rna_from_idx, rc.rna_to_idx
    return 0, rc.restype_num - 1


def _idx_to_name():
    return {i: rc.restype_1to3[(one, rc.moltype(i))]
            for i, one in enumerate(rc.restypes)}


def _scan_one(path_str, rlo, rhi):
    """Accumulate count/sum/sumsq/min/max per (restype, atom pair). Worker side."""
    path = Path(path_str)
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"ok": False, "err": f"{path.name}: {type(exc).__name__}: {exc}"}
    missing = [k for k in ("coords", "coord_mask", "residue_type") if k not in graph]
    if missing:
        return {"ok": False, "err": f"{path.name}: missing {'/'.join(missing)}"}

    R = rhi - rlo + 1
    shape = (R, NPAIR)
    cnt = torch.zeros(shape, dtype=torch.float64)
    s1 = torch.zeros(shape, dtype=torch.float64)
    s2 = torch.zeros(shape, dtype=torch.float64)
    mn = torch.full(shape, float("inf"), dtype=torch.float64)
    mx = torch.full(shape, float("-inf"), dtype=torch.float64)

    rt = graph.residue_type.long()
    sel = (rt >= rlo) & (rt <= rhi)
    if not bool(sel.any()):
        return {"ok": True, "stats": (cnt, s1, s2, mn, mx), "n_res": 0,
                "n_by_type": torch.zeros(R, dtype=torch.float64)}

    c = graph.coords[sel].double()
    m = graph.coord_mask[sel].bool()
    r = (rt[sel] - rlo).clamp(0, R - 1)
    # Per-type residue counts: a real covalent bond is present in essentially
    # every residue of its type, so coverage must be judged relative to this,
    # not against an absolute count that is meaningless at 270k residues.
    n_by_type = torch.bincount(r, minlength=R).to(torch.float64)

    # A placeholder coordinate is not a measurement; coord_mask only tests x.
    ok = m & ~(c == FILL_VALUE_COORDS).any(-1)

    d = (c[:, PAIR_I, :] - c[:, PAIR_J, :]).norm(dim=-1)  # [k, NPAIR]
    valid = ok[:, PAIR_I] & ok[:, PAIR_J] & (d < PAIR_CUTOFF) & torch.isfinite(d)
    if not bool(valid.any()):
        return {"ok": True, "stats": (cnt, s1, s2, mn, mx),
                "n_res": int(sel.sum()), "n_by_type": n_by_type}

    pidx = torch.arange(NPAIR, device=d.device).expand_as(d)
    key = (r[:, None] * NPAIR + pidx)[valid]
    dv = d[valid].to(torch.float64)

    cnt.view(-1).index_add_(0, key, torch.ones_like(dv))
    s1.view(-1).index_add_(0, key, dv)
    s2.view(-1).index_add_(0, key, dv * dv)
    mn.view(-1).scatter_reduce_(0, key, dv, reduce="amin", include_self=False)
    mx.view(-1).scatter_reduce_(0, key, dv, reduce="amax", include_self=False)
    return {"ok": True, "stats": (cnt, s1, s2, mn, mx),
            "n_res": int(sel.sum()), "n_by_type": n_by_type}


def _collect(paths, workers, rlo, rhi, quiet):
    R = rhi - rlo + 1
    shape = (R, NPAIR)
    cnt = torch.zeros(shape, dtype=torch.float64)
    s1 = torch.zeros(shape, dtype=torch.float64)
    s2 = torch.zeros(shape, dtype=torch.float64)
    mn = torch.full(shape, float("inf"), dtype=torch.float64)
    mx = torch.full(shape, float("-inf"), dtype=torch.float64)
    nbt = torch.zeros(R, dtype=torch.float64)
    errors, done, nres = [], 0, 0

    def merge(res):
        nonlocal nres
        if not res["ok"]:
            errors.append(res["err"])
            return
        c_, a_, b_, lo_, hi_ = res["stats"]
        cnt.add_(c_); s1.add_(a_); s2.add_(b_)
        torch.minimum(mn, lo_, out=mn)
        torch.maximum(mx, hi_, out=mx)
        nbt.add_(res["n_by_type"])
        nres += res["n_res"]

    if workers <= 1:
        for p in paths:
            merge(_scan_one(str(p), rlo, rhi))
            done += 1
            if not quiet and done % 500 == 0:
                print(f"  ... {done}/{len(paths)}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_scan_one, str(p), rlo, rhi) for p in paths]
            for f in as_completed(futs):
                merge(f.result())
                done += 1
                if not quiet and done % 500 == 0:
                    print(f"  ... {done}/{len(paths)}", file=sys.stderr)

    mean = torch.where(cnt > 0, s1 / cnt.clamp(min=1), torch.zeros_like(s1))
    var = (s2 / cnt.clamp(min=1) - mean ** 2).clamp(min=0)
    return {"cnt": cnt, "mean": mean, "std": var.sqrt(), "min": mn, "max": mx,
            "n_by_type": nbt, "errors": errors, "n_res": nres,
            "n_files": len(paths)}


def _build_graph(bonded):
    """Bond adjacency -> all-pairs graph distance (BFS from every atom)."""
    n = bonded.shape[0]
    adj = [set() for _ in range(n)]
    for i, j in zip(*np.nonzero(bonded)):
        adj[i].add(int(j)); adj[j].add(int(i))
    gd = np.full((n, n), 99, dtype=np.int8)
    for s in range(n):
        gd[s, s] = 0
        frontier = [s]
        dist = 0
        seen = {s}
        while frontier:
            dist += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); gd[s, v] = dist; nxt.append(v)
            frontier = nxt
    return gd, adj


def _ideal_intra_group_distances():
    """Bond lengths implied by rigid_group_atom_positions, for cross-checking."""
    out = {}
    table = rc.rigid_group_atom_positions
    for name, entries in table.items():
        groups = {}
        for atom, grp, xyz in entries:
            groups.setdefault(grp, {})[atom] = np.asarray(xyz, dtype=np.float64)
        d = {}
        for g in groups.values():
            names = sorted(g)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    n1, n2 = names[a], names[b]
                    dist = float(np.linalg.norm(g[n1] - g[n2]))
                    if dist < BOND_CUTOFF:
                        d[tuple(sorted((n1, n2)))] = dist
        out[name] = d
    return out


def build_tables(stats, rlo, rhi, tol_floor, tol_k, min_count,
                 min_frac=MIN_FRAC, include_unknown=False):
    """Return target/mask/tol/graph_dist in full atom65 layout, plus per-type info."""
    R = rhi - rlo + 1
    target = np.zeros((rc.restype_num + 1, NATOM, NATOM), dtype=np.float32)
    mask = np.zeros_like(target, dtype=bool)
    tol = np.zeros_like(target)
    gdist = np.full_like(target, 99, dtype=np.int8)
    names = _idx_to_name()
    info = {}

    for k in range(R):
        rt = rlo + k
        name = names.get(rt, f"idx{rt}")
        n_res_type = float(stats["n_by_type"][k])
        cnt = stats["cnt"][k].numpy()
        mean = stats["mean"][k].numpy()
        std = stats["std"][k].numpy()
        lo = stats["min"][k].numpy()
        hi = stats["max"][k].numpy()

        m2 = np.zeros((NATOM, NATOM), dtype=np.float64)
        c2 = np.zeros_like(m2)
        s2 = np.zeros_like(m2)
        lo2 = np.zeros_like(m2)
        hi2 = np.zeros_like(m2)
        for p in range(NPAIR):
            i, j = int(PAIR_I[p]), int(PAIR_J[p])
            m2[i, j] = m2[j, i] = mean[p]
            c2[i, j] = c2[j, i] = cnt[p]
            s2[i, j] = s2[j, i] = std[p]
            lo2[i, j] = lo2[j, i] = lo[p]
            hi2[i, j] = hi2[j, i] = hi[p]

        # Coverage relative to this type's residue count is the binding gate. An
        # absolute count admits pairs seen only in a handful of broken entries:
        # at full scale a spurious G "bond" had 101 observations out of 226,824
        # residues (0.045%) and still passed min_count=50, giving G 26 bonds
        # instead of 25. Real bonds sit at 96-98% coverage.
        need = max(min_count, min_frac * n_res_type)
        trusted = c2 >= need
        if not include_unknown and name in UNKNOWN_NAMES:
            trusted = np.zeros_like(trusted)
        bonded = trusted & (m2 > 0) & (m2 < BOND_CUTOFF)
        np.fill_diagonal(bonded, False)
        gd, adj = _build_graph(bonded)

        keep = trusted & (gd <= 2) & (gd > 0)
        t = np.where(keep, m2, 0.0)
        tl = np.where(keep, np.maximum(tol_floor, tol_k * s2), 0.0)
        target[rt] = t.astype(np.float32)
        mask[rt] = keep
        tol[rt] = tl.astype(np.float32)
        gdist[rt] = np.where(keep, gd, 99).astype(np.int8)

        nb = int(bonded.sum() // 2)
        n13 = int((keep & (gd == 2)).sum() // 2)
        info[name] = {
            "n_atoms": int(sum(1 for v in adj if v)),
            "n_bonds": nb, "n_13": n13,
            "bond_mean": m2[bonded], "bond_std": s2[bonded],
            "bond_cnt": c2[bonded],
            "gd": gd, "bonded": bonded, "m2": m2, "s2": s2, "c2": c2,
            "lo2": lo2, "hi2": hi2,
            "adj": adj,
        }
    return target, mask, tol, gdist, info


def report(info, stats, rlo, rhi, tol_floor, tol_k, min_count, ideal,
           min_frac=MIN_FRAC, include_unknown=False):
    print()
    print("=" * 78)
    print("Derived restraints (bond-graph distance <= 2, torsion-independent)")
    print("=" * 78)
    print(f"residues scanned: {stats['n_res']:,}   files: {stats['n_files']:,}")
    print(f"bond cutoff {BOND_CUTOFF} A, pair cutoff {PAIR_CUTOFF} A, "
          f"tol = max({tol_floor}, {tol_k}*std)")
    print(f"trust gate: count >= max({min_count}, {min_frac} * n_res_of_type)"
          f"   unknown bases {'included' if include_unknown else 'excluded'}")
    print()
    hdr = (f"{'restype':<8s} {'atoms':>6s} {'bonds':>6s} {'1-3':>6s} "
           f"{'bond mean':>10s} {'bond std':>9s} {'max std':>8s} "
           f"{'min cnt':>9s} {'cover':>7s}")
    print(hdr); print("-" * len(hdr))
    for k, (name, d) in enumerate(info.items()):
        nrt = float(stats["n_by_type"][k]) if k < len(stats["n_by_type"]) else 0.0
        if not d["n_bonds"]:
            print(f"{name:<8s} {d['n_atoms']:>6d} {0:>6d} {0:>6d} "
                  f"{'-':>10s} {'-':>9s} {'-':>8s} {'-':>9s} {'-':>7s}")
            continue
        bm, bs = d["bond_mean"], d["bond_std"]
        mc = int(d["bond_cnt"].min())
        cov = f"{100.0 * mc / nrt:.1f}%" if nrt else "-"
        print(f"{name:<8s} {d['n_atoms']:>6d} {d['n_bonds']:>6d} {d['n_13']:>6d} "
              f"{bm.mean():>10.4f} {bs.mean():>9.4f} {bs.max():>8.4f} "
              f"{mc:>9,d} {cov:>7s}")

    # Acceptance gate: a connected bond graph must satisfy
    # bonds == atoms - 1 + independent cycles. Excess means spurious edges
    # (rare broken entries whose pair mean fell under the bond cutoff);
    # shortfall means a fractured graph. Both invalidate the table.
    print()
    print("=" * 78)
    print("Acceptance: bonds == atoms - 1 + cycles  (sugar 1 + purine 2 / pyrimidine 1)")
    print("=" * 78)
    n_fail = 0
    for name, d in info.items():
        if name not in EXPECTED_CYCLES:
            print(f"  {name:<4s} skipped (unknown base, no expected cycle count)")
            continue
        exp = d["n_atoms"] - 1 + EXPECTED_CYCLES[name]
        got = d["n_bonds"]
        if got == exp:
            print(f"  {name:<4s} atoms {d['n_atoms']:>2d}  bonds {got:>2d}  "
                  f"expected {exp:>2d}  PASS")
        else:
            n_fail += 1
            kind = "SPURIOUS EDGES" if got > exp else "FRACTURED GRAPH"
            print(f"  {name:<4s} atoms {d['n_atoms']:>2d}  bonds {got:>2d}  "
                  f"expected {exp:>2d}  FAIL ({got - exp:+d}, {kind})")
    if n_fail:
        print(f"\n  {n_fail} restype(s) FAILED -- do not use this table. "
              f"Raise --min-frac, or inspect the low-coverage pairs in the CSV.")
    else:
        print("\n  All standard restypes PASS.")

    print()
    print("=" * 78)
    print("Cross-check: derived bond lengths vs rigid_group_atom_positions")
    print("=" * 78)
    print("(only intra-group bonds are comparable; the sugar ring is cross-group")
    print(" and therefore absent from the ideal table -- that is the gap being filled)")
    any_diff = False
    for name, d in info.items():
        ref = ideal.get(name)
        if not ref or not d["n_bonds"]:
            continue
        rows = []
        for (a, b), v in sorted(ref.items()):
            if a not in rc.atom_order or b not in rc.atom_order:
                continue
            i, j = rc.atom_order[a], rc.atom_order[b]
            got = d["m2"][i, j]
            if got <= 0:
                rows.append((a, b, v, float("nan"), "not observed"))
                continue
            diff = got - v
            flag = "OK" if abs(diff) <= 0.05 else "DIFF"
            if flag == "DIFF":
                any_diff = True
            rows.append((a, b, v, got, f"{diff:+.4f} {flag}"))
        print(f"\n  {name}: {len(rows)} comparable bonds")
        for a, b, v, got, note in rows:
            gs = f"{got:.4f}" if got == got else "   -  "
            print(f"    {a:>4s}-{b:<4s} ideal {v:.4f}  derived {gs}  {note}")
    if not any_diff:
        print("\n  All comparable bonds agree within 0.05 A.")

    print()
    print("=" * 78)
    print("Stereocentres (distance restraints do NOT fix chirality)")
    print("=" * 78)
    for name, d in info.items():
        adj = d["adj"]
        if not adj:
            continue
        centres = []
        for i in range(NATOM):
            if len(adj[i]) >= 3:
                centres.append(rc.atom_types[i])
        if centres:
            print(f"  {name}: {len(centres)} atoms with >=3 bonds: {' '.join(centres)}")
    print("  A separate signed-volume term is required to pin handedness.")


def write_csv(path, info, mask, target, tol, gdist, rlo, rhi):
    names = _idx_to_name()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restype", "atom_i", "atom_j", "graph_dist", "target_A",
                    "tol_A", "observed_std_A", "observed_min_A", "observed_max_A",
                    "n_obs"])
        for k in range(rhi - rlo + 1):
            rt = rlo + k
            name = names.get(rt, f"idx{rt}")
            d = info[name]
            for p in range(NPAIR):
                i, j = int(PAIR_I[p]), int(PAIR_J[p])
                if not mask[rt, i, j]:
                    continue
                # min/max distinguish outlier pollution (max far above target)
                # from genuine spread (max close to target), which decides
                # whether a loose bond needs filtering or a wider tolerance.
                w.writerow([name, rc.atom_types[i], rc.atom_types[j],
                            int(gdist[rt, i, j]), f"{target[rt, i, j]:.4f}",
                            f"{tol[rt, i, j]:.4f}", f"{d['s2'][i, j]:.4f}",
                            f"{d['lo2'][i, j]:.4f}", f"{d['hi2'][i, j]:.4f}",
                            int(d["c2"][i, j])])
    print(f"\nCSV written to {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Derive torsion-independent distance restraints from the cache")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--restypes", choices=["na", "dna", "rna", "all"], default="na")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC,
                    help="trust a pair only if seen in at least this fraction of "
                         "its restype's residues. This is the binding gate at "
                         "full scale, where an absolute count is meaningless")
    ap.add_argument("--include-unknown", action="store_true",
                    help="also build restraints for DX/X; their atom sets are "
                         "inconsistent so their bond graphs fracture")
    ap.add_argument("--tol-floor", type=float, default=0.02,
                    help="minimum tolerance in A (covalent tolerance)")
    ap.add_argument("--tol-k", type=float, default=2.0,
                    help="tolerance = max(tol_floor, tol_k * observed std)")
    ap.add_argument("--out", default="geometry_restraints.npz")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = []
    for raw in args.paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.pt")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"WARN: skipping missing path {p}", file=sys.stderr)
    if args.limit is not None:
        paths = random.Random(args.seed).sample(paths, min(args.limit, len(paths)))
    if not paths:
        sys.exit("ERROR: no .pt files found")

    rlo, rhi = _restype_range(args.restypes)
    if not args.quiet:
        print(f"{len(paths)} files, restypes {args.restypes} (idx {rlo}..{rhi}), "
              f"workers={args.workers}")

    stats = _collect(paths, args.workers, rlo, rhi, args.quiet)
    if stats["n_res"] == 0:
        sys.exit("ERROR: no residues of the requested types found")

    target, mask, tol, gdist, info = build_tables(
        stats, rlo, rhi, args.tol_floor, args.tol_k, args.min_count,
        args.min_frac, args.include_unknown)
    report(info, stats, rlo, rhi, args.tol_floor, args.tol_k, args.min_count,
           _ideal_intra_group_distances(), args.min_frac, args.include_unknown)

    np.savez_compressed(args.out, target=target, mask=mask, tol=tol,
                        graph_dist=gdist, rlo=rlo, rhi=rhi,
                        bond_cutoff=BOND_CUTOFF, tol_floor=args.tol_floor,
                        tol_k=args.tol_k, min_frac=args.min_frac)
    print(f"\nRestraint tables written to {args.out}")
    print(f"  constrained pairs total: {int(mask.sum() // 2):,}")
    if args.csv:
        write_csv(args.csv, info, mask, target, tol, gdist, rlo, rhi)
    if stats["errors"]:
        print(f"\n{len(stats['errors'])} files failed; first 5:", file=sys.stderr)
        for e in stats["errors"][:5]:
            print(f"  {e}", file=sys.stderr)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
