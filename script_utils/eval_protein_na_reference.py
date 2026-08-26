#!/usr/bin/env python3
"""Evaluate generated protein--DNA complexes against matching GT PDBs."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
DNA = {"DA", "DC", "DG", "DT", "DI", "A", "C", "G", "T"}
BB = ("P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'")


def chains(path):
    model = next(PDBParser(QUIET=True).get_structure("x", str(path)).get_models())
    protein, dna = [], []
    for chain in model:
        names = {r.resname.strip().upper() for r in chain if not r.id[0].strip()}
        if names <= AA and names:
            protein.append(chain.id)
        elif names <= DNA and names:
            dna.append(chain.id)
    # TODO: support complexes with multiple protein or nucleic-acid chains.
    if len(protein) != 1 or len(dna) != 2:
        raise ValueError(f"expected 1 protein + 2 DNA chains, got {protein} + {dna}")
    return protein[0], dna


def residues(path, chain_id):
    model = next(PDBParser(QUIET=True).get_structure("x", str(path)).get_models())
    result = []
    for residue in model[chain_id]:
        if residue.id[0].strip() or residue.id[2] != " ":
            continue
        result.append((residue.resname.strip().upper(), {
            a.get_name().strip(): np.asarray(a.coord, dtype=float)
            for a in residue.get_unpacked_list() if not a.get_name().strip().startswith("H")
        }))
    return result


def kabsch(x, y):
    if len(x) < 3:
        raise ValueError("fewer than 3 protein CA pairs")
    cx, cy = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - cx).T @ (y - cy))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, cy - cx @ r


def contacts(protein, dna, cutoff=4.5):
    p = [(i, xyz) for i, (_, atoms) in enumerate(protein) for xyz in atoms.values()]
    n = [(i, xyz) for i, (_, atoms) in enumerate(dna) for xyz in atoms.values()]
    if not p or not n:
        return set()
    d = np.linalg.norm(np.asarray([x[1] for x in p])[:, None] - np.asarray([x[1] for x in n]), axis=-1)
    return {(p[i][0], n[j][0]) for i, j in zip(*np.where(d <= cutoff))}


def f1(pred, true):
    hit = len(pred & true)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(true) if true else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(generated, reference):
    gp, gd = chains(generated)
    rp, rd = chains(reference)
    gpro, rpro = residues(generated, gp), residues(reference, rp)
    gna, rna = [residues(generated, c) for c in gd], [residues(reference, c) for c in rd]
    if len(gpro) != len(rpro) or any(len(g) != len(r) for g, r in zip(gna, rna)):
        raise ValueError("generated/reference chain lengths differ")
    pairs = [(g[1]["CA"], r[1]["CA"]) for g, r in zip(gpro, rpro) if "CA" in g[1] and "CA" in r[1]]
    rotation, translation = kabsch(*map(np.asarray, zip(*pairs)))
    backbone_g, backbone_r, all_g, all_r = [], [], [], []
    native_map, generated_map = set(), set()
    for chain_i, (gchain, rchain) in enumerate(zip(gna, rna)):
        native_map |= {(p, (chain_i, n)) for p, n in contacts(rpro, rchain)}
        generated_map |= {(p, (chain_i, n)) for p, n in contacts(gpro, gchain)}
        for g, r in zip(gchain, rchain):
            for name in BB:
                if name in g[1] and name in r[1]:
                    backbone_g.append(g[1][name] @ rotation + translation)
                    backbone_r.append(r[1][name])
            if g[0] == r[0]:
                for name in set(g[1]) & set(r[1]):
                    all_g.append(g[1][name] @ rotation + translation)
                    all_r.append(r[1][name])
    if not native_map or not backbone_g:
        raise ValueError("reference interface or DNA backbone is empty")
    return {
        "protein_aligned_na_backbone_rmsd": float(np.sqrt(np.mean((np.asarray(backbone_g) - backbone_r) ** 2))),
        "protein_aligned_na_all_atom_rmsd": float(np.sqrt(np.mean((np.asarray(all_g) - all_r) ** 2))) if all_g else float("nan"),
        "protein_interface_f1": f1({p for p, _ in generated_map}, {p for p, _ in native_map}),
        "contact_map_f1": f1(generated_map, native_map),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", default="protein_na_reference_eval.csv")
    args = parser.parse_args()
    rows = []
    for generated in sorted(Path(args.generated_dir).rglob("*.pdb")):
        match = re.search(r"_motif_(.+)$", generated.stem)
        task = match.group(1) if match else generated.parent.name
        reference = Path(args.reference_dir) / f"{task}_gt.pdb"
        row = {"task_id": task, "generated_pdb": str(generated), "reference_pdb": str(reference)}
        try:
            row.update(evaluate(generated, reference))
            row["status"] = "ok"
        except Exception as error:
            row.update(status="error", error=str(error))
        rows.append(row)
    if not rows:
        raise ValueError("no generated PDB files found")
    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {len(rows)} structures; results saved to {args.output}")


if __name__ == "__main__":
    main()
