#!/usr/bin/env python3
"""Evaluate generated protein--DNA complexes against matching GT PDBs."""

import argparse
import csv
import re
from itertools import permutations
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser

AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
AA = set(AA1)
DNA = {"DA", "DC", "DG", "DT", "DI", "A", "C", "G", "T"}
DNA1 = {name: name[-1] for name in DNA}
BB = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")


def load_complex(path):
    model = next(PDBParser(QUIET=True).get_structure("x", str(path)).get_models())
    protein, dna = [], []
    for chain in model:
        chain_residues = []
        for residue in chain:
            if residue.id[0].strip() or residue.id[2] != " ":
                continue
            chain_residues.append((residue.resname.strip().upper(), {
                atom.get_name().strip(): np.asarray(atom.coord, dtype=float)
                for atom in residue.get_unpacked_list()
                if not atom.get_name().strip().startswith("H")
            }))
        names = {residue[0] for residue in chain_residues}
        if names <= AA and names:
            protein.append(chain_residues)
        elif names <= DNA and names:
            dna.append(chain_residues)
    # TODO: support complexes with multiple protein or nucleic-acid chains.
    if len(protein) != 1 or len(dna) != 2:
        raise ValueError(f"expected 1 protein + 2 DNA chains, got {len(protein)} + {len(dna)}")
    return protein[0], dna


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


def rmsd(x, y):
    return float(np.sqrt(np.mean(np.sum((np.asarray(x) - np.asarray(y)) ** 2, axis=1))))


def align_residues(generated, reference, alphabet):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
    alignment = aligner.align(
        "".join(alphabet[r[0]] for r in generated),
        "".join(alphabet[r[0]] for r in reference),
    )[0]
    return [(i, j) for (a, b), (c, d) in zip(*alignment.aligned)
            for i, j in zip(range(a, b), range(c, d))]


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


def evaluate_dna_pairing(gna, rna, pairing, protein, reference_protein,
                         gen_to_ref, rotation, translation):
    backbone_g, backbone_r, all_g, all_r = [], [], [], []
    native_map, generated_map = set(), set()
    mapped_ref = set(gen_to_ref.values())
    same_base_count = 0
    for generated_i, reference_i in enumerate(pairing):
        generated_chain, reference_chain = gna[generated_i], rna[reference_i]
        dna_pairs = align_residues(generated_chain, reference_chain, DNA1)
        dna_to_ref = dict(dna_pairs)
        mapped_ref_dna = set(dna_to_ref.values())
        native_map |= {(p, (reference_i, n)) for p, n in contacts(reference_protein, reference_chain)
                       if p in mapped_ref and n in mapped_ref_dna}
        generated_map |= {(gen_to_ref[p], (reference_i, dna_to_ref[n]))
                          for p, n in contacts(protein, generated_chain)
                          if p in gen_to_ref and n in dna_to_ref}
        for i, j in dna_pairs:
            generated_residue, reference_residue = generated_chain[i], reference_chain[j]
            for name in BB:
                if name in generated_residue[1] and name in reference_residue[1]:
                    backbone_g.append(generated_residue[1][name])
                    backbone_r.append(reference_residue[1][name])
            if DNA1[generated_residue[0]] == DNA1[reference_residue[0]]:
                same_base_count += 1
                for name in sorted(set(generated_residue[1]) & set(reference_residue[1])):
                    all_g.append(generated_residue[1][name])
                    all_r.append(reference_residue[1][name])
    if not native_map or len(backbone_g) < 3:
        raise ValueError("reference interface or DNA backbone is empty")
    backbone_g = np.asarray(backbone_g)
    backbone_r = np.asarray(backbone_r)
    protein_aligned_backbone = backbone_g @ rotation + translation
    dna_rotation, dna_translation = kabsch(backbone_g, backbone_r)
    return {
        "protein_aligned_na_backbone_rmsd": rmsd(protein_aligned_backbone, backbone_r),
        "protein_aligned_na_same_base_all_atom_rmsd": rmsd(np.asarray(all_g) @ rotation + translation, all_r) if all_g else float("nan"),
        "na_same_base_residue_coverage": same_base_count / sum(map(len, rna)),
        "na_alone_backbone_rmsd": rmsd(backbone_g @ dna_rotation + dna_translation, backbone_r),
        "protein_interface_f1": f1({p for p, _ in generated_map}, {p for p, _ in native_map}),
        "contact_map_f1": f1(generated_map, native_map),
        "dna_chain_pairing": ",".join(map(str, pairing)),
    }


def evaluate(generated, reference):
    gpro, gna = load_complex(generated)
    rpro, rna = load_complex(reference)
    residue_pairs = align_residues(gpro, rpro, AA1)
    pairs = [(gpro[i][1]["CA"], rpro[j][1]["CA"]) for i, j in residue_pairs
             if "CA" in gpro[i][1] and "CA" in rpro[j][1]]
    rotation, translation = kabsch(*map(np.asarray, zip(*pairs)))
    gen_to_ref = dict(residue_pairs)
    candidates = [evaluate_dna_pairing(gna, rna, pairing, gpro, rpro, gen_to_ref,
                                       rotation, translation)
                  for pairing in permutations(range(2))]
    return min(candidates, key=lambda result: result["protein_aligned_na_backbone_rmsd"])


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
