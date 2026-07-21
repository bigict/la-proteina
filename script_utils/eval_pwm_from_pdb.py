#!/usr/bin/env python
"""
Evaluate PWM constructed from N sampled PDB structures against JASPAR ground truth.

Pipeline:
    N PDBs (with DNA double-strand)
        → extract DNA sequences from each PDB
        → build PWM by counting base frequencies
        → align with JASPAR PWM (IC-weighted PCC scoring)
        → compute DeepPBS metrics: MAE, IC_corr, brier_multi, IC_diff

Input file format (one protein per line, '#' for comments):
    protein_id,jaspar_pwm_id,pdb_dir
    protein_id,jaspar_pwm_id,pdb_file1;pdb_file2;pdb_file3

Examples:
    python eval_pwm_from_pdb.py input.txt -o results/
    python eval_pwm_from_pdb.py input.txt -o results/ --pdb_root /data/sampled/
"""

import argparse
import json
import csv
import functools
import os
import sys
import glob
import pickle
import logging
import multiprocessing as mp
import numpy as np
from scipy.stats import pearsonr, entropy

from Bio.PDB import PDBParser, MMCIFParser

from openfold.np import residue_constants as rc
from proteinfoundation.utils import pwm_utils
assert rc.DNA in rc.restype_list

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# =====================================================================
# 1. DNA Sequence Extraction from PDB
# =====================================================================

STANDARD_DNA = set(
    rc.restype_1to3[(k, rc.DNA)] for k in rc.restypes[rc.dna_from_idx: rc.dna_to_idx]
)


def _fix_resname(rname):
    """Normalize modified nucleotide names to standard."""
    if rname == "DI":
        return "DG"
    if rname == "DU":
        return "DT"
    return rname


def _is_dna_residue(resname):
    if resname in STANDARD_DNA:
        return True
    return False


def _resname_to_base(resname):
    """DA→A, DC→C, DG→G, DT→T (handles modified nucleotides)."""
    fixed = _fix_resname(resname)
    if fixed in ("DA", "DC", "DG", "DT"):
        return fixed[-1]
    raise ValueError(f"Cannot map residue '{resname}' to a DNA base")


def _parse_structure(pdb_file):
    """Parse PDB/mmCIF file, return Biopython Structure object."""
    ext = os.path.splitext(pdb_file)[1].lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("sample", pdb_file)


def _get_c1p_coord(residue):
    """Get C1' atom coordinate from a nucleotide residue."""
    for name in ("C1'", "C1*"):
        if name in residue:
            return residue[name].coord
    return None


def _find_paired_residues(chain_a, chain_b, max_dist=15.0):
    """
    Identify base-paired residues between two DNA chains by C1'-C1' distance.

    Watson-Crick C1'-C1' distance ≈ 10.5Å; we use 15Å as generous cutoff.
    Uses greedy nearest-neighbor matching to avoid one-to-many assignments.

    Returns
    -------
    paired_a : list of residues from chain_a that have a pair
    paired_b : list of corresponding residues from chain_b (same order)
    """
    c1a = [(res, _get_c1p_coord(res)) for res in chain_a]
    c1b = [(res, _get_c1p_coord(res)) for res in chain_b]
    c1a = [(r, c) for r, c in c1a if c is not None]
    c1b = [(r, c) for r, c in c1b if c is not None]

    if not c1a or not c1b:
        return [], []

    # Build distance matrix
    dist = np.zeros((len(c1a), len(c1b)))
    for i, (_, ci) in enumerate(c1a):
        for j, (_, cj) in enumerate(c1b):
            dist[i, j] = np.linalg.norm(ci - cj)

    # Greedy matching: repeatedly pick the closest pair
    used_a, used_b = set(), set()
    pairs = []
    n_pairs = min(len(c1a), len(c1b))
    for _ in range(n_pairs):
        if dist.size == 0:
            break
        idx = np.argmin(dist)
        i, j = divmod(idx, dist.shape[1])
        if dist[i, j] > max_dist:
            break
        pairs.append((c1a[i][0], c1b[j][0]))
        used_a.add(i)
        used_b.add(j)
        dist[i, :] = np.inf
        dist[:, j] = np.inf

    # Sort by chain_a residue number to preserve 5'→3' order
    pairs.sort(key=lambda p: p[0].get_id()[1])
    paired_a = [p[0] for p in pairs]
    paired_b = [p[1] for p in pairs]
    return paired_a, paired_b


def extract_dna_sequences(pdb_file):
    """
    Extract double-strand DNA sequences from a PDB file.

    Strategy: first identify base-paired residues by C1'-C1' spatial distance,
    then extract sequences from paired residues only.  This correctly handles
    cases where the two strands have different lengths (e.g. terminal fraying
    or missing residues) — unpaired bases are excluded before any trimming.

    Returns
    -------
    dict with keys:
        'strand1' : str  — 5'→3' sequence of the first (longer) chain
        'strand2' : str  — 5'→3' sequence of the second chain (reversed)
        'length'  : int  — number of paired base pairs
    or None if no DNA found.
    """
    structure = _parse_structure(pdb_file)

    dna_chains = []
    for model in structure:
        for chain in model:
            residues = []
            for res in chain:
                hetflag = res.get_id()[0]
                resname = res.get_resname().strip()
                is_std = resname in STANDARD_DNA
                is_het = hetflag.startswith("H_") or hetflag == "W"
                if is_std or (is_het and _is_dna_residue(resname)):
                    residues.append(res)

            if len(residues) >= 3:
                residues.sort(key=lambda r: r.get_id()[1])
                dna_chains.append(residues)
        break  # first model only

    if len(dna_chains) < 2:
        return None

    dna_chains.sort(key=len, reverse=True)
    chain_a, chain_b = dna_chains[0], dna_chains[1]

    # ── Pair first: identify base pairs via C1'-C1' distance ─────────
    paired_a, paired_b = _find_paired_residues(chain_a, chain_b)

    if len(paired_a) < 3:
        # Fallback: revert to simple truncation when pairing fails
        logging.warning(f"  base-pairing detection failed "
                        f"({len(paired_a)} pairs), falling back to min-length trim")
        paired_a = chain_a[:min(len(chain_a), len(chain_b))]
        paired_b = chain_b[:min(len(chain_a), len(chain_b))]
    elif len(paired_a) != len(chain_a) or len(paired_b) != len(chain_b):
        logging.info(f"  paired {len(paired_a)} bp "
                     f"(strand lengths: {len(chain_a)}, {len(chain_b)})")

    # ── Then trim: only use successfully paired bases to build sequences ──
    seq1 = "".join(_resname_to_base(r.get_resname()) for r in paired_a)
    # chain_b is the reverse strand (3'→5'), reverse paired result to get 5'→3'
    paired_b_53 = sorted(paired_b, key=lambda r: r.get_id()[1], reverse=True)
    seq2 = "".join(_resname_to_base(r.get_resname()) for r in paired_b_53)

    assert len(seq1) == len(seq2), \
        f"paired sequences length mismatch: {len(seq1)} vs {len(seq2)}"

    return {
        "strand1": seq1,
        "strand2": seq2,
        "length": len(seq1)
    }


# =====================================================================
# 2. PWM Construction from Sequences
# =====================================================================

def build_pwm_from_sequences(sequences):
    """
    Build PWM (position frequency matrix) from a list of equal-length sequences.

    Parameters
    ----------
    sequences : list of str
        DNA sequences (A/C/G/T), all same length L.

    Returns
    -------
    pwm : ndarray, shape (L, 4)
        Each row sums to 1.0.
    """
    L = len(sequences[0])
    counts = np.zeros((L, 4))
    for seq in sequences:
        assert len(seq) == L, \
            f"Sequence length mismatch: expected {L}, got {len(seq)}"
        for i, base in enumerate(seq):
            if base in BASE_MAP:
                counts[i, BASE_MAP[base]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


# =====================================================================
# 3. PWM Loading (JASPAR + H11MO)
# =====================================================================

DATA_PATH = os.getenv(
    "DATA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pwms.pickle")
)

# Cache the pickle to avoid re-loading for every entry
_pwm_cache = {}


def _get_pwm_dict():
    if not _pwm_cache:
        _pwm_cache["data"] = pickle.load(open(DATA_PATH, "rb"))
    return _pwm_cache["data"]


def detect_pwm_source(pwm_id):
    """Detect PWM source from ID format: "jaspar" or "h11mo"."""
    if ".H11MO." in pwm_id:
        return "h11mo"
    return "jaspar"


def load_pwm(pwm_id):
    """
    Load PWM by key (JASPAR or H11MO) and trim low-IC flanking columns.

    Both sources live in the same pwms.pickle and share the same
    Biopython Motif format, so the loading logic is identical.

    Key formats (as stored in pwms.pickle):
        JASPAR : 'MA0162.1.jaspar'   (also accepts 'MA0162.1')
        H11MO  : 'HSF1_HUMAN.H11MO.0.A'

    Returns
    -------
    pwm : ndarray, shape (K, 4)
    """
    pwm_dict = _get_pwm_dict()

    # Resolve key: JASPAR keys in pickle have '.jaspar' suffix
    key = pwm_id
    if key not in pwm_dict:
        key_with_suffix = f"{pwm_id}.jaspar"
        if key_with_suffix in pwm_dict:
            key = key_with_suffix
        else:
            raise KeyError(f"PWM '{pwm_id}' not found in {DATA_PATH}. "
                           f"Tried '{pwm_id}' and '{key_with_suffix}'. "
                           f"Available keys: {len(pwm_dict)} total")

    untrimmed = np.array(list(pwm_dict[key].pwm.values())).T

    start = 0
    for i in range(untrimmed.shape[0]):
        if entropy(untrimmed[i], [0.25] * 4, base=2) > 0.5:
            break
        start = i + 1

    end = untrimmed.shape[0]
    for i in range(untrimmed.shape[0]):
        if entropy(untrimmed[-(i + 1)], [0.25] * 4, base=2) > 0.5:
            break
        end = untrimmed.shape[0] - i - 1

    return untrimmed[start:end]


# =====================================================================
# 4. PWM Alignment (IC-weighted PCC, ungapped)
# =====================================================================

def align_pwm_to_seq(pwm, seq_onehot, global_align=False):
    """
    Align PWM with a one-hot encoded sequence.
    Auto-detects which is longer and swaps arguments.

    Returns
    -------
    seq_start, pwm_start, overlap_len, score
    """
    min_overlap = min(pwm.shape[0], seq_onehot.shape[0]) if global_align else 1
    return  pwm_utils.pwm_seq_align(
        pwm, seq_onehot, min_overlap=min_overlap
    )


# =====================================================================
# 5. Metrics (identical to deeppbs.nn.metrics.metrics)
# =====================================================================

def compute_all_metrics(gt, pred):
    """Compute all four DeepPBS multi-class metrics on aligned overlap."""
    return {
        "mae": pwm_utils.mae(gt, pred),
        "ic_corr": pwm_utils.ic_corr(gt, pred),
        "brier_multi": pwm_utils.brier_multi(gt, pred),
        "ic_diff": pwm_utils.ic_diff(gt, pred),
    }


# =====================================================================
# 6. Main Pipeline
# =====================================================================
def align_pwm_to_pdb(ref_pwm, pdb_root, pdb_file, global_align=False):
    failed = 0
    if True:
        fpath = os.path.join(pdb_root, pdb_file) if pdb_root else pdb_file
        if not os.path.isfile(fpath):
            logging.warning(f"  file not found: {fpath}")
            return False
            # failed += 1
            # continue
        try:
            seqs = extract_dna_sequences(fpath)
            if seqs is None:
                logging.warning(f"  no DNA found: {os.path.basename(fpath)}")
                return False
                # failed += 1
                # continue

            s1 = seqs["strand1"]
            s2 = seqs["strand2"]

            # align each strand with reference PWM
            ss1, ps1, k1, score_s1 = align_pwm_to_seq(
                ref_pwm, rc.sequence_to_onehot(s1, BASE_MAP), global_align=global_align
            )
            ss2, ps2, k2, score_s2 = align_pwm_to_seq(
                ref_pwm, rc.sequence_to_onehot(s2, BASE_MAP), global_align=global_align
            )

            if score_s1 >= score_s2:
                return True, s1[ss1:ss1 + k1], ps1, k1, "strand1"
                aligned_frags.append(s1[ss1:ss1 + k1])
                pwm_starts.append(ps1)
                overlap_len.append(k1)
                strand_counts["strand1"] += 1
            else:
                return True, s2[ss2:ss2 + k2], ps2, k2, "strand2"
                aligned_frags.append(s2[ss2:ss2 + k2])
                pwm_starts.append(ps2)
                overlap_len.append(k2)
                strand_counts["strand2"] += 1

        except Exception as e:
            logging.warning(f"  parse error {os.path.basename(fpath)}: {e}")
            return False
            # failed += 1

def evaluate_protein(protein_id, pwm_id, pdb_files, pdb_root=None, global_align=False):
    """
    Full evaluation for one protein.

    Parameters
    ----------
    protein_id : str
    pwm_id     : str   — PWM key, JASPAR (e.g. 'MA1563.1') or H11MO
                       (e.g. 'HSF1_HUMAN.H11MO.0.A')
    pdb_files  : list of str — paths to sampled PDB files
    pdb_root   : str or None — if set, prepend to relative pdb_files paths

    Returns
    -------
    result dict or None on failure
    """
    pwm_source = detect_pwm_source(pwm_id)
    logging.info(f"═══ {protein_id} | {pwm_source.upper()}={pwm_id} | "
                 f"N={len(pdb_files)} PDBs ═══")

    # ── 1. Load reference PWM ───────────────────────────────────────
    ref_pwm = load_pwm(pwm_id)
    logging.info(f"  reference PWM shape: {ref_pwm.shape} ({pwm_source})")

    # ── 2. Per-PDB: extract both strands, align, keep best + alignment info ──
    aligned_frags = []   # aligned sequence fragments
    pwm_starts = []      # ref PWM start positions for each PDB
    overlap_len = []     # overlap lengths for each PDB
    strand_counts = {"strand1": 0, "strand2": 0}
    failed = 0
    
    with mp.Pool(processes=min(mp.cpu_count(), len(pdb_files))) as p:
        for ok, aligned_frag, pwm_start, k, strand in p.imap(
            functools.partial(
                align_pwm_to_pdb, ref_pwm, pdb_root, global_align=global_align
            ), pdb_files
        ):
            if ok:
                aligned_frags.append(aligned_frag)
                pwm_starts.append(pwm_start)
                overlap_len.append(k)
                strand_counts[strand] += 1
            else:
                failed += 1

    if len(aligned_frags) < 2:
        logging.error(f"  insufficient sequences ({len(aligned_frags)}) for {protein_id}")
        return None

    logging.info(f"  strand selection: strand1={strand_counts['strand1']}, "
                 f"strand2={strand_counts['strand2']}  (failed: {failed})")

    # ── 3. Build PWM by position-weighted accumulation ──────────────
    # Each fragment is placed at its reference PWM coordinate; positions
    # are averaged only over PDBs that cover them.
    ref_len = len(ref_pwm)
    pwm_acc = np.zeros((ref_len, 4))
    pwm_cov = np.zeros(ref_len, dtype=int)

    for frag, ps, k in zip(aligned_frags, pwm_starts, overlap_len):
        oh = rc.sequence_to_onehot(frag, BASE_MAP)  # (k, 4)
        pwm_acc[ps:ps + k] += oh
        pwm_cov[ps:ps + k] += 1

    covered = pwm_cov > 0
    constructed_pwm = np.zeros_like(pwm_acc)
    constructed_pwm[covered] = pwm_acc[covered] / pwm_cov[covered][:, None]
    # uncovered positions: uniform (will not be evaluated)

    logging.info(f"  fragments: {len(aligned_frags)}, "
                 f"lengths: {min(overlap_len)}–{max(overlap_len)}, "
                 f"ref coverage: {covered.sum()}/{ref_len}")

    # ── 4. Compute metrics on covered positions ─────────────────────
    gt = ref_pwm[covered]
    pred = constructed_pwm[covered]
    metrics = compute_all_metrics(gt, pred)

    logging.info(f"  MAE={metrics['mae']:.4f}  IC_corr={metrics['ic_corr']:.4f}  "
                 f"Brier={metrics['brier_multi']:.4f}  IC_diff={metrics['ic_diff']:.4f}")

    return {
        "protein_id": protein_id,
        "pwm_id": pwm_id,
        "pwm_source": pwm_source,
        "num_pdbs": len(aligned_frags),
        "num_failed": failed,
        "strand1_count": strand_counts["strand1"],
        "strand2_count": strand_counts["strand2"],
        "ref_covered": int(covered.sum()),
        "ref_total": ref_len,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PWM from sampled PDBs against reference PWM "
                    "(JASPAR or H11MO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input file format (one protein per line):
    # comment lines start with '#'
    protein_id,pwm_id,pdb_dir

PWM ID formats:
    JASPAR : MA0162.1
    H11MO  : HSF1_HUMAN.H11MO.0.A

Examples:
    # All .pdb/.cif files in a directory:
    python eval_pwm_from_pdb.py input.txt -o results/

    # Override PDB root directory:
    python eval_pwm_from_pdb.py input.txt -o results/ --pdb_root /data/sampled/
        """)

    parser.add_argument("input_file",
                        help="Text file: protein_id,pwm_id,pdb_dir per line")
    parser.add_argument("-o", "--output_dir", default="./eval_output",
                        help="Output directory (default: ./eval_output)")
    parser.add_argument("--pdb_root", default=None,
                        help="Prepend this path to relative PDB paths/dirs")
    parser.add_argument("--job_id", default="",
                        help="pdb file patthens")
    parser.add_argument("--global_align", action="store_true",
                        help="Do global align pwm with sequence.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    for line in open(args.input_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(",")
        if len(parts) < 3:
            logging.warning(f"skip malformed line: {line}")
            continue

        protein_id = parts[0].strip()
        pwm_id = parts[1].strip()
        pdb_input = parts[2].strip()

        # ── resolve PDB file list ───────────────────────────────────
        if args.pdb_root:
            pdb_input = os.path.join(args.pdb_root, pdb_input)

        if os.path.isdir(pdb_input):
            pdb_files = sorted(
                glob.glob(os.path.join(pdb_input, f"**/{args.job_id}*.pdb"))
                + glob.glob(os.path.join(pdb_input, f"**/{args.job_id}*.cif"))
            )
        elif ";" in pdb_input:
            pdb_files = [f.strip() for f in pdb_input.split(";") if f.strip()]
        elif os.path.isfile(pdb_input):
            pdb_files = [pdb_input]
        else:
            logging.warning(f"skip {protein_id}: '{pdb_input}' not found")
            continue

        if not pdb_files:
            logging.warning(f"skip {protein_id}: no PDB files found")
            continue

        result = evaluate_protein(
            protein_id, pwm_id, pdb_files, global_align=args.global_align
        )
        if result:
            results.append(result)

    if not results:
        logging.error("No proteins evaluated successfully.")
        return

    # ── Write outputs ───────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results → {json_path}")

    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    fieldnames = list(results[0].keys())
    metric_names = ["mae", "ic_corr", "brier_multi", "ic_diff"]
    mean_row = {k: "" for k in fieldnames}
    mean_row["protein_id"] = "mean"
    for m in metric_names:
        mean_row[m] = float(np.mean([r[m] for r in results]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(mean_row)
    logging.info(f"Results → {csv_path}")

    # ── Summary ─────────────────────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluated {len(results)} proteins")
    for m in ["mae", "ic_corr", "brier_multi", "ic_diff"]:
        vals = [r[m] for r in results]
        logging.info(f"  {m:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
