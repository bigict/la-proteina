"""Rescale the nucleic-acid rows of the decoder output heads to the protein scale.

The checkpoint-expansion tool initialized appended weight blocks with
``torch.randn`` (std 1) instead of ``nn.Linear``'s default
``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` (std ``1/sqrt(3*fan_in)``), making the new
rows ~48x too large at fan_in 768. A staged curriculum defers rather than removes
this: an early protein-only stage gives the nucleic-acid rows no gradient, so the
oversized start survives into the stage that first trains them.

Measured on a trained checkpoint: protein row norm 0.658, nucleic-acid row norm
21.957 -- still 33x apart after a week of training. An oversized row norm makes
the head's output sensitivity proportional to |W|, so input variation is
amplified even when the mean output has been pulled back to the right scale.

This script scales only the nucleic-acid rows, by a factor derived from the
checkpoint's own measured norms, and writes a NEW file. Every other tensor is
copied byte-for-byte.

IMPORTANT: scaling W also scales the output, so the head's mean prediction
shrinks by the same factor. This is NOT a drop-in replacement -- it must be
followed by fine-tuning so the useful component regrows at the corrected
amplitude. Treat the result as a starting point for a short run, not a finished
model.

Usage:
    python script_utils/rescale_na_head.py IN.ckpt OUT.ckpt [--logits] [--dry-run]

The training entry point loads ``pretrain_ckpt_path`` via
``model.load_state_dict(ckpt["state_dict"], strict=False)`` and only when the run
directory has no ``last`` checkpoint yet. So point ``pretrain_ckpt_path`` at the
new file AND use a fresh run directory (a new ``run_name_``), otherwise the
existing ``last.ckpt`` takes precedence and this file is never read.
"""

import argparse
import sys
from pathlib import Path

import torch

from openfold.np import residue_constants as rc

STRUCT_SUFFIX = "decoder.struct_linear.1.weight"
LOGIT_SUFFIX = "decoder.logit_linear.1.weight"


def _find_key(state, suffix):
    """Locate exactly one state_dict key ending in suffix."""
    hits = [k for k in state if k.endswith(suffix)]
    if len(hits) != 1:
        sys.exit(
            f"ERROR: expected exactly one key ending in {suffix!r}, found {len(hits)}: "
            f"{hits[:5]}. Inspect the checkpoint's keys before proceeding."
        )
    return hits[0]


def _row_norm_stats(weight, lo, hi):
    """Mean/median L2 norm of rows [0, lo) and [lo, hi)."""
    if weight.ndim != 2 or weight.shape[0] != hi:
        sys.exit(
            f"ERROR: expected a 2-D weight with {hi} rows, got {tuple(weight.shape)}"
        )
    ref = weight[:lo].norm(dim=1)
    new = weight[lo:hi].norm(dim=1)
    return {
        "ref_mean": float(ref.mean()), "ref_median": float(ref.median()),
        "new_mean": float(new.mean()), "new_median": float(new.median()),
    }


def _report(label, before, after, factor):
    print(f"\n{label}")
    print(f"  {'':10s}{'before':>12s}{'after':>12s}")
    print(f"  {'ref mean':10s}{before['ref_mean']:>12.4f}{after['ref_mean']:>12.4f}")
    print(f"  {'new mean':10s}{before['new_mean']:>12.4f}{after['new_mean']:>12.4f}")
    print(f"  {'new/ref':10s}{before['new_mean']/before['ref_mean']:>11.2f}x"
          f"{after['new_mean']/after['ref_mean']:>11.2f}x")
    print(f"  applied factor {factor:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="input checkpoint (never modified)")
    ap.add_argument("dst", help="output checkpoint (must not exist)")
    ap.add_argument("--logits", action="store_true",
                    help="also rescale the nucleic-acid restype rows of logit_linear")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the factor and norms without writing anything")
    ap.add_argument("--force", action="store_true", help="allow overwriting dst")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        sys.exit(f"ERROR: no such file: {src}")
    # Never write back to the source, even via a symlink or a relative alias.
    if dst.exists() and dst.resolve() == src.resolve():
        sys.exit("ERROR: dst resolves to src; refusing to modify the input")
    if dst.exists() and not (args.force or args.dry_run):
        sys.exit(f"ERROR: {dst} exists; pass --force to overwrite")

    # Row layout comes from the residue constants, not hardcoded indices.
    if "P" not in rc.atom_order:
        sys.exit("ERROR: nucleic acids are disabled in this build (no 'P' atom); "
                 "run with proteina_restype_list including DNA/RNA")
    n_prot_atoms = rc.atom_order["P"]          # first nucleic-acid atom index
    n_atoms = rc.atom_type_num
    struct_lo, struct_hi = n_prot_atoms * 3, n_atoms * 3
    print(f"atom_type_num={n_atoms}, protein atoms=0..{n_prot_atoms-1}, "
          f"NA atoms={n_prot_atoms}..{n_atoms-1}")
    print(f"struct rows: protein [0:{struct_lo}), NA [{struct_lo}:{struct_hi})")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        sys.exit("ERROR: checkpoint has no 'state_dict'; this tool expects a "
                 "Lightning checkpoint")
    state = ckpt["state_dict"]

    s_key = _find_key(state, STRUCT_SUFFIX)
    W = state[s_key]
    before = _row_norm_stats(W, struct_lo, struct_hi)
    if before["new_mean"] <= 0:
        sys.exit("ERROR: nucleic-acid rows have zero norm; nothing to rescale")
    factor = before["ref_mean"] / before["new_mean"]
    ratio = before["new_mean"] / before["ref_mean"]
    print(f"\n{s_key} {tuple(W.shape)}")
    print(f"  measured NA/protein row-norm ratio: {ratio:.2f}x")
    if ratio < 1.5:
        print("  NOTE: ratio is already near 1 -- the scale defect is not present "
              "in this checkpoint. Rescaling would do more harm than good.")

    # Scale a clone so the loaded source tensor is never mutated in place.
    new_W = W.clone()
    new_W[struct_lo:struct_hi] *= factor
    state[s_key] = new_W
    after = _row_norm_stats(new_W, struct_lo, struct_hi)
    _report(s_key, before, after, factor)

    if args.logits:
        l_key = _find_key(state, LOGIT_SUFFIX)
        L = state[l_key]
        lo = rc.dna_from_idx
        hi = rc.restype_num
        if lo <= 0:
            sys.exit("ERROR: rc.dna_from_idx is unset; cannot locate NA restype rows")
        print(f"\n{l_key} {tuple(L.shape)}  protein [0:{lo}), NA [{lo}:{hi})")
        lb = _row_norm_stats(L, lo, hi)
        lf = lb["ref_mean"] / lb["new_mean"]
        new_L = L.clone()
        new_L[lo:hi] *= lf
        state[l_key] = new_L
        _report(l_key, lb, _row_norm_stats(new_L, lo, hi), lf)

    # Everything else in the checkpoint is left untouched. The pretrain load path
    # reads only state_dict, so optimizer/scheduler entries are irrelevant here.
    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"\nWrote {dst}")
    print(f"  source unchanged: {src}")
    print("\nNext: fine-tune from this file. Scaling shrank the head's output by "
          f"{factor:.4f}, so the mean prediction must regrow -- expect a transient "
          "rise in struct loss before geometry improves.")
    print("Point pretrain_ckpt_path at the new file AND use a fresh run_name_, "
          "or an existing last.ckpt will take precedence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
