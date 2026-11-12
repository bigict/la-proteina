#!/bin/bash
# La-Proteina change-sync check: run this on the training machine. All OK means
# the tree synced completely. Lives in script_utils/ and cd's to the repo root.
#
# Markers use grep -F (fixed strings), not regex: a mis-escaped pattern would
# otherwise fail to match and look exactly like "the code did not sync".
set -u
cd "$(dirname "$0")/.." || { echo "ERROR: cannot cd to the repo root" >&2; exit 2; }

# Sentinel check: a failed lookup must hard-error, never degrade into a wall of
# MISMATCH lines that reads as "unsynced tree".
for sentinel in \
    proteinfoundation/utils/coors_utils.py \
    proteinfoundation/nn/feature_factory.py \
    script_utils/model_ops.py; do
    if [ ! -f "$sentinel" ]; then
        echo "ERROR: $sentinel not found -- current dir is not the repo root ($(pwd))" >&2
        echo "       Place this script under <repo>/script_utils/ and run it again." >&2
        exit 2
    fi
done

fail=0
check() { # check <file> <expected_count> <fixed_string>
    local n
    if [ ! -f "$1" ]; then
        printf 'MISSING   --/%s  %s\n' "$2" "$1"
        fail=1
        return
    fi
    n=$(grep -cF -- "$3" "$1")
    if [ "$n" = "$2" ]; then
        printf 'OK        %s/%s  %s  «%s»\n' "$n" "$2" "$1" "$3"
    else
        printf 'MISMATCH  %s/%s  %s  «%s»\n' "$n" "$2" "$1" "$3"
        fail=1
    fi
}

echo "repo: $(pwd)"
echo

# --- Round 1: checkpoint-expansion init scale (torch.randn -> nn.Linear default / zero-init) ---
check script_utils/model_ops.py 1 'def _linear_default_init'
check script_utils/model_ops.py 0 'torch.randn('
check script_utils/model_ops.py 2 'torch.zeros('
check script_utils/model_ops.py 1 'assert b.shape[1] == enc_fan_in'
check script_utils/model_ops.py 1 'assert b.shape[1] == fan_in == 909'

# --- Round 2: angle binning (periodic torsions, [0,pi] range for bond angles) ---
check proteinfoundation/nn/feature_factory.py 1 'def bin_and_one_hot_periodic'
check proteinfoundation/nn/feature_factory.py 1 'def bin_and_one_hot_bond_angle'
check proteinfoundation/nn/feature_factory.py 1 'N_ANGLE_BINS = 21'
check proteinfoundation/nn/feature_factory.py 0 'linspace(-torch.pi'

# --- Round 3: OP1/OP2 canonicalization at the source ---
check proteinfoundation/utils/coors_utils.py 1 'def canonicalize_phosphate_oxygens'
check proteinfoundation/datasets/transforms.py 1 'class CanonicalizePhosphateOxygensTransform'
check proteinfoundation/utils/motif_utils.py 2 'x_motif = canonicalize_phosphate_oxygens'
check proteinfoundation/utils/pdb_utils.py 1 'atom_positions = canonicalize_phosphate_oxygens'
for c in ucond motif_aa motif_tip; do
    check "configs/dataset/pdb/pdb_train_$c.yaml" 1 'CanonicalizePhosphateOxygensTransform'
done

# --- Upstream d2d31d4 / 37f71e9: struct and seq use sum(w*L)/(sum(w*mask)+eps) ---
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'err = (err * w).sum(dim=-1) / ((mask * w).sum(dim=-1) + eps)'
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'torch.sum(seq_loss * w, dim=-1) / (torch.sum(mask * w, dim=-1) + eps)'

# --- Round 4: log unweighted KL (the normalization itself came from upstream d2d31d4) ---
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'nres = torch.sum(mask, dim=-1)'
check proteinfoundation/partial_autoencoder/autoencoder.py 0 'mask > 0'
# The weight belongs only at the compute_kl_penalty call site; the logging path
# must use the unweighted mask.
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'token_level_w(aatype, self.cfg_ae.loss.kl)'

# --- Tooling: OP1/OP2 flip-rate statistics ---
# Must import the production rule. A count of 0 here means someone reimplemented
# the rule inside the script, decoupling the statistic from the training pipeline.
check script_utils/stat_op1_op2_flip.py 1 'from proteinfoundation.utils.coors_utils import canonicalize_phosphate_oxygens'
# Per-entry distribution: separates "different between entries" from "random within".
check script_utils/stat_op1_op2_flip.py 1 'def _entry_bucket'
check script_utils/stat_op1_op2_flip.py 1 'def _select_entries'
check script_utils/stat_op1_op2_flip.py 1 'def _report_entries'
check script_utils/stat_op1_op2_flip.py 1 'def _write_entry_csv'
check script_utils/stat_op1_op2_flip.py 1 '"--min-na-res"'
check script_utils/stat_op1_op2_flip.py 1 '"--by-entry-csv"'
# The denominator must be the decidable count (one site in _select_entries, one
# in _write_entry_csv); otherwise missing-atom/near-planar residues dilute the
# rate toward 0% and fabricate within-entry consistency.
check script_utils/stat_op1_op2_flip.py 2 'n_dec = e["n_na"] - e["n_skip"]'
# --na-only: exclude protein-NA complexes as whole entries
check script_utils/stat_op1_op2_flip.py 1 '"--na-only"'
check script_utils/stat_op1_op2_flip.py 1 'is_prot = (rtype >= rc.prot_from_idx) & (rtype <= rc.prot_to_idx)'
# placeholder coords must not be mistaken for broken bond lengths
check script_utils/stat_op1_op2_flip.py 1 'FILL_VALUE_COORDS = 1e-5'
check script_utils/stat_op1_op2_flip.py 1 'valid = present & ~is_fill'
# P-O bond-length sanity window
check script_utils/stat_op1_op2_flip.py 1 'P_O_MIN, P_O_MAX = 1.2, 2.0'
# flip% denominator is the decidable count in both tables
check script_utils/stat_op1_op2_flip.py 2 'nd = n - mi - dg'
# the eps advice must point at RAISING eps, never lowering it
check script_utils/stat_op1_op2_flip.py 0 'eps too large, lower it'
check script_utils/stat_op1_op2_flip.py 1 'Consider --eps'

echo
if [ "$fail" = 0 ]; then
    echo "==> All markers in sync"
else
    echo "==> MISMATCH/MISSING present: tree is not fully synced, do not start training"
fi
exit "$fail"
