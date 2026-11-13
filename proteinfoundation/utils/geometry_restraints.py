"""Torsion-independent distance restraints for nucleic-acid local geometry.

Coordinate L2 and LDDT supply no sharp signal at chemical precision: LDDT's
smallest threshold is 0.5 A, so a 0.1 A bond error is essentially free, and
coordinate L2 needs 0.01 A per-atom accuracy to hold a bond to 0.02 A, where its
contribution is ~1e-6 nm^2 and drowned in the total. This term adds a flat-bottom
penalty directly on the distances that define bond lengths and angles.

Only pairs at bond-graph distance 1 (bonds) and 2 (1-3, which fix angles via the
law of cosines) are restrained. Both are independent of every torsion, so the
term cannot fight legitimate conformational freedom such as sugar pucker or
syn/anti. Graph distance >= 3 is deliberately absent from the table.

Distances do not determine handedness -- a mirror image has identical pairwise
distances -- so a separate signed-volume term is still needed for chirality.

The table is produced offline by script_utils/build_geometry_restraints.py and
carries target / mask / tol in angstrom, indexed [restype, atom_i, atom_j].
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from openfold.np import residue_constants as rc

ANG_PER_NM = 10.0
EPS = 1e-6


class GeometryRestraintTable:
    """Compact pair list extracted from the dense [restype, atom, atom] table.

    The dense table is 31x65x65 but only ~437 pairs are restrained across all
    restypes, so the union of active (i, j) is a few dozen. Computing just those
    avoids a full 65x65 distance matrix per residue.
    """

    def __init__(self, path: str):
        z = np.load(Path(path))
        target, mask, tol = z["target"], z["mask"], z["tol"]
        if target.shape != mask.shape or target.shape != tol.shape:
            raise ValueError(f"inconsistent table shapes: {target.shape} "
                             f"{mask.shape} {tol.shape}")
        upper = np.triu(mask.any(axis=0), k=1)
        ii, jj = np.nonzero(upper)
        if len(ii) == 0:
            raise ValueError(f"no restrained pairs in {path}")
        self.pair_i = torch.as_tensor(ii, dtype=torch.long)
        self.pair_j = torch.as_tensor(jj, dtype=torch.long)
        # Stored in nm so every downstream quantity matches the other losses.
        self.target = torch.as_tensor(target[:, ii, jj], dtype=torch.float32) / ANG_PER_NM
        self.tol = torch.as_tensor(tol[:, ii, jj], dtype=torch.float32) / ANG_PER_NM
        self.valid = torch.as_tensor(mask[:, ii, jj], dtype=torch.bool)
        self.n_pairs = int(len(ii))
        self.n_restrained = int(mask.sum() // 2)

    def to(self, device) -> "GeometryRestraintTable":
        self.pair_i = self.pair_i.to(device)
        self.pair_j = self.pair_j.to(device)
        self.target = self.target.to(device)
        self.tol = self.tol.to(device)
        self.valid = self.valid.to(device)
        return self


def _is_na(residue_type: Tensor) -> Bool[Tensor, "..."]:
    lo, hi = rc.dna_from_idx, rc.rna_to_idx
    if lo < 0 or hi < 0:  # DNA/RNA disabled in this build
        return torch.zeros_like(residue_type, dtype=torch.bool)
    return (residue_type >= lo) & (residue_type <= hi)


def compute_geom_restraint_loss(
    coors_nm: Float[Tensor, "b n a 3"],
    residue_type: torch.Tensor,
    coord_mask: Bool[Tensor, "b n a"],
    residue_mask: Bool[Tensor, "b n"],
    table: GeometryRestraintTable,
    weight: float,
    na_only: bool = True,
) -> Dict[str, Float[Tensor, "b"]]:
    """Flat-bottom penalty on restrained distances, plus logging-only diagnostics.

    pen = max(0, |d - target| - tol)^2, summed over pairs and divided by the
    number of residues the term applies to. Zero gradient inside tolerance, so
    capacity is not spent pushing 0.001 A below an already-satisfied bond.

    Args:
        coors_nm: predicted coordinates in nm, [b, n, atom_type_num, 3]
        residue_type: ground-truth restype index, [b, n]
        coord_mask: ground-truth atom presence, [b, n, atom_type_num]
        residue_mask: valid residues, [b, n]
        table: loaded restraint table
        weight: multiplies the returned geom loss; 0 disables it
        na_only: restrict to nucleic-acid residues (protein geometry is already
            well learned by the pretrained model)

    Returns:
        Dict of per-batch-element tensors. Keys ending in "_justlog" are excluded
        from the total loss by the caller's convention.
    """
    pi, pj = table.pair_i, table.pair_j
    d = torch.linalg.norm(coors_nm[..., pi, :] - coors_nm[..., pj, :], dim=-1)

    rt = residue_type.clamp(0, table.target.shape[0] - 1)
    target = table.target[rt]  # [b, n, P]
    tol = table.tol[rt]
    active = table.valid[rt] & residue_mask[..., None]
    active = active & coord_mask[..., pi] & coord_mask[..., pj]
    if na_only:
        active = active & _is_na(residue_type)[..., None]

    dev = (d - target).abs()
    excess = torch.clamp(dev - tol, min=0.0)
    pen = (excess ** 2) * active

    denom = active.any(dim=-1).sum(dim=-1).to(pen.dtype)  # residues with >=1 pair
    loss_b = pen.sum(dim=(-1, -2)) / (denom + EPS)

    n_active = active.sum(dim=(-1, -2)).to(pen.dtype)
    n_viol = (active & (excess > 0)).sum(dim=(-1, -2)).to(pen.dtype)
    excess_ang = excess * ANG_PER_NM * active
    out = {
        "geom": loss_b * weight,
        "geom_viol_frac_justlog": n_viol / (n_active + EPS),
        "geom_excess_mean_ang_justlog": excess_ang.sum(dim=(-1, -2)) / (n_viol + EPS),
        "geom_excess_max_ang_justlog": excess_ang.amax(dim=(-1, -2)),
    }
    return out
