import torch

from openfold.np import residue_constants as rc

def token_level_w(aatype, w):
    return torch.where(
        (aatype >= rc.dna_from_idx) & (aatype <= rc.dna_to_idx),
        w.get("dna_w", 1.0),
        torch.where(
            (aatype >= rc.rna_from_idx) & (aatype <= rc.rna_to_idx),
            w.get("rna_w", 1.0),
            w.get("prot_w", 1.0)
        )
    )
