from typing import Dict

import einops
import torch

from openfold.np import residue_constants as rc
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK
from proteinfoundation.nn.feature_factory import FeatureFactory


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(
        dtype=torch.bool, device=device
    )  # [21, 37]


class ResidualLayer(torch.nn.Module):
    """Residual layer with pre-LN"""

    def __init__(self, dim):
        super(ResidualLayer, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.linear(out)
        out = self.softplus(out)
        return x + out


class DecoderFFLocal(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(DecoderFFLocal, self).__init__()
        nlayers = kwargs["decoder"]["nlayers"]
        token_dim = kwargs["decoder"]["token_dim"]  # hidden dim

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["decoder"]["feats_seq"],
            dim_feats_out=token_dim,
            use_ln_out=False,
            mode="seq",
            **kwargs["decoder"],
        )

        layers = []
        for _ in range(nlayers):
            layers.append(ResidualLayer(dim=token_dim))
        self.ff_nn = torch.nn.Sequential(*layers)

        self.logit_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, rc.restype_num, bias=False),
        )
        self.struct_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, int(rc.atom_type_num * 3), bias=False),
        )

    def forward(
        self, input: Dict[str, torch.Tensor], apply_residue_type_filter: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            input: {
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        ca_coors_nm = input["ca_coors_nm"]  # [b, n, 3]
        aatype = input["residue_type"]  # [b, n] boolean
        mask = input["residue_mask"]  # [b, n] boolean

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        seqs = self.ff_nn(seqs) * mask[..., None]  # [b, n, token_dim]

        # Get logits
        logits_out = self.logit_linear(seqs) * mask[..., None]  # [b, n, 20]

        # Get coordinates
        coors_flat_nm = self.struct_linear(seqs) * mask[..., None]  # [b, n, 37 * 3]
        coors_a37_nm = einops.rearrange(
            coors_flat_nm, "b n (a t) -> b n a t", a=rc.atom_type_num, t=3
        )  # [b, n, 37, 3]

        is_dna = (aatype >= rc.dna_from_idx) & (aatype <= rc.dna_to_idx)
        is_rna = (aatype >= rc.rna_from_idx) & (aatype <= rc.rna_to_idx)
        is_na = torch.logical_or(is_dna, is_rna)  # [b, n]
        ca_idx = torch.where(
            ~is_na, rc.atom_order["CA"], rc.atom_order.get("P", rc.atom_order["CA"])
        )
        ca_mask = F.one_hot(ca_idx, num_classes=rc.atom_type_num)[..., None]  # [b, n, 37, 1]
        # coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0 + ca_coors_nm
        coors_a37_nm = coors_a37_nm * (1 - ca_mask) + ca_coors_nm[..., None, :] * ca_mask

        # Get sequence
        aatype_max = torch.argmax(logits_out, dim=-1)  # [b, n]
        if apply_residue_type_filter:
            logits_mask = torch.zeros(
                len(rc.restype_list), rc.restype_num, device=logits_out.device
            )
            logits_mask[rc.PROT - 1, rc.prot_from_idx: rc.prot_to_idx] = 1.
            if -1 < rc.dna_from_idx < rc.dna_to_idx:
                logits_mask[rc.DNA - 1, rc.dna_from_idx: rc.dna_to_idx] = 1.
            if -1 < rc.rna_from_idx < rc.rna_to_idx:
                logits_mask[rc.RNA - 1, rc.rna_from_idx: rc.rna_to_idx] = 1.
            residue_type_idx = torch.where(
                ~is_na, rc.PROT - 1, torch.where(is_dna, rc.DNA - 1, rc.RNA - 1)
            )
            logits_mask = logits_mask[residue_type_idx]
            aatype_max = torch.argmax(
                logits_out * logits_mask + torch.min(logits_out) * (1. - logits_mask),
                dim=-1,
            )  # [b,, n]
        aatype_max = aatype_max * mask  # [b, n]

        # Get atom_mask
        aa_a37_mask = get_atom_mask(device=logits_out.device)  # [21, 37] boolean
        atom_mask = aa_a37_mask[aatype_max, :]  # [b, n, 37] boolean
        atom_mask = atom_mask * mask[..., None]  # [b, n, 37] boolean

        output = {
            "coors_nm": coors_a37_nm,  # [b, n, 37, 3]
            "seq_logits": logits_out,  # [b, n, 20]
            "residue_mask": mask,  # [b, n]
            "aatype_max": aatype_max,  # [b, n]
            "atom_mask": atom_mask,  # [b, n, 37]
        }
        return output
