#!/usr/bin/env python
"""Export DNA sequence-recovery probabilities from one protein-DNA complex."""

import argparse
import csv
from pathlib import Path

import torch

from openfold.np import residue_constants as rc
from openfold.np.residue_constants import resname_to_idx
from openfold.utils.feats import atom_gather
from proteinfoundation.datasets.pdb_data import protein_to_pyg
from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    CoordsToNanometers,
)
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.dense_padding_data_loader import (
    dense_padded_from_data_list,
)


BASES = "ACGT"
FILL_COORD = 1e-5


def load_complex(pdb_path: str):
    graph = protein_to_pyg(
        path=pdb_path,
        chain_selection="all",
        fill_value_coords=FILL_COORD,
    )
    graph.coord_mask = (graph.coords != FILL_COORD)[..., 0]
    graph.residue_type = torch.tensor(
        [resname_to_idx[residue] for residue in graph.residues], dtype=torch.long
    )
    graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

    chain_ids = [str(residue_id).split(":", 1)[0] for residue_id in graph.residue_id]
    is_dna = (
        (graph.residue_type >= rc.dna_from_idx)
        & (graph.residue_type <= rc.dna_to_idx)
    )
    dna_chains = sorted({chain for chain, dna in zip(chain_ids, is_dna) if dna})
    if len(dna_chains) != 2:
        raise ValueError(
            f"Expected exactly two DNA chains, found {dna_chains}"
        )
    reference_chain = dna_chains[0]
    graph.selected_dna = torch.tensor(
        [chain == reference_chain for chain in chain_ids]
    ) & is_dna

    for transform in (
        CoordsToNanometers(),
        CenterStructureTransform(),
        ChainBreakPerResidueTransform(),
    ):
        graph = transform(graph)
    return dense_padded_from_data_list([graph]), reference_chain


@torch.inference_mode()
def recover_pwm(model: AutoEncoder, batch, device: torch.device):
    model = model.eval().to(device)
    batch = batch.to(device)
    residue_mask = batch.mask_dict["coords"][..., 0, 0]
    batch.mask = residue_mask

    encoded = model.encoder(batch)
    aatype = batch.residue_type
    is_na = (
        ((aatype >= rc.dna_from_idx) & (aatype <= rc.dna_to_idx))
        | ((aatype >= rc.rna_from_idx) & (aatype <= rc.rna_to_idx))
    )
    anchor_index = torch.where(
        is_na,
        rc.atom_order["P"],
        rc.atom_order["CA"],
    )
    anchors_nm = atom_gather(batch.coords_nm, anchor_index, -2) * residue_mask[..., None]

    decoded = model.decoder(
        {
            "z_latent": encoded["mean"],
            "ca_coors_nm": anchors_nm,
            "residue_type": aatype,
            "residue_mask": residue_mask,
            "mask": residue_mask,
        }
    )
    dna_logits = decoded["seq_logits"][0, batch.selected_dna[0].bool()]
    return torch.softmax(
        dna_logits[:, rc.dna_from_idx : rc.dna_to_idx], dim=-1
    ).cpu()


def save_pwm(pwm: torch.Tensor, output_path: Path) -> None:
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", *BASES, "predicted_base"])
        for position, probabilities in enumerate(pwm.tolist(), start=1):
            writer.writerow(
                [
                    position,
                    *probabilities,
                    BASES[max(range(4), key=probabilities.__getitem__)],
                ]
            )


def save_both_orientations(pwm: torch.Tensor, output_prefix: str) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    forward_path = prefix.parent / f"{prefix.name}_forward.csv"
    reverse_path = prefix.parent / f"{prefix.name}_reverse_complement.csv"
    save_pwm(pwm, forward_path)
    save_pwm(pwm.flip((0, 1)), reverse_path)
    return forward_path, reverse_path


def predict_one(model, pdb_path: Path, output_prefix: Path, device: torch.device):
    batch, reference_chain = load_complex(str(pdb_path))
    pwm = recover_pwm(model, batch, device)
    forward_path, reverse_path = save_both_orientations(pwm, str(output_prefix))
    print(
        f"{pdb_path.name}: chain={reference_chain}, positions={len(pwm)}, "
        f"outputs={forward_path.name},{reverse_path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Complex PDB file or directory")
    parser.add_argument("--checkpoint", required=True, help="Protein-DNA AE checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    pdb_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdb"))
    if not pdb_files:
        parser.error(f"no PDB files found at {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoEncoder.load_from_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    for pdb_path in pdb_files:
        predict_one(model, pdb_path, output_dir / pdb_path.stem, device)
    print(f"Predicted {len(pdb_files)} complexes into {output_dir}")


if __name__ == "__main__":
    main()
