#!/usr/bin/env python
"""Evaluate AE soft PWM predictions from protein-DNA complex structures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import eval_pwm_from_pdb as pdb_eval
from predict_pwm_from_complex import load_complex, recover_pwm
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.pwm_utils import ic_weighted_pcc


def align_pwms(reference: np.ndarray, predicted: np.ndarray):
    """Globally align the shorter PWM and choose forward or reverse complement."""
    best = None
    for orientation, candidate in (
        ("forward", predicted),
        ("reverse_complement", predicted[::-1, ::-1]),
    ):
        overlap = min(len(reference), len(candidate))
        for ref_start in range(len(reference) - overlap + 1):
            ref = reference[ref_start : ref_start + overlap]
            for pred_start in range(len(candidate) - overlap + 1):
                pred = candidate[pred_start : pred_start + overlap]
                score = sum(
                    np.nan_to_num(ic_weighted_pcc(p, r, gt=r)[0], nan=0.0)
                    for r, p in zip(ref, pred)
                )
                if best is None or score > best[0]:
                    best = score, orientation, ref_start, pred_start, ref, pred
    return best


def evaluate(model, pdb_path, pwm_id, device):
    batch, reference_chain = load_complex(str(pdb_path))
    predicted = recover_pwm(model, batch, device).numpy()
    reference = pdb_eval.load_pwm(pwm_id)
    score, orientation, ref_start, pred_start, ref, pred = align_pwms(
        reference, predicted
    )
    metrics = {
        key: float(value) for key, value in pdb_eval.compute_all_metrics(ref, pred).items()
    }
    return {
        "reference_chain": reference_chain,
        "orientation": orientation,
        "alignment_score": float(score),
        "reference_start": ref_start,
        "prediction_start": pred_start,
        "overlap_length": len(ref),
        "reference_length": len(reference),
        "prediction_length": len(predicted),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="CSV lines: protein_id,pwm_id,complex_pdb")
    parser.add_argument("--checkpoint", required=True, help="Protein-DNA AE checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pdb-root", default=".")
    parser.add_argument(
        "--pwm-data",
        default=str(Path(__file__).with_name("pwms.pickle")),
        help="JASPAR/H11MO PWM pickle",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_eval.DATA_PATH = args.pwm_data
    pdb_eval._pwm_cache.clear()

    device = torch.device(args.device)
    model = AutoEncoder.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    ).eval().to(device)

    results = []
    with open(args.input_file, newline="") as handle:
        for row in csv.reader(line for line in handle if not line.lstrip().startswith("#")):
            if not row or len(row) != 3:
                continue
            protein_id, pwm_id, pdb_file = (value.strip() for value in row)
            result = evaluate(
                model,
                Path(args.pdb_root) / pdb_file,
                pwm_id,
                device,
            )
            results.append({"protein_id": protein_id, "pwm_id": pwm_id, **result})

    if not results:
        raise ValueError("No valid evaluation rows found")

    fieldnames = list(results[0])
    with open(output_dir / "eval_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metric_names = ["mae", "ic_corr", "brier_multi", "ic_diff"]
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in results])),
            "std": float(np.std([row[metric] for row in results])),
        }
        for metric in metric_names
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Evaluated {len(results)} complexes; results saved to {output_dir}")


if __name__ == "__main__":
    main()
