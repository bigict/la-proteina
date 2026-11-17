"""Batch MolProbity validation of generated structures.

Requires a Phenix/cctbx environment (``libtbx``, ``iotbx``, ``mmtbx``); it cannot
run in the plain training environment.

The primary outputs for nucleic-acid geometry are ``bond_rmsd`` and
``angle_rmsd``: deviations from standard restraint ideals, purely local, with no
contribution from rigid-body placement error. Training-side metrics such as
all-atom RMSD cannot resolve bond-length precision, so these are the reference
measurement.

Both are whole-model aggregates. When a file contains protein and nucleic acid
together, the well-refined protein bonds dilute the nucleic-acid signal, so
validate one molecule type per file (e.g. a ``na-*.pdb`` glob) or split the
structure first.

Usage:
    python script_utils/molprobity_eval.py \
        "inference/inference_ae/<run>/na-v1-dna-*.pdb" results.csv

The first argument is a glob pattern (quote it so the shell does not expand it).
A bare directory is also accepted and expands to ``<dir>/**/*.pdb``.
"""

import os
import sys
import glob

import pandas as pd
import libtbx.load_env

# A locally installed probe/reduce is not registered in this libtbx build, so
# has_module() reports them missing and MolProbity would skip clash detection.
# Assert their presence for those two names only; everything else keeps the
# original behaviour.
_orig_has_module = libtbx.env.has_module
libtbx.env.has_module = lambda name: True if name in ["probe", "reduce"] else _orig_has_module(name)

import iotbx.pdb
import mmtbx.model
import mmtbx.validation.molprobity

# Geometry first: it is the measurement this script exists for, and it must
# survive a failure in the optional protein-only or probe-dependent metrics.
_METRIC_GETTERS = (
    ("bond_rmsd", "rms_bonds"),
    ("angle_rmsd", "rms_angles"),
    ("clashscore", "clashscore"),
    ("molprobity_score", "molprobity_score"),
    ("ramachandran_favored", "rama_favored"),
    ("ramachandran_outliers", "rama_outliers"),
    ("rotamer_outliers", "rota_outliers"),
)
# Reported as fractions by the API, stored as percentages.
_PERCENT_METRICS = frozenset(
    ("ramachandran_favored", "ramachandran_outliers", "rotamer_outliers")
)
# Undefined without protein. The API returns 0.0 rather than None for a
# nucleic-acid-only model, which would otherwise read as a genuine score of
# zero and drag the dataset mean down.
_PROTEIN_ONLY_METRICS = frozenset(
    ("ramachandran_favored", "ramachandran_outliers", "rotamer_outliers")
)
_METRICS = tuple(name for name, _ in _METRIC_GETTERS)
_ROUNDING = {
    "clashscore": 2,
    "molprobity_score": 2,
    "ramachandran_favored": 2,
    "ramachandran_outliers": 2,
    "rotamer_outliers": 2,
    "bond_rmsd": 4,
    "angle_rmsd": 2,
}


def _extract_metrics(mp_engine):
    """Read every metric independently so one failure cannot discard the rest.

    Returns (values, errors). A metric that raises is stored as None and its
    exception recorded, instead of aborting the whole structure. This matters
    because clashscore depends on probe/reduce and the Ramachandran/rotamer
    metrics are undefined for a nucleic-acid-only model.
    """
    values, errors = {}, {}
    for name, method in _METRIC_GETTERS:
        try:
            value = getattr(mp_engine, method)()
        except Exception as exc:
            values[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if value is not None and name in _PERCENT_METRICS:
            value = value * 100
        values[name] = value
    return values, errors


def _composition(model):
    """Count protein and nucleic-acid residues, or (None, None) if unavailable.

    Used only to decide whether the protein-only metrics apply. Any failure
    falls back to reporting the raw API values, so a residue-classification
    problem cannot break the geometry measurement this script exists for.
    """
    try:
        groups = model.get_hierarchy().residue_groups()
        n_prot = sum(1 for rg in groups if rg.is_protein())
        n_na = sum(1 for rg in groups if rg.is_nucleic_acid())
        return n_prot, n_na
    except Exception:
        return None, None


def _resolve_pattern(pattern):
    """Accept a glob pattern, or a directory to search recursively for .pdb."""
    if os.path.isdir(pattern):
        return os.path.join(pattern, "**", "*.pdb")
    return pattern


def process_pdb_directory(dir_path, output_csv):
    pattern = _resolve_pattern(dir_path)
    pdb_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Pattern: {pattern}")
    print(f"Found {len(pdb_files)} PDB structures to evaluate.")
    if not pdb_files:
        print("Nothing matched. Pass a quoted glob such as \"dir/na-*.pdb\".")
        return

    records = []
    for idx, pdb_path in enumerate(pdb_files):
        basename = os.path.basename(pdb_path)
        record = {"pdb_name": basename}
        try:
            pdb_inp = iotbx.pdb.input(file_name=pdb_path)
            processed_model = mmtbx.model.manager(model_input=pdb_inp)
            mp_engine = mmtbx.validation.molprobity.molprobity(model=processed_model)
        except Exception as exc:
            record.update({m: None for m in _METRICS})
            record["status"] = "model_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            print(f"[{idx+1}/{len(pdb_files)}] FAILED to build model: {basename} | {record['error']}")
            continue

        values, errors = _extract_metrics(mp_engine)
        n_prot, n_na = _composition(processed_model)
        record["n_protein_residues"] = n_prot
        record["n_na_residues"] = n_na
        # Report N/A rather than a misleading 0.0 when there is no protein.
        if n_prot == 0:
            for name in _PROTEIN_ONLY_METRICS:
                values[name] = None
        record.update(values)
        # Geometry is the point of the run: flag its loss explicitly.
        geometry_missing = [m for m in ("bond_rmsd", "angle_rmsd") if values[m] is None]
        if geometry_missing:
            record["status"] = "geometry_missing"
            record["error"] = "; ".join(f"{m}: {errors[m]}" for m in geometry_missing)
            print(f"[{idx+1}/{len(pdb_files)}] GEOMETRY MISSING: {basename} | {record['error']}")
        else:
            record["status"] = "ok" if not errors else "partial"
            record["error"] = "; ".join(f"{m}: {e}" for m, e in errors.items())
            note = f" (optional metrics unavailable: {', '.join(errors)})" if errors else ""
            print(
                f"[{idx+1}/{len(pdb_files)}] {basename}: "
                f"bond_rmsd={values['bond_rmsd']:.4f} angle_rmsd={values['angle_rmsd']:.2f}{note}"
            )
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[+] Saved per-structure results to: {output_csv}")

    n_total = len(df)
    n_geom = int(df["bond_rmsd"].notna().sum())
    print(f"[+] Geometry obtained for {n_geom}/{n_total} structures.")
    if "status" in df:
        counts = df["status"].value_counts().to_dict()
        print(f"[+] Status breakdown: {counts}")
    if n_geom == 0:
        print("\nNo bond/angle values were obtained; see the error column in the CSV.")
        return

    print("\n" + "=" * 50)
    print("      DATASET QUALITY SUMMARY (PANDAS)")
    print("=" * 50)
    summary_df = df[list(_METRICS)].describe().loc[["mean", "50%", "min", "max"]]
    summary_df.index = ["Mean", "Median (50%)", "Minimum", "Maximum"]
    print(summary_df.round(_ROUNDING).to_string())
    # Protein-only metrics are undefined on a nucleic-acid-only model; show the
    # sample size so an empty column is not mistaken for a zero score.
    print("\nNon-missing count per metric:")
    print(df[list(_METRICS)].notna().sum().to_string())
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        process_pdb_directory(sys.argv[1], sys.argv[2])
    else:
        print(
            "Usage: python molprobity_eval.py '<glob-or-directory>' <output_csv>\n"
            '  e.g. python molprobity_eval.py "inference/inference_ae/<run>/na-v1-dna-*.pdb" results.csv'
        )
