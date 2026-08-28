#!/usr/bin/env python3
"""
Merge Protein + NA train metadata with weighted downsampling.

Use pdb_filter.py for NA dirty-ID removal.


Input source (default): <data_path>/pdb_train
Output target:          <data_path>_merge/pdb_train
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ID_COLUMN = "id"
SEQUENCE_COLUMN = "sequence"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def read_fasta(path: Path, id_column: str, seq_column: str) -> pd.DataFrame:
    records: List[Tuple[str, str]] = []
    cur_id: Optional[str] = None
    cur_seq: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id = text[1:]
                cur_seq = []
            else:
                cur_seq.append(text)

    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq)))

    return pd.DataFrame(records, columns=[id_column, seq_column])


def write_fasta(df: pd.DataFrame, out_path: Path, id_column: str, seq_column: str) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row[id_column]}\n{row[seq_column]}\n")


def read_cluster_tsv(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_idx} in {path}: expected rep<TAB>member."
                )
            rep_id, member_id = parts
            mapping.setdefault(rep_id, []).append(member_id)
    return mapping


def write_cluster_tsv(cluster_map: Dict[str, List[str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rep_id, members in cluster_map.items():
            for member_id in members:
                f.write(f"{rep_id}\t{member_id}\n")


def ensure_column(df: pd.DataFrame, col: str, name: str) -> None:
    if col not in df.columns:
        raise ValueError(f"{name} missing required column: {col!r}")


def discover_identifiers(source_pdb_train_dir: Path) -> List[str]:
    return sorted(p.stem for p in source_pdb_train_dir.glob("df_*.csv"))


def select_identifier_by_tokens(
    identifiers: List[str],
    role: str,
    fraction: str,
    min_length: int,
    max_length: int,
    moltype: str,
) -> str:
    candidates = [
        x
        for x in identifiers
        if f"_f{fraction}_" in x
        and f"_minl{min_length}_" in x
        and f"_maxl{max_length}_" in x
        and f"_mt{moltype}_".lower() in x.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"No {role} identifier found for f={fraction}, minl={min_length}, "
            f"maxl={max_length}, mt={moltype}."
        )
    raise ValueError(
        f"Ambiguous {role} identifiers for f={fraction}, minl={min_length}, "
        f"maxl={max_length}, mt={moltype}: {candidates[:20]}. "
        f"Please pass --{role}-identifier explicitly."
    )


def assert_identifier_matches_run_tokens(
    identifier: str,
    role: str,
    fraction: str,
    min_length: int,
    max_length: int,
) -> None:
    required_tokens = [f"_f{fraction}_", f"_minl{min_length}_", f"_maxl{max_length}_"]
    missing = [t for t in required_tokens if t not in identifier]
    if missing:
        raise ValueError(
            f"{role} identifier does not match run selectors. "
            f"Missing tokens {missing} in {identifier}"
        )


def assert_identifier_moltype(identifier: str, role: str, moltype: str) -> None:
    token = f"_mt{moltype}_".lower()
    if token not in identifier.lower():
        raise ValueError(
            f"{role} identifier must contain moltype token {token}: {identifier}"
        )


def build_paths_from_identifier(
    source_pdb_train_dir: Path,
    identifier: str,
    cluster_similarity: str,
) -> Tuple[Path, Path, Path]:
    df_path = source_pdb_train_dir / f"{identifier}.csv"
    seq_path = source_pdb_train_dir / f"seq_{identifier}.fasta"

    cluster_exact = (
        source_pdb_train_dir / f"cluster_seqid_{cluster_similarity}_{identifier}_test.tsv"
    )
    if cluster_exact.exists():
        return df_path, seq_path, cluster_exact

    available = sorted(
        p.name
        for p in source_pdb_train_dir.glob(f"cluster_seqid_*_{identifier}_test.tsv")
    )
    raise ValueError(
        f"Input cluster TSV not found for identifier={identifier!r} and "
        f"cluster_similarity={cluster_similarity}: {cluster_exact}. "
        f"Available files: {available[:20]}"
    )


def build_paths_from_simple_stem(
    source_pdb_train_dir: Path,
    file_stem: str,
    cluster_similarity: str,
) -> Tuple[Path, Path, Path]:
    df_path = source_pdb_train_dir / f"{file_stem}.csv"
    seq_path = source_pdb_train_dir / f"seq_{file_stem}.fasta"
    cluster_path = source_pdb_train_dir / f"cluster_seqid_{cluster_similarity}_{file_stem}_test.tsv"
    if cluster_path.exists():
        return df_path, seq_path, cluster_path

    available = sorted(
        p.name for p in source_pdb_train_dir.glob(f"cluster_seqid_*_{file_stem}_test.tsv")
    )
    raise ValueError(
        f"Input cluster TSV not found for simplified NA stem={file_stem!r} and "
        f"cluster_similarity={cluster_similarity}: {cluster_path}. "
        f"Available files: {available[:20]}"
    )


def subset_by_molecule_type(df: pd.DataFrame, allowed_types: Set[str], df_name: str) -> pd.DataFrame:
    if "molecule_type" not in df.columns:
        return df
    mt = df["molecule_type"].astype(str).str.lower().str.strip()
    out = df[mt.isin(allowed_types)].copy()
    if len(out) == 0:
        raise ValueError(
            f"{df_name}: molecule_type filtering kept 0 rows. "
            f"Allowed={sorted(allowed_types)}, observed={sorted(set(mt.unique()))[:10]}"
        )
    return out


def find_na_homopolymer_ids(
    na_seq_df: pd.DataFrame,
    id_column: str,
    seq_column: str,
) -> Set[str]:
    bad_ids: Set[str] = set()
    for _, row in na_seq_df.iterrows():
        seq = str(row[seq_column]).strip().upper()
        seq = "".join(seq.split())
        if seq and len(set(seq)) == 1:
            bad_ids.add(str(row[id_column]))
    return bad_ids


def run_pdb_filter(
    pdb_filter_script: Path,
    id_list_path: Path,
    output_dir: Path,
    df_pdb_csv: Path,
    seq_df_pdb_fasta: Path,
    cluster_seqid_tsv: Path,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(pdb_filter_script),
        "-i",
        str(id_list_path),
        "-o",
        str(output_dir),
        str(df_pdb_csv),
        str(seq_df_pdb_fasta),
        str(cluster_seqid_tsv),
    ]
    subprocess.run(cmd, check=True)
    return (
        output_dir / df_pdb_csv.name,
        output_dir / seq_df_pdb_fasta.name,
        output_dir / cluster_seqid_tsv.name,
    )


def filter_cluster_by_ids(
    cluster_map: Dict[str, List[str]],
    valid_ids: Set[str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for rep_id, members in cluster_map.items():
        if rep_id not in valid_ids:
            continue
        kept = [m for m in members if m in valid_ids]
        if kept:
            out[rep_id] = kept
    return out


def compute_downsample_targets(
    n_protein_clusters: int,
    n_na_clusters: int,
    protein_weight: float,
    na_weight: float,
) -> Tuple[int, int]:
    if protein_weight < 0 or na_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if protein_weight == 0 and na_weight == 0:
        raise ValueError("At least one weight must be > 0.")
    if n_protein_clusters == 0 and n_na_clusters == 0:
        raise ValueError("No clusters available after filtering.")

    if protein_weight == 0:
        return 0, n_na_clusters
    if na_weight == 0:
        return n_protein_clusters, 0

    scale = min(n_protein_clusters / protein_weight, n_na_clusters / na_weight)
    protein_target = int(math.floor(scale * protein_weight))
    na_target = int(math.floor(scale * na_weight))
    protein_target = max(1, protein_target) if n_protein_clusters > 0 else 0
    na_target = max(1, na_target) if n_na_clusters > 0 else 0
    return min(protein_target, n_protein_clusters), min(na_target, n_na_clusters)


def sample_without_replacement(keys: List[str], target_n: int, rng: random.Random) -> List[str]:
    if target_n <= 0:
        return []
    if target_n > len(keys):
        raise ValueError(
            f"target_n={target_n} > available={len(keys)}; downsampling mode forbids oversampling."
        )
    return rng.sample(keys, target_n)


def ensure_symlink(link_path: Path, source_path: Path) -> None:
    source = source_path.resolve()
    if not source.exists():
        raise ValueError(f"Symlink source does not exist: {source_path}")
    if link_path.is_symlink():
        if link_path.resolve() == source:
            return
        link_path.unlink()
        link_path.symlink_to(source, target_is_directory=source.is_dir())
        return
    if link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
        link_path.symlink_to(source, target_is_directory=source.is_dir())
        return
    link_path.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Protein/NA df+seq_df+cluster.tsv with weighted downsampling and "
            "write train-ready files into <DATA_PATH>_merge/pdb_train."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Dataset root, e.g. datasetcard/pdb_768")
    parser.add_argument(
        "--source-pdb-train-dir",
        type=Path,
        default=None,
        help="Optional source override. Default: <data-path>/pdb_train",
    )
    parser.add_argument(
        "--na-source-pdb-train-dir",
        type=Path,
        default=None,
        help=(
            "Optional NA source override. If set, NA is read from this directory "
            "instead of --source-pdb-train-dir."
        ),
    )
    parser.add_argument(
        "--na-simple-file-stem",
        type=str,
        default="pdb_train",
        help=(
            "File stem for simplified NA naming in --na-source-pdb-train-dir: "
            "<stem>.csv, seq_<stem>.fasta, cluster_seqid_<sim>_<stem>_test.tsv."
        ),
    )
    parser.add_argument("--fraction", type=str, required=True, help="Token f<value>, e.g. 0.85")
    parser.add_argument("--min-length", type=int, required=True, help="Token minl<value>")
    parser.add_argument("--max-length", type=int, required=True, help="Token maxl<value>")
    parser.add_argument("--protein-identifier", type=str, default=None, help="Optional explicit df_... identifier")
    parser.add_argument("--na-identifier", type=str, default=None, help="Optional explicit df_... identifier")
    parser.add_argument(
        "--train-file-identifier",
        type=str,
        default=None,
        help=(
            "Output identifier. Default: use na-identifier (mtNone-style), "
            "which matches dataselector.molecule_type=None naming."
        ),
    )
    parser.add_argument(
        "--protein_cluster_similarity",
        type=str,
        default="0.5",
        help="Protein input cluster similarity token.",
    )
    parser.add_argument(
        "--na_cluster_similarity",
        type=str,
        default="0.5",
        help="NA input cluster similarity token.",
    )
    parser.add_argument(
        "--output_cluster_similarity",
        type=str,
        default=None,
        help=(
            "Similarity token used only in output cluster filename. "
            "Default: na_cluster_similarity."
        ),
    )
    parser.add_argument("--protein-weight", type=float, default=1.0)
    parser.add_argument("--na-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--removed-na-id-filename", type=str, default="removed_na_dirty_ids.txt")
    parser.add_argument("--pdb-filter-script", type=Path, default=Path("script_utils/pdb_filter.py"))
    parser.add_argument("--enable-symlink", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_pdb_train = args.source_pdb_train_dir or (args.data_path / "pdb_train")
    na_source_pdb_train = args.na_source_pdb_train_dir or source_pdb_train
    out_data_path = args.data_path.parent / f"{args.data_path.name}_merge"
    out_pdb_train = out_data_path / "pdb_train"
    out_pdb_train.mkdir(parents=True, exist_ok=True)

    if not source_pdb_train.exists():
        raise ValueError(f"Source pdb_train dir not found: {source_pdb_train}")
    if not na_source_pdb_train.exists():
        raise ValueError(f"NA source pdb_train dir not found: {na_source_pdb_train}")

    identifiers = discover_identifiers(source_pdb_train)
    if not identifiers:
        raise ValueError(f"No df_*.csv found under {source_pdb_train}")

    protein_identifier = args.protein_identifier or select_identifier_by_tokens(
        identifiers=identifiers,
        role="protein",
        fraction=args.fraction,
        min_length=args.min_length,
        max_length=args.max_length,
        moltype="protein",
    )
    assert_identifier_matches_run_tokens(
        protein_identifier, "protein", args.fraction, args.min_length, args.max_length
    )
    assert_identifier_moltype(protein_identifier, "protein", "protein")

    # Input similarities are controlled independently for protein and NA.
    protein_similarity = args.protein_cluster_similarity
    na_similarity = args.na_cluster_similarity
    # Output filename token defaults to NA similarity for mtNone train compatibility.
    out_similarity = args.output_cluster_similarity or na_similarity

    protein_df_path, protein_seq_path, protein_cluster_path = build_paths_from_identifier(
        source_pdb_train,
        protein_identifier,
        protein_similarity,
    )

    if args.na_source_pdb_train_dir is None:
        na_identifier = args.na_identifier or select_identifier_by_tokens(
            identifiers=identifiers,
            role="na",
            fraction=args.fraction,
            min_length=args.min_length,
            max_length=args.max_length,
            moltype="None",
        )
        assert_identifier_matches_run_tokens(
            na_identifier, "na", args.fraction, args.min_length, args.max_length
        )
        assert_identifier_moltype(na_identifier, "na", "None")
        na_df_path, na_seq_path, na_cluster_path = build_paths_from_identifier(
            source_pdb_train,
            na_identifier,
            na_similarity,
        )
    else:
        if args.na_identifier:
            # Advanced override: keep old identifier-style loading in custom NA source.
            na_identifier = args.na_identifier
            na_df_path, na_seq_path, na_cluster_path = build_paths_from_identifier(
                na_source_pdb_train,
                na_identifier,
                na_similarity,
            )
        else:
            na_identifier = args.na_simple_file_stem
            na_df_path, na_seq_path, na_cluster_path = build_paths_from_simple_stem(
                na_source_pdb_train,
                na_identifier,
                na_similarity,
            )

    for p in [
        protein_df_path,
        protein_seq_path,
        protein_cluster_path,
        na_df_path,
        na_seq_path,
        na_cluster_path,
    ]:
        if not p.exists():
            raise ValueError(f"Input file not found: {p}")

    protein_df = read_table(protein_df_path)
    protein_seq_df = read_fasta(protein_seq_path, ID_COLUMN, SEQUENCE_COLUMN)
    protein_cluster = read_cluster_tsv(protein_cluster_path)

    na_df = read_table(na_df_path)
    na_seq_df = read_fasta(na_seq_path, ID_COLUMN, SEQUENCE_COLUMN)
    na_cluster = read_cluster_tsv(na_cluster_path)

    ensure_column(protein_df, ID_COLUMN, "protein_df")
    ensure_column(protein_seq_df, ID_COLUMN, "protein_seq_df")
    ensure_column(protein_seq_df, SEQUENCE_COLUMN, "protein_seq_df")
    ensure_column(na_df, ID_COLUMN, "na_df")
    ensure_column(na_seq_df, ID_COLUMN, "na_seq_df")
    ensure_column(na_seq_df, SEQUENCE_COLUMN, "na_seq_df")

    # mtNone source contains mixed molecule types, so we split by molecule_type.
    protein_df = subset_by_molecule_type(protein_df, {"protein"}, "protein_df")
    na_df = subset_by_molecule_type(na_df, {"na", "dna", "rna"}, "na_df")

    protein_df[ID_COLUMN] = protein_df[ID_COLUMN].astype(str)
    protein_seq_df[ID_COLUMN] = protein_seq_df[ID_COLUMN].astype(str)
    na_df[ID_COLUMN] = na_df[ID_COLUMN].astype(str)
    na_seq_df[ID_COLUMN] = na_seq_df[ID_COLUMN].astype(str)

    removed_na_ids = find_na_homopolymer_ids(na_seq_df, ID_COLUMN, SEQUENCE_COLUMN)
    if removed_na_ids:
        if not args.pdb_filter_script.exists():
            raise ValueError(f"pdb_filter.py not found: {args.pdb_filter_script}")

        tmp_dir = out_pdb_train / "_tmp_na_filter"
        ids_file = tmp_dir / "na_dirty_ids.txt"
        filtered_dir = tmp_dir / "filtered_na"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with ids_file.open("w", encoding="utf-8") as f:
            for pid in sorted(removed_na_ids):
                f.write(f"{pid}\n")

        f_df, f_seq, f_cluster = run_pdb_filter(
            pdb_filter_script=args.pdb_filter_script,
            id_list_path=ids_file,
            output_dir=filtered_dir,
            df_pdb_csv=na_df_path,
            seq_df_pdb_fasta=na_seq_path,
            cluster_seqid_tsv=na_cluster_path,
        )
        na_df = read_table(f_df)
        na_seq_df = read_fasta(f_seq, ID_COLUMN, SEQUENCE_COLUMN)
        na_cluster = read_cluster_tsv(f_cluster)

        na_df = subset_by_molecule_type(na_df, {"na", "dna", "rna"}, "na_df")
        na_df[ID_COLUMN] = na_df[ID_COLUMN].astype(str)
        na_seq_df[ID_COLUMN] = na_seq_df[ID_COLUMN].astype(str)

    protein_valid_ids = set(protein_df[ID_COLUMN]) & set(protein_seq_df[ID_COLUMN])
    na_valid_ids = set(na_df[ID_COLUMN]) & set(na_seq_df[ID_COLUMN])
    if not protein_valid_ids:
        raise ValueError("No valid Protein IDs left.")
    if not na_valid_ids:
        raise ValueError("No valid NA IDs left.")

    overlap = protein_valid_ids & na_valid_ids
    if overlap:
        raise ValueError(f"Protein/NA ID overlap detected, e.g. {sorted(overlap)[:10]}")

    protein_cluster = filter_cluster_by_ids(protein_cluster, protein_valid_ids)
    na_cluster = filter_cluster_by_ids(na_cluster, na_valid_ids)
    if not protein_cluster and args.protein_weight > 0:
        raise ValueError("Protein cluster map empty after filtering.")
    if not na_cluster and args.na_weight > 0:
        raise ValueError("NA cluster map empty after filtering.")

    protein_target, na_target = compute_downsample_targets(
        len(protein_cluster), len(na_cluster), args.protein_weight, args.na_weight
    )
    protein_rep_ids = sample_without_replacement(list(protein_cluster.keys()), protein_target, rng)
    na_rep_ids = sample_without_replacement(list(na_cluster.keys()), na_target, rng)

    merged_cluster: Dict[str, List[str]] = {rid: protein_cluster[rid] for rid in protein_rep_ids}
    for rid in na_rep_ids:
        if rid in merged_cluster:
            raise ValueError(f"Representative collision: {rid}")
        merged_cluster[rid] = na_cluster[rid]

    selected_ids: Set[str] = set()
    for members in merged_cluster.values():
        selected_ids.update(members)

    merged_df = pd.concat([protein_df, na_df], ignore_index=True)
    merged_seq_df = pd.concat([protein_seq_df, na_seq_df], ignore_index=True)
    merged_df = merged_df[merged_df[ID_COLUMN].isin(selected_ids)].reset_index(drop=True)
    merged_seq_df = merged_seq_df[merged_seq_df[ID_COLUMN].isin(selected_ids)].reset_index(drop=True)

    if args.train_file_identifier:
        ident = args.train_file_identifier
    elif na_identifier.startswith("df_"):
        ident = na_identifier
    else:
        # Keep output naming compatible with the repository's df_..._mtNone_... pattern.
        ident = protein_identifier.replace("_mtprotein_", "_mtNone_")
    removed_paths: List[Path] = []
    # Always clean historical generated outputs so naming stays consistent run-to-run.
    for pattern in [
        "df_*.csv",
        "seq_df_*.fasta",
        "cluster_seqid_*_test.tsv",
        "cluster_seqid_*_test.fasta",
    ]:
        for path in out_pdb_train.glob(pattern):
            if path.exists() or path.is_symlink():
                path.unlink()
                removed_paths.append(path)
    # Backward compatibility for custom output identifiers from older runs.
    for pattern in ["*.csv", "seq_*.fasta"]:
        for path in out_pdb_train.glob(pattern):
            if path.exists() and path not in removed_paths:
                if path.name in {"removed_na_dirty_ids.txt"}:
                    continue
                path.unlink()
                removed_paths.append(path)
    removed_ids_path = out_pdb_train / args.removed_na_id_filename
    if removed_ids_path.exists() or removed_ids_path.is_symlink():
        removed_ids_path.unlink()
        removed_paths.append(removed_ids_path)
    tmp_filter_dir = out_pdb_train / "_tmp_na_filter"
    if tmp_filter_dir.exists():
        shutil.rmtree(tmp_filter_dir)
        removed_paths.append(tmp_filter_dir)

    sim = out_similarity
    out_df = out_pdb_train / f"{ident}.csv"
    out_seq = out_pdb_train / f"seq_{ident}.fasta"
    out_cluster_tsv = out_pdb_train / f"cluster_seqid_{sim}_{ident}_test.tsv"
    out_cluster_fasta = out_pdb_train / f"cluster_seqid_{sim}_{ident}_test.fasta"
    out_removed_ids = out_pdb_train / args.removed_na_id_filename

    merged_df.to_csv(out_df, index=False)
    write_fasta(merged_seq_df, out_seq, ID_COLUMN, SEQUENCE_COLUMN)
    write_cluster_tsv(merged_cluster, out_cluster_tsv)

    rep_seq_df = merged_seq_df[
        merged_seq_df[ID_COLUMN].isin(list(merged_cluster.keys()))
    ].drop_duplicates(subset=[ID_COLUMN], keep="first")
    rep_ids = set(rep_seq_df[ID_COLUMN].astype(str))
    expected_rep_ids = set(merged_cluster.keys())
    if rep_ids != expected_rep_ids:
        missing = expected_rep_ids - rep_ids
        raise ValueError(
            f"Missing representative sequences for: {sorted(list(missing))[:10]}"
        )
    write_fasta(rep_seq_df, out_cluster_fasta, ID_COLUMN, SEQUENCE_COLUMN)

    with out_removed_ids.open("w", encoding="utf-8") as f:
        for pid in sorted(removed_na_ids):
            f.write(f"{pid}\n")

    if args.enable_symlink:
        ensure_symlink(out_pdb_train / "processed", source_pdb_train / "processed")
        ensure_symlink(out_pdb_train / "raw", source_pdb_train / "raw")

    print("Done.")
    print(f"  source pdb_train: {source_pdb_train}")
    print(f"  na source pdb_train: {na_source_pdb_train}")
    print(f"  output pdb_train: {out_pdb_train}")
    print(f"  protein identifier: {protein_identifier}")
    print(f"  na identifier: {na_identifier}")
    print(f"  output identifier: {ident}")
    print(f"  protein cluster similarity: {protein_similarity}")
    print(f"  na cluster similarity: {na_similarity}")
    print(f"  output cluster similarity token: {out_similarity}")
    print(f"  input clusters: protein={len(protein_cluster)}, na={len(na_cluster)}")
    print(f"  target clusters: protein={protein_target}, na={na_target}")
    print(f"  removed NA dirty IDs: {len(removed_na_ids)}")
    print(f"  overwrite output: True (removed {len(removed_paths)} existing artifacts)")


if __name__ == "__main__":
    main()
