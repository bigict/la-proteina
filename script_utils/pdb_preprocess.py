from collections import defaultdict
import itertools
import math
import pathlib
import random
from typing import List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from graphein.protein.graphs import (
    read_pdb_to_dataframe, select_chains, sort_dataframe
)
from graphein.protein.tensor.sequence import get_sequence
from graphein.protein.utils import save_pdb_df_to_pdb


def yield_chain_permutations(chain_group: List[str], topk:Optional[int] = None):
    if topk is None or topk >= math.factorial(len(chain_group)):
        yield from itertools.permutations(chain_group)
    else:
        i = 0
        chain_permutations_seen = set()
        while i < topk:
            random.shuffle(chain_group)
            if tuple(chain_group) not in chain_permutations_seen:
                yield chain_group
                chain_permutations_seen.add(tuple(chain_group))
                i += 1


def read_pdb(
    path: Optional[Union[str, pathlib.Path]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
) -> pd.DataFrame:
    df = read_pdb_to_dataframe(
        path=path,
        pdb_code=pdb_code,
        uniprot_id=uniprot_id,
        model_index=model_index
    )
    df = df.loc[df["record_name"] == "ATOM"]
    df = sort_dataframe(df)

    return df


def save_pdb(path: pathlib.Path, df: pd.DataFrame) -> None:
    save_pdb_df_to_pdb(df, path)


def rearrange_pdb(
    df: pd.DataFrame, chain_list: List[str], pseudo_linker_length: int = 128
) -> pd.DataFrame:
    chains = sorted(
        df.groupby(by=["chain_id"]), key=lambda g: chain_list.index(g[0][0])
    )

    residue_number_last = 0
    for i, (_, g) in enumerate(chains):
        residue_number_min = g["residue_number"].min()
        residue_number_offset = (
            residue_number_last + i * pseudo_linker_length
        )
        g["residue_number"] = g["residue_number"].apply(
            lambda x: x - residue_number_min + 1 + residue_number_offset
        )
        residue_number_last = g["residue_number"].max()
        chains[i] = g

    return pd.concat(chains).reset_index(drop=True)


def main(args):
    random.seed()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chain_group_dict = defaultdict(list)
    if args.chain_idx:
        with open(args.chain_idx, "r") as f:
            for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
                pid, *chains = line.split()
                chain_group_dict[pid].append(chains)
    print(f"# items in chain_idx: {len(chain_group_dict)}")

    for pdb_file in tqdm(args.pdb_file):
        df_pdb = read_pdb(path=pdb_file)
        chain_groups = chain_group_dict.get(
            pdb_file.stem, [df_pdb["chain_id"].unique().tolist()]
        )

        for chain_group in chain_groups:
            df = (
                select_chains(df_pdb, chain_group) if chain_group else df_pdb.copy()
            )
            if df.empty:
                continue

            if args.sequence_max_length is not None:
                if len(get_sequence(df, list_of_three=True)) > args.sequence_max_length:
                    continue

            for chain_selection in (
                yield_chain_permutations(chain_group, args.topk_chain_permutation)
                if args.do_chain_permutation else [chain_group]
            ):
                # rearrange chains & residue_number
                df = rearrange_pdb(
                    df, chain_selection, pseudo_linker_length=args.pseudo_linker_length
                )

                chain_selection = "-".join(chain_selection)
                save_pdb(
                    args.output_dir / f"{args.pdb_prefix}{pdb_file.stem}_{chain_selection}.pdb", df
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pdb_file", type=pathlib.Path, nargs="+", help="list of pdb files"
    )
    parser.add_argument(
        "-o", "--output_dir", type=pathlib.Path, default=".", help="output dir"
    )
    parser.add_argument(
        "--pdb_prefix", type=str, default="", help="add a prefix to each output pdb file"
    )
    parser.add_argument("--chain_idx", type=str, default=None, help="chain idx file")
    parser.add_argument(
        "--do_chain_permutation", action="store_true", help="do chain permutation"
    )
    parser.add_argument(
        "--topk_chain_permutation", type=int, default=None, help="topk chain permutation"
    )
    parser.add_argument(
        "--sequence_max_length", type=int, default=None, help="maximum sequence length"
    )
    parser.add_argument(
        "--pseudo_linker_length",
        type=int,
        default=128,
        help="add a pseudo linker between chains"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    main(args)
