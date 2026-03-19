from collections import defaultdict
import itertools
import pathlib
import random
from typing import List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from graphein.protein.graphs import (
    read_pdb_to_dataframe, select_chains, sort_dataframe
)
from graphein.protein.utils import save_pdb_df_to_pdb


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
        for i, chain_selection in enumerate(
            chain_group_dict.get(pdb_file.stem, [df_pdb["chain_id"].unique().tolist()])
        ):
            df = select_chains(df_pdb, chain_selection)

            # do chain permutations
            chain_selection_list = list(
                itertools.permutations(chain_selection)
                if args.do_chain_permutation else [chain_selection]
            )

            random.shuffle(chain_selection_list)

            for chain_selection in chain_selection_list[:args.topk_chain_permutation]:
                # rearrange chains & residue_number
                df = rearrange_pdb(
                    df, chain_selection, pseudo_linker_length=args.pseudo_linker_length
                )

                chain_selection = "".join(chain_selection)
                save_pdb(
                    args.output_dir / f"{pdb_file.stem}_{chain_selection}.pdb", df
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
    parser.add_argument("--chain_idx", type=str, default=None, help="chain idx file")
    parser.add_argument(
        "--do_chain_permutation", action="store_true", help="do chain permutation"
    )
    parser.add_argument(
        "--topk_chain_permutation", type=int, default=None, help="topk chain permutation"
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
