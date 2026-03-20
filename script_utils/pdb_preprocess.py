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


def iter_chain_orders(args, chain_group):
    if not args.do_chain_permutation:
        yield tuple(chain_group)
        return

    permutation_iter = itertools.permutations(tuple(chain_group))
    sampled_count = 0

    while True:
        batch = list(itertools.islice(permutation_iter, 100))
        if not batch:
            return

        random.shuffle(batch)
        keep_n = 10 if len(batch) == 100 else len(batch)
        for chain_order in batch[:keep_n]:
            yield chain_order
            sampled_count += 1
            if (
                args.topk_chain_permutation is not None
                and sampled_count >= args.topk_chain_permutation
            ):
                return


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
            residue_number_last + (pseudo_linker_length if i > 0 else 0)
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
        full_df = read_pdb(path=pdb_file)
        chain_groups = chain_group_dict.get(
            pdb_file.stem, [full_df["chain_id"].unique().tolist()]
        )

        for chain_group in chain_groups:
            selected_df = (
                select_chains(full_df, chain_group)
                if chain_group else full_df.copy()
            )
            if selected_df.empty:
                continue

            for chain_selection in iter_chain_orders(args, chain_group):
                # rearrange chains & residue_number
                rearranged_df = rearrange_pdb(
                    selected_df.copy(),
                    chain_selection,
                    pseudo_linker_length=args.pseudo_linker_length,
                )

                chain_selection = "".join(chain_selection)
                save_pdb(
                    args.output_dir / f"{pdb_file.stem}_{chain_selection}.pdb", rearranged_df
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
