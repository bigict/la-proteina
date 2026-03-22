import csv
import pathlib
from typing import List

from Bio import SeqIO


def dataset_file_split(path: pathlib.Path) -> List[pathlib.Path]:
    name = path.name

    k = name.find(',')
    if k != -1:
        name, seq_id = name[:k], name[k + 1:]
    else:
        seq_id = 0.5

    return path.parent, name, seq_id


def filter_main(args):
    pdb_list = set()

    with open(args.pdb_list, "r") as f:
        for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
            pdb_list.add(line)

    if args.verbose:
        print(f"Number of pid to filter out: {len(pdb_list)}")

    out_dir, out_name, out_seq_id = dataset_file_split(args.db_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir, in_name, in_seq_id = dataset_file_split(args.db_in)

    # df_pdb_csv
    with open(in_dir / f"{in_name}.csv", "r") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["id"] not in pdb_list]
    with open(out_dir / f"{out_name}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # seq_df_pdb_fasta
    records = [
        record for record in SeqIO.parse(in_dir / f"seq_{in_name}.fasta", "fasta")
        if record.id not in pdb_list
    ]
    SeqIO.write(records, out_dir / f"seq_{out_name}.fasta", "fasta-2line")

    # cluster_seqid_tsv
    rows = []
    with open(in_dir / f"cluster_seqid_{in_seq_id}_{in_name}_test.tsv", "r") as f:
        for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
            rep_id, pdb_id = line.split()
            if pdb_id not in pdb_list:
                rows.append((rep_id, pdb_id))
    with open(out_dir / f"cluster_seqid_{out_seq_id}_{out_name}_test.tsv", "w") as f:
        for rep_id, pdb_id in rows:
            f.write(f"{rep_id}\t{pdb_id}\n")

    if args.verbose:
        print(f"Output: {out_dir}")


def filter_add_argument(parser):
    parser.add_argument(
        "-i", "--pdb_list",
        type=pathlib.Path,
        help="list of pid will be filtered out"
    )
    parser.add_argument(
        "-o", "--db_out", type=pathlib.Path, default="./pdb_train", help="database output"
    )
    parser.add_argument("db_in", type=pathlib.Path, help="dataset input")


if __name__ == "__main__":
    import argparse

    commands = {
        "filter": (filter_main, filter_add_argument),
    }

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest="command", required=True)
    for cmd, (_, add_argument) in commands.items():
        cmd_parser = sub_parsers.add_parser(cmd)
        add_argument(cmd_parser)
        cmd_parser.add_argument(
            "-v", "--verbose", action="store_true", help="verbose"
        )

    args = parser.parse_args()

    work_fn, _ = commands[args.command]
    work_fn(args)
