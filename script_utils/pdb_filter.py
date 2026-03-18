import csv
import pathlib

from Bio import SeqIO


def main(args):
    pdb_list = set()

    with open(args.pdb_list, "r") as f:
        for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
            pdb_list.add(line)

    if args.verbose:
        print(f"Number of pid to filter out: {len(pdb_list)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # df_pdb_csv
    with open(args.df_pdb_csv, "r") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["id"] not in pdb_list]
    with open(args.output_dir / args.df_pdb_csv.name, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # seq_df_pdb_fasta
    records = [
        record for record in SeqIO.parse(args.seq_df_pdb_fasta, "fasta")
        if record.id not in pdb_list
    ]
    SeqIO.write(
        records, args.output_dir / args.seq_df_pdb_fasta.name, "fasta-2line"
    )

    # cluster_seqid_tsv
    rows = []
    with open(args.cluster_seqid_tsv, "r") as f:
        for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
            rep_id, pdb_id = line.split()
            if pdb_id not in pdb_list:
                rows.append((rep_id, pdb_id))
    with open(args.output_dir / args.cluster_seqid_tsv.name, "w") as f:
        for rep_id, pdb_id in rows:
            f.write(f"{rep_id}\t{pdb_id}\n")

    if args.verbose:
        print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--pdb_list",
        type=pathlib.Path,
        help="list of pid will be filtered out"
    )
    parser.add_argument(
        "-o", "--output_dir", type=pathlib.Path, default=".", help="output_dir"
    )
    parser.add_argument("df_pdb_csv", type=pathlib.Path, help="df_pdb_*.csv")
    parser.add_argument(
        "seq_df_pdb_fasta", type=pathlib.Path, help="df_pdb_*.fasta"
    )
    parser.add_argument(
        "cluster_seqid_tsv", type=pathlib.Path, help="cluster_seqid_*.tsv"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    main(args)
