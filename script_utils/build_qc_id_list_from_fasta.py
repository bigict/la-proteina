import argparse
import math
from collections import Counter
from pathlib import Path


def iter_fasta(path: Path):
    pid = None
    chunks = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pid is not None:
                    yield pid, "".join(chunks)
                pid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if pid is not None:
        yield pid, "".join(chunks)


def entropy_norm(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h / math.log2(5)


def default_path(fasta: Path, suffix: str) -> Path:
    return fasta.parent / f"{fasta.stem}{suffix}"


def main(args):
    out_drop_ids = args.out_drop_ids or default_path(args.seq_fasta, "_drop_id_list.txt")
    out_drop_ids.parent.mkdir(parents=True, exist_ok=True)

    with open(out_drop_ids, "w") as f_ids:
        for pid, raw_seq in iter_fasta(args.seq_fasta):
            seq = "".join(raw_seq.upper().split())
            length = len(seq)
            x_ratio = (seq.count("X") / length) if length > 0 else 1.0
            h_norm = entropy_norm(seq)

            drop = (
                (length < args.min_length)
                or (x_ratio > args.max_x_ratio)
                or (h_norm < args.min_entropy_norm)
            )
            if not drop:
                continue

            f_ids.write(f"{pid}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build drop_id_list.txt")
    parser.add_argument("seq_fasta", type=Path, help="Path to seq_*.fasta / seq_df_*.fasta")
    parser.add_argument("--min_length", type=int, default=15)
    parser.add_argument("--max_x_ratio", type=float, default=0.30)
    parser.add_argument("--min_entropy_norm", type=float, default=0.20)
    parser.add_argument("-o", dest="out_drop_ids", type=Path, default=None)
    main(parser.parse_args())
