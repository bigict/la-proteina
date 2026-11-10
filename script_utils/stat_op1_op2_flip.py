#!/usr/bin/env python3
"""统计 OP1/OP2 规范化规则在数据集上的实际翻转比例。

背景：OP1/OP2 是化学等价的磷酸氧，PDB 里的命名是任意的。在此规则接入之前，
所有区分 OP1/OP2 的特征（chi_angles_atoms[NA][0] 的 alpha 替代、
RelativeResidueOrientationPairFeat 的全部 5 个通道、
BackbonePairDistancesNanometerPairFeat 的 2/4 通道、
BackboneBondAnglesSeqFeat 的 theta_x/theta_y）都带有数据集级的标签噪声。
本脚本量化那个噪声：有多少比例的核酸残基会被规则交换 OP1/OP2。

判 flip 用的是生产代码 `canonicalize_phosphate_oxygens` 本身（对比交换前后的
OP1 槽位），而不是重新实现一遍规则，所以规则以后再改，本脚本自动跟着变。
signed volume 只用于诊断（区分"本来就规范"和"规则放弃判定"），不参与决策。

结果的读法：
  flip% 接近 50%  => 原始 OP1/OP2 命名与几何无关，等于在数据集内随机化，
                     此前那些特征承受的噪声是最坏情况。
  flip% 接近 0/100% => 原始命名本身高度一致，噪声担忧被高估了。

用法：
  python script_utils/stat_op1_op2_flip.py $DATA_PATH/pdb_train/processed
  python script_utils/stat_op1_op2_flip.py <dir_or_pt> [<dir_or_pt> ...] \
      --limit 2000 --workers 8 --csv op1_op2_flip.csv

参数可以是 .pt 文件，也可以是目录（递归搜索 *.pt）。读取的是预处理缓存里的
graph.coords / graph.coord_mask / graph.residue_type，即规范化之前的原始标注。
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from openfold.np import residue_constants as rc
from proteinfoundation.utils.coors_utils import canonicalize_phosphate_oxygens

# NB: 不用 rc.resname_to_idx 反查。它由 rc.resnames 构建，而 resnames 末尾误将
# 元组 ("UNK", PROT) 而非字符串 "UNK" 入了列，这里直接从 restypes 正向构建。
IDX_TO_NAME = {
    i: rc.restype_1to3[(one, rc.moltype(i))] for i, one in enumerate(rc.restypes)
}

I_OP1 = rc.atom_order["OP1"]
I_OP2 = rc.atom_order["OP2"]
I_P = rc.atom_order["P"]
I_O5P = rc.atom_order["O5'"]

N_SV_BUCKETS = 3  # |sv| < 1e-3, < 1e-2, < 1e-1
SV_THRESHOLDS = (1e-3, 1e-2, 1e-1)


def _bincount(x, minlength):
    """torch.bincount on a possibly-empty selection."""
    if x.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long)
    return torch.bincount(x, minlength=minlength)


def _scan_one(path_str, eps):
    """扫描单个 .pt，返回按残基类型聚合的计数。子进程中执行。"""
    path = Path(path_str)
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # 单个坏文件不该中断整轮统计
        return {"ok": False, "err": f"{path.name}: {type(exc).__name__}: {exc}"}

    missing = [k for k in ("coords", "coord_mask", "residue_type") if k not in graph]
    if missing:
        return {"ok": False, "err": f"{path.name}: missing {'/'.join(missing)}"}

    n_slots = rc.restype_num + 1
    coords = graph.coords.float()
    mask = graph.coord_mask.bool()
    rtype = graph.residue_type.long()

    is_na = (
        ((rtype >= rc.dna_from_idx) & (rtype <= rc.dna_to_idx))
        | ((rtype >= rc.rna_from_idx) & (rtype <= rc.rna_to_idx))
    )
    if not bool(is_na.any()):
        return {"ok": True, "rows": {}, "sv": None}

    sel = torch.nonzero(is_na, as_tuple=True)[0]
    c, m, rt = coords[sel], mask[sel], rtype[sel]

    # 生产规则实际会交换哪些残基
    canon = canonicalize_phosphate_oxygens(c, m, eps=eps)
    flipped = torch.any(canon[..., I_OP1, :] != c[..., I_OP1, :], dim=-1)

    # 以下仅用于诊断
    present = m[..., I_OP1] & m[..., I_OP2] & m[..., I_P] & m[..., I_O5P]
    sv = torch.linalg.vecdot(
        torch.linalg.cross(c[..., I_OP1, :] - c[..., I_P, :],
                           c[..., I_O5P, :] - c[..., I_P, :], dim=-1),
        c[..., I_OP2, :] - c[..., I_P, :],
        dim=-1,
    )
    degenerate = present & (sv.abs() <= eps)

    rows = {}
    n_rt = _bincount(rt, n_slots)
    flip_rt = _bincount(rt[flipped], n_slots)
    miss_rt = _bincount(rt[~present], n_slots)
    degen_rt = _bincount(rt[degenerate], n_slots)
    for i in torch.nonzero(n_rt, as_tuple=True)[0].tolist():
        rows[IDX_TO_NAME.get(i, f"idx{i}")] = (
            int(n_rt[i]), int(flip_rt[i]), int(miss_rt[i]), int(degen_rt[i])
        )

    av = sv[present & ~degenerate].abs()
    sv_stat = None
    if av.numel():
        sv_stat = {
            "n": int(av.numel()),
            "min": float(av.min()),
            "max": float(av.max()),
            "buckets": [int((av < t).sum()) for t in SV_THRESHOLDS],
        }
    return {"ok": True, "rows": rows, "sv": sv_stat}


def _collect(paths, workers, eps, quiet):
    total = len(paths)
    if total == 0:
        sys.exit("ERROR: 没找到任何 .pt 文件")

    rows = defaultdict(lambda: [0, 0, 0, 0])
    sv = {"n": 0, "min": float("inf"), "max": 0.0, "buckets": [0] * N_SV_BUCKETS}
    errors, done, n_na_files = [], 0, 0

    def _merge(res):
        nonlocal n_na_files
        if not res["ok"]:
            errors.append(res["err"])
            return
        if res["rows"]:
            n_na_files += 1
        for name, (n, f, mi, dg) in res["rows"].items():
            acc = rows[name]
            acc[0] += n
            acc[1] += f
            acc[2] += mi
            acc[3] += dg
        s = res["sv"]
        if s:
            sv["n"] += s["n"]
            sv["min"] = min(sv["min"], s["min"])
            sv["max"] = max(sv["max"], s["max"])
            for i in range(N_SV_BUCKETS):
                sv["buckets"][i] += s["buckets"][i]

    if workers <= 1:
        for p in paths:
            _merge(_scan_one(str(p), eps))
            done += 1
            if not quiet and done % 500 == 0:
                print(f"  ... {done}/{total}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_scan_one, str(p), eps) for p in paths]
            for fut in as_completed(futs):
                _merge(fut.result())
                done += 1
                if not quiet and done % 500 == 0:
                    print(f"  ... {done}/{total}", file=sys.stderr)

    return rows, sv, errors, total, n_na_files


DNA_NAMES = ("DA", "DC", "DG", "DT", "DX")
RNA_NAMES = ("A", "C", "G", "U", "X")


def _group(rows, names):
    acc = [0, 0, 0, 0]
    for nm in names:
        for i, v in enumerate(rows.get(nm, (0, 0, 0, 0))):
            acc[i] += v
    return acc


def _fmt_row(label, r):
    n, f, mi, dg = r
    pct = f"{100.0 * f / n:>7.2f}%" if n else f"{'-':>8s}"
    return f"{label:<10s} {n:>12,d} {f:>12,d} {pct:>9s} {mi:>10,d} {dg:>12,d}"


HEADER = (
    f"{'restype':<10s} {'n_res':>12s} {'n_flipped':>12s} {'flip%':>9s} "
    f"{'missing':>10s} {'degenerate':>12s}"
)


def _report(rows, sv, errors, n_files, n_na_files, eps):
    print()
    print(HEADER)
    print("-" * len(HEADER))

    present_names = [n for n in DNA_NAMES + RNA_NAMES if n in rows]
    other = sorted(set(rows) - set(present_names))
    for nm in present_names + other:
        print(_fmt_row(nm, rows[nm]))

    print("-" * len(HEADER))
    dna, rna = _group(rows, DNA_NAMES), _group(rows, RNA_NAMES)
    allr = [dna[i] + rna[i] for i in range(4)]
    for name in other:
        for i, v in enumerate(rows[name]):
            allr[i] += v
    print(_fmt_row("DNA", dna))
    print(_fmt_row("RNA", rna))
    print(_fmt_row("ALL", allr))

    n, f = allr[0], allr[1]
    print()
    print(f"扫描文件 {n_files:,d} 个（其中 {n_na_files:,d} 个含核酸），"
          f"核酸残基 {n:,d} 个")
    if n:
        print(f"OP1/OP2 被交换：{f:,d} 个 = {100.0 * f / n:.2f}%")
        pct = 100.0 * f / n
        if 40.0 <= pct <= 60.0:
            print("  => 接近 50%：原始命名与几何基本无关，规范化前那些区分")
            print("     OP1/OP2 的特征在数据集内被 effectively 随机化。")
        elif pct < 5.0 or pct > 95.0:
            print("  => 远离 50%：原始命名本身高度一致，标签噪声远小于预期，")
            print("     规范化主要是防御性的（保证跨来源一致）。")
        else:
            print("  => 介于两者之间：部分来源一致、部分不一致，混合噪声。")

    if allr[2]:
        print(f"注意：{allr[2]:,d} 个残基因缺 OP1/OP2/P/O5' 原子而无法判定，"
              f"规则按设计保持原样。")
    if allr[3]:
        print(f"注意：{allr[3]:,d} 个残基磷酸基近平面（|sv| <= {eps:g}），"
              f"规则放弃判定。")

    if sv["n"]:
        print()
        print(f"|signed volume| 分布（已判定的 {sv['n']:,d} 个残基，单位 A^3）：")
        print(f"  min = {sv['min']:.4g}    max = {sv['max']:.4g}")
        for t, c in zip(SV_THRESHOLDS, sv["buckets"]):
            print(f"  |sv| < {t:<6g} : {c:,d}")
        margin = sv["min"] / eps if eps else float("inf")
        print(f"  最小值 / eps = {margin:.3g}  "
              f"({'eps 安全，无误判风险' if margin > 100 else 'eps 偏大，建议调小'})")

    if errors:
        print()
        print(f"读取失败 {len(errors)} 个文件，前 10 条：", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restype", "n_res", "n_flipped", "flip_pct",
                    "n_missing_atom", "n_degenerate"])
        for nm in sorted(rows, key=lambda k: (k not in DNA_NAMES + RNA_NAMES, k)):
            n, f, mi, dg = rows[nm]
            w.writerow([nm, n, f, f"{100.0 * f / n:.4f}" if n else "", mi, dg])
    print(f"\nCSV 已写入 {path}")


def _resolve(inputs):
    out = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pt")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"WARN: 路径不存在，跳过：{p}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="统计 OP1/OP2 规范化规则的数据集翻转比例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", help=".pt 文件或目录（目录递归搜索 *.pt）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只扫描这么多文件；用 --seed 做确定性随机抽样，"
                         "而不是取排序后的前 N 个（后者在目录按分子类型分文件时会有偏）")
    ap.add_argument("--seed", type=int, default=0, help="--limit 抽样的随机种子")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="并行进程数（1 = 串行，便于调试）")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="近平面判定阈值，需与 canonicalize_phosphate_oxygens 一致")
    ap.add_argument("--csv", default=None, help="额外写出 per-restype CSV")
    ap.add_argument("--quiet", action="store_true", help="不打印进度")
    args = ap.parse_args()

    paths = _resolve(args.paths)
    if args.limit is not None:
        k = min(args.limit, len(paths))
        paths = random.Random(args.seed).sample(paths, k)
        if not args.quiet:
            print(f"--limit：随机抽样 {k} 个文件（seed={args.seed}）")
    if not args.quiet:
        print(f"待扫描 {len(paths)} 个 .pt，workers={args.workers}, eps={args.eps:g}")

    rows, sv, errors, n_files, n_na_files = _collect(
        paths, args.workers, args.eps, args.quiet
    )
    if not rows:
        sys.exit("ERROR: 没有统计到任何核酸残基——检查路径是否指向预处理缓存，"
                 "以及 restype_list 是否启用了 DNA/RNA")

    _report(rows, sv, errors, n_files, n_na_files, args.eps)
    if args.csv:
        _write_csv(args.csv, rows)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
