#!/usr/bin/env python3
"""统计 OP1/OP2 规范化规则在数据集上的实际翻转比例。

背景：OP1/OP2 是化学等价的磷酸氧，PDB 里的命名是任意的。在此规则接入之前，
所有区分 OP1/OP2 的特征（chi_angles_atoms[NA][0] 的 alpha 替代、
RelativeResidueOrientationPairFeat 的全部 5 个通道、
BackbonePairDistancesNanometerPairFeat 的 2/4 通道、
BackboneBondAnglesSeqFeat 的 theta_x/theta_y）都带有数据集级的标签噪声。
本脚本量化那个噪声。

判 flip 用的是生产代码 `canonicalize_phosphate_oxygens` 本身（对比交换前后的
OP1 槽位），而不是重新实现一遍规则，所以规则以后再改，本脚本自动跟着变。
signed volume 只用于诊断（区分"本来就规范"和"规则放弃判定"），不参与决策。

两个层次的统计：

1) 全局 / per-restype 翻转率。接近 50% 说明原始命名与几何无关。

2) per-entry 翻转率分布。全局 50% 有两种完全不同的成因，必须区分：
   (a) entry 内一致、entry 间不同 —— 每个 entry 的翻转率贴近 0% 或 100%，
       分布双峰。此时 OP1/OP2 是系统性的 per-entry 属性，与 deposition
       软件/年代相关，是**可学习的混杂因子**，比随机噪声更糟。
   (b) entry 内即随机 —— 翻转率集中在 50% 附近，是纯噪声。
   两种情形下论文能主张的强度不同，修复的价值表述也不同。

per-entry 比率的分母是**可判定残基数**（剔除缺原子与近平面的残基），
否则那些残基会把比率稀释向 0%，伪造出"entry 内一致"。
另有 --min-na-res 过滤核酸残基过少的 entry：只有 1 个核酸残基的 entry
翻转率必然是 0% 或 100%，会虚高"一致"比例。

用法：
  python script_utils/stat_op1_op2_flip.py $DATA_PATH/pdb_train/processed
  python script_utils/stat_op1_op2_flip.py <dir_or_pt> [<dir_or_pt> ...] \
      --limit 2000 --workers 8 --csv op1_op2_flip.csv \
      --by-entry-csv op1_op2_by_entry.csv --min-na-res 4

参数可以是 .pt 文件，也可以是目录（递归搜索 *.pt）。读取的是预处理缓存里的
graph.coords / graph.coord_mask / graph.residue_type，即规范化之前的原始标注。
注意不要用 pdb_utils.from_pdb_string 来量原始混淆——它自己就会做规范化。
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

ENTRY_BUCKET_LABELS = (
    "= 0%", "(0%,5%)", "[5%,45%)", "[45%,55%]", "(55%,95%)", "[95%,100%)", "= 100%",
)


def _entry_bucket(pct):
    """把 per-entry 翻转率映射到互斥且完备的 7 个桶之一。"""
    if pct == 0.0:
        return 0
    if pct < 5.0:
        return 1
    if pct < 45.0:
        return 2
    if pct <= 55.0:
        return 3
    if pct < 95.0:
        return 4
    if pct < 100.0:
        return 5
    return 6


def _bincount(x, minlength):
    """torch.bincount on a possibly-empty selection."""
    if x.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long)
    return torch.bincount(x, minlength=minlength)


def _scan_one(path_str, eps):
    """扫描单个 .pt，返回按残基类型聚合的计数 + 该 entry 的汇总。子进程中执行。"""
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
        return {"ok": True, "rows": {}, "sv": None, "entry": None}

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

    entry = {
        "name": path.stem,
        "n_na": int(n_rt.sum()),
        "n_flip": int(flip_rt.sum()),
        "n_skip": int(miss_rt.sum() + degen_rt.sum()),
    }

    av = sv[present & ~degenerate].abs()
    sv_stat = None
    if av.numel():
        sv_stat = {
            "n": int(av.numel()),
            "min": float(av.min()),
            "max": float(av.max()),
            "buckets": [int((av < t).sum()) for t in SV_THRESHOLDS],
        }
    return {"ok": True, "rows": rows, "sv": sv_stat, "entry": entry}


def _collect(paths, workers, eps, quiet):
    total = len(paths)
    if total == 0:
        sys.exit("ERROR: 没找到任何 .pt 文件")

    rows = defaultdict(lambda: [0, 0, 0, 0])
    sv = {"n": 0, "min": float("inf"), "max": 0.0, "buckets": [0] * N_SV_BUCKETS}
    entries = []
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
        if res.get("entry"):
            entries.append(res["entry"])
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

    return rows, sv, errors, total, n_na_files, entries


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


def _select_entries(entries, min_na_res):
    """按可判定残基数过滤 entry，返回 (kept, n_small, n_undecided)。

    kept 元素为 (name, n_decided, n_flip, pct)。分母用可判定数而非全部核酸
    残基数，避免缺原子/近平面残基把比率稀释向 0%。
    """
    kept, n_small, n_undecided = [], 0, 0
    for e in entries:
        n_dec = e["n_na"] - e["n_skip"]
        if n_dec <= 0:
            n_undecided += 1
            continue
        if n_dec < min_na_res:
            n_small += 1
            continue
        kept.append((e["name"], n_dec, e["n_flip"], 100.0 * e["n_flip"] / n_dec))
    return kept, n_small, n_undecided


def _report_entries(entries, min_na_res):
    """打印 per-entry 翻转率分布，并区分 entry 内一致 vs entry 内随机。"""
    if not entries:
        return
    kept, n_small, n_undecided = _select_entries(entries, min_na_res)

    print()
    print("=" * 72)
    print("per-entry 翻转率分布")
    print("=" * 72)
    print(f"含核酸的 entry: {len(entries):,d}")
    if n_undecided:
        print(f"  排除 {n_undecided:,d} 个：全部核酸残基都无法判定（缺原子/近平面）")
    if n_small:
        print(f"  排除 {n_small:,d} 个：可判定核酸残基 < {min_na_res}"
              f"（残基太少时比率必然贴近 0%/100%，会虚高'一致'比例）")
    if not kept:
        print(f"  没有 entry 满足 --min-na-res={min_na_res}，无法给出分布。")
        return
    print(f"  纳入统计: {len(kept):,d}")

    counts = [0] * len(ENTRY_BUCKET_LABELS)
    for _, _, _, pct in kept:
        counts[_entry_bucket(pct)] += 1

    n_kept = len(kept)
    print()
    print(f"{'翻转率区间':<14s} {'entry 数':>10s} {'占比':>9s}  直方图")
    print("-" * 72)
    for label, c in zip(ENTRY_BUCKET_LABELS, counts):
        frac = c / n_kept
        bar = "#" * int(round(frac * 40))
        print(f"{label:<14s} {c:>10,d} {frac * 100:>8.2f}%  {bar}")

    n_res = sum(k[1] for k in kept)
    n_flip = sum(k[2] for k in kept)
    agg = 100.0 * n_flip / n_res
    cons = sum(1 for k in kept if k[3] < 5.0 or k[3] >= 95.0)
    mixed = sum(1 for k in kept if 45.0 <= k[3] <= 55.0)
    cons_f, mixed_f = cons / n_kept, mixed / n_kept

    print()
    print(f"纳入 entry 的合计翻转率（残基加权）: {n_flip:,d}/{n_res:,d} = {agg:.2f}%")
    print(f"entry 内一致（<5% 或 >=95%）: {cons:,d}/{n_kept:,d} = {cons_f * 100:.2f}%")
    print(f"entry 内混杂（45%–55%）    : {mixed:,d}/{n_kept:,d} = {mixed_f * 100:.2f}%")

    print()
    if cons_f >= 0.8 and 40.0 <= agg <= 60.0:
        print("=> 情形 (a)：entry 内高度一致，但 entry 之间约定不同（分布双峰）。")
        print("   OP1/OP2 是系统性的 per-entry 属性，很可能与 deposition 软件/年代")
        print("   相关，属于**可学习的混杂因子**——比随机噪声更糟，因为模型可能")
        print("   真的去拟合它。建议用 --by-entry-csv 导出后按年代/分辨率/实验方法")
        print("   交叉分析来确认相关性。")
    elif mixed_f >= 0.5:
        print("=> 情形 (b)：entry 内部即随机，属于纯噪声（方差），不是可学习混杂。")
    elif cons_f >= 0.8:
        print("=> entry 内一致，且整体翻转率偏离 50%：原始命名本身相当一致，")
        print("   规范化主要是防御性的（保证跨来源/跨工具一致），噪声担忧被高估。")
    else:
        print("=> 混合情形：部分 entry 一致、部分 entry 内随机。建议导出 per-entry")
        print("   CSV 后分组看，单一全局数字不足以描述。")


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
        pct = 100.0 * f / n
        print(f"OP1/OP2 被交换：{f:,d} 个 = {pct:.2f}%")
        if 40.0 <= pct <= 60.0:
            print("  => 接近 50%：原始命名与几何基本无关。但还需看下面的 per-entry")
            print("     分布才能区分'entry 间不同'与'entry 内随机'。")
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


def _write_entry_csv(path, entries, min_na_res):
    """导出每个 entry 的计数，便于与 PDB 元数据（年代/分辨率/方法）交叉分析。"""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "n_na", "n_decided", "n_flipped", "flip_pct",
                    "n_skipped", "included", "bucket"])
        for e in sorted(entries, key=lambda x: x["name"]):
            n_dec = e["n_na"] - e["n_skip"]
            included = n_dec >= min_na_res and n_dec > 0
            pct = (100.0 * e["n_flip"] / n_dec) if n_dec > 0 else None
            w.writerow([
                e["name"], e["n_na"], n_dec, e["n_flip"],
                f"{pct:.4f}" if pct is not None else "",
                e["n_skip"], int(included),
                ENTRY_BUCKET_LABELS[_entry_bucket(pct)] if included else "",
            ])
    print(f"per-entry CSV 已写入 {path}")


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
        description="统计 OP1/OP2 规范化规则的数据集翻转比例（全局 + per-entry 分布）",
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
    ap.add_argument("--min-na-res", type=int, default=4,
                    help="per-entry 分布只纳入可判定核酸残基数 >= 此值的 entry"
                         "（默认 4；太少的 entry 比率必然贴近 0%%/100%%，会虚高一致性）")
    ap.add_argument("--csv", default=None, help="额外写出 per-restype CSV")
    ap.add_argument("--by-entry-csv", default=None,
                    help="额外写出 per-entry CSV（含未纳入统计的 entry，便于交叉分析）")
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

    rows, sv, errors, n_files, n_na_files, entries = _collect(
        paths, args.workers, args.eps, args.quiet
    )
    if not rows:
        sys.exit("ERROR: 没有统计到任何核酸残基——检查路径是否指向预处理缓存，"
                 "以及 restype_list 是否启用了 DNA/RNA")

    _report(rows, sv, errors, n_files, n_na_files, args.eps)
    _report_entries(entries, args.min_na_res)
    if args.csv:
        _write_csv(args.csv, rows)
    if args.by_entry_csv:
        _write_entry_csv(args.by_entry_csv, entries, args.min_na_res)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
