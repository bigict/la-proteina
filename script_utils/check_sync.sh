#!/bin/bash
# La-Proteina 改动同步校验：在训练机上跑，全部 OK 才说明代码同步完整。
# 放在 script_utils/ 下，自动 cd 到仓库根目录。
#
# marker 用 grep -F 做定点字符串匹配，不用正则——避免转义写错导致
# "匹配不上但看起来像代码没同步" 这类静默误判。
set -u
cd "$(dirname "$0")/.." || { echo "ERROR: 无法 cd 到仓库根目录" >&2; exit 2; }

# 哨兵检查：定位失败时必须硬报错，不能退化成"全是 MISMATCH"的错误结论。
for sentinel in \
    proteinfoundation/utils/coors_utils.py \
    proteinfoundation/nn/feature_factory.py \
    script_utils/model_ops.py; do
    if [ ! -f "$sentinel" ]; then
        echo "ERROR: 找不到 $sentinel —— 当前目录不是仓库根 ($(pwd))" >&2
        echo "       请把本脚本放在 <repo>/script_utils/ 下再运行。" >&2
        exit 2
    fi
done

fail=0
check() { # check <file> <expected_count> <fixed_string>
    local n
    if [ ! -f "$1" ]; then
        printf 'MISSING   --/%s  %s\n' "$2" "$1"
        fail=1
        return
    fi
    n=$(grep -cF -- "$3" "$1")
    if [ "$n" = "$2" ]; then
        printf 'OK        %s/%s  %s  «%s»\n' "$n" "$2" "$1" "$3"
    else
        printf 'MISMATCH  %s/%s  %s  «%s»\n' "$n" "$2" "$1" "$3"
        fail=1
    fi
}

echo "repo: $(pwd)"
echo

# --- 第1轮：checkpoint 扩容的初始化尺度（torch.randn -> nn.Linear 默认尺度 / zero-init）---
check script_utils/model_ops.py 1 'def _linear_default_init'
check script_utils/model_ops.py 0 'torch.randn('
check script_utils/model_ops.py 2 'torch.zeros('
check script_utils/model_ops.py 1 'assert b.shape[1] == enc_fan_in'
check script_utils/model_ops.py 1 'assert b.shape[1] == fan_in == 909'

# --- 第2轮：角度分箱（循环化 + 键角改用 [0,pi] 值域）---
check proteinfoundation/nn/feature_factory.py 1 'def bin_and_one_hot_periodic'
check proteinfoundation/nn/feature_factory.py 1 'def bin_and_one_hot_bond_angle'
check proteinfoundation/nn/feature_factory.py 1 'N_ANGLE_BINS = 21'
check proteinfoundation/nn/feature_factory.py 0 'linspace(-torch.pi'

# --- 第3轮：OP1/OP2 源头规范化 ---
check proteinfoundation/utils/coors_utils.py 1 'def canonicalize_phosphate_oxygens'
check proteinfoundation/datasets/transforms.py 1 'class CanonicalizePhosphateOxygensTransform'
check proteinfoundation/utils/motif_utils.py 2 'x_motif = canonicalize_phosphate_oxygens'
check proteinfoundation/utils/pdb_utils.py 1 'atom_positions = canonicalize_phosphate_oxygens'
for c in ucond motif_aa motif_tip; do
    check "configs/dataset/pdb/pdb_train_$c.yaml" 1 'CanonicalizePhosphateOxygensTransform'
done

# --- 上游 d2d31d4 / 37f71e9：struct 与 seq 改为 Σ(w·L)/(Σ(w·mask)+eps) ---
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'err = (err * w).sum(dim=-1) / ((mask * w).sum(dim=-1) + eps)'
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'torch.sum(seq_loss * w, dim=-1) / (torch.sum(mask * w, dim=-1) + eps)'

# --- 第4轮：KL 日志改用未加权（归一化本身由上游 d2d31d4 完成）---
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'nres = torch.sum(mask, dim=-1)'
check proteinfoundation/partial_autoencoder/autoencoder.py 0 'mask > 0'
# 权重只应出现在 compute_kl_penalty 调用处；日志路径必须用未加权 mask
check proteinfoundation/partial_autoencoder/autoencoder.py 1 'token_level_w(aatype, self.cfg_ae.loss.kl)'

# --- 附加工具：OP1/OP2 翻转比例统计 ---
# 必须 import 生产规则本身；若这里为 0，说明脚本被改成自己重实现了一份规则，
# 统计结果就会与训练管线脱钩。
check script_utils/stat_op1_op2_flip.py 1 'from proteinfoundation.utils.coors_utils import canonicalize_phosphate_oxygens'
# per-entry 分布：区分 "entry 间约定不同" 与 "entry 内随机"
check script_utils/stat_op1_op2_flip.py 1 'def _entry_bucket'
check script_utils/stat_op1_op2_flip.py 1 'def _select_entries'
check script_utils/stat_op1_op2_flip.py 1 'def _report_entries'
check script_utils/stat_op1_op2_flip.py 1 'def _write_entry_csv'
check script_utils/stat_op1_op2_flip.py 1 '"--min-na-res"'
check script_utils/stat_op1_op2_flip.py 1 '"--by-entry-csv"'
# 分母必须是可判定残基数（_select_entries 与 _write_entry_csv 各一处），
# 否则缺原子/近平面残基会把比率稀释向 0%，伪造出 "entry 内一致"
check script_utils/stat_op1_op2_flip.py 2 'n_dec = e["n_na"] - e["n_skip"]'

echo
if [ "$fail" = 0 ]; then
    echo "==> 全部同步 OK"
else
    echo "==> 有 MISMATCH/MISSING，代码未同步完整，不要开始训练"
fi
exit "$fail"
