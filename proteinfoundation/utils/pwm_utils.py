import math

from scipy.stats import pearsonr, entropy


def ic_weighted_pcc(true_column, pred_column, gt=None):
    if gt is None:
        gt = true_column
    pcc, _ = pearsonr(true_column, pred_column)
    ic_weight = entropy(gt, [0.25] * 4, base=2) / 2

    return pcc * ic_weight, ic_weight

def ic_corr(true, pred, bkg=None):
    """Pearson correlation of per-position Information Content."""
    if bkg is None:
        bkg = [0.25] * 4
    ic_true = entropy(true, bkg, base=2, axis=1)
    ic_pred = entropy(pred, bkg, base=2, axis=1)

    r = pearsonr(ic_true, ic_pred)  # safe pearsonr
    return 0.0 if math.isnan(r.statistic) else r.statistic

def ic_diff(pwm, pred, bkg=None):
    """Negative mean absolute IC difference."""
    if bkg is None:
        bkg = [0.25] * 4
    columnwise = -abs(
        entropy(pwm, bkg, base=2, axis=1) - entropy(pred, bkg, base=2, axis=1)
    )
    return columnwise.mean()

def brier_multi(targets, probs):
    """Multi-class Brier score: sum of squared diff per position, then average."""
    errors = (probs - targets) ** 2
    return errors.sum(1).mean()

def mae(targets, probs):
    """Mean Absolute Error: sum |diff| per position, then average."""
    errors = abs(probs - targets)
    return errors.sum(1).mean()

def pwm_seq_align(pwm, seq, min_overlap=1):
    """
    Ungapped alignment maximizing IC-weighted PCC dot product.
    Mirrors deeppbs.align_PWM_seq.ungappedAlign.

    Parameters
    ----------
    pwm        : ndarray (Lp, 4)  — the longer array
    seq        : ndarray (Ls, 4)  — the shorter array
    min_overlap: int              — reference for IC weighting (usually the JASPAR PWM)

    Returns
    -------
    seq_start, pwm_start, overlap_len, best_score
    """
    assert pwm.shape[1] == 4
    assert min_overlap <= min(pwm.shape[0], seq.shape[0])
    max_score = -9999
    opt_i = opt_j = opt_k = 0
    l, s = pwm.shape[0], seq.shape[0]

    for i in range(0, s):
        for k in range(min_overlap, s - i + 1):   # k overlap length
            for j in range(0, l - k + 1):
                score = 0
                for col in range(k):    # IC_weighted_PCC scoring
                    col_score, _ = ic_weighted_pcc(
                        seq[i + col, :], pwm[j + col, :], gt=pwm[j + col, :]
                    )
                    score += col_score
                if score > max_score:
                    max_score = score
                    opt_i = i
                    opt_j = j
                    opt_k = k

    return opt_i, opt_j, opt_k, max_score
