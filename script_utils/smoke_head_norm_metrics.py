"""Smoke test for the struct-head row-norm metrics added to GradAndWeightAnalysisCallback.

Runs without torch/numpy: the real method body is extracted from
proteinfoundation/utils/training_analysis_utils.py via AST and executed against a
minimal weight stand-in, so the assertions cover the shipped code rather than a
reimplementation of it.

Expected norms are derived in closed form from a parity pattern, not by repeating
the implementation's sum-of-squares, so a wrong axis or a mean-of-abs slip fails.

Usage:
    python script_utils/smoke_head_norm_metrics.py
"""

import ast
import math
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "proteinfoundation" / "utils" / "training_analysis_utils.py"
METHOD = "_get_struct_head_row_norms"
CALLER = "on_before_optimizer_step"
HEAD_KEY = "decoder.struct_linear.1.weight"


class _Vec:
    def __init__(self, values):
        self.values = values

    def mean(self):
        return sum(self.values) / len(self.values)


class _Weight:
    """Only the operations the callback performs on a parameter."""

    def __init__(self, rows):
        self.rows = rows
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, sl):
        assert isinstance(sl, slice) and sl.step is None, "unexpected indexing"
        return _Weight(self.rows[sl])

    def norm(self, dim):
        assert dim == 1, f"expected a row norm, got dim={dim}"
        return _Vec([math.sqrt(sum(v * v for v in r)) for r in self.rows])


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Module:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return list(self._named)


class _RC:
    def __init__(self, atom_order, atom_type_num):
        self.atom_order = atom_order
        self.atom_type_num = atom_type_num


def _extract(name):
    tree = ast.parse(TARGET.read_text())
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "GradAndWeightAnalysisCallback"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return cls, fn


def _load_method(rc):
    _, fn = _extract(METHOD)
    ns = {"torch": type("T", (), {"no_grad": _NoGrad}), "rc": rc}
    exec(textwrap.dedent(ast.get_source_segment(TARGET.read_text(), fn)), ns)
    return ns[METHOD]


def _pattern(base, n_rows, fan_in, offset=0):
    """Row i, col j = base * (1 + (i+offset)%2) * (1 + j%3)."""
    return [
        [base * (1 + (i + offset) % 2) * (1 + j % 3) for j in range(fan_in)]
        for i in range(n_rows)
    ]


def _closed_form(base, n_rows, fan_in):
    """Mean row norm of _pattern, counted by parity instead of summed."""
    per_row = math.sqrt(sum((1 + j % 3) ** 2 for j in range(fan_in)))
    odd = sum(1 for i in range(n_rows) if i % 2)
    return base * per_row * (n_rows + odd) / n_rows


def _weight(lo, hi, fan_in, prot_base, na_base):
    rows = _pattern(prot_base, lo, fan_in) + _pattern(na_base, hi - lo, fan_in)
    return _Weight(rows)


def main():
    src = TARGET.read_text()
    cls, fn = _extract(METHOD)

    # The metric must be wired into the callback, not merely defined.
    caller = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == CALLER
    )
    calls = [
        n for n in ast.walk(caller)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == METHOD
    ]
    assert len(calls) == 1, f"{CALLER} must call {METHOD} exactly once, found {len(calls)}"
    print(f"PASS  {CALLER} calls {METHOD} (AST-verified call site)")

    # Row bounds: prefer the real constants, fall back to the documented layout.
    try:
        sys.path.insert(0, str(REPO))
        from openfold.np import residue_constants as real_rc

        n_prot, n_atoms = real_rc.atom_order["P"], real_rc.atom_type_num
        print(f"      row bounds from real residue_constants: P={n_prot}, atoms={n_atoms}")
    except Exception as e:
        n_prot, n_atoms = 37, 65
        print(f"      SKIP real residue_constants ({type(e).__name__}); "
              f"using documented layout P={n_prot}, atoms={n_atoms}")
    lo, hi, fan_in = n_prot * 3, n_atoms * 3, 768
    rc = _RC({"P": n_prot}, n_atoms)
    method = _load_method(rc)

    p_exp = _closed_form(0.0208, lo, fan_in)
    n_exp = _closed_form(1.0, hi - lo, fan_in)

    # 1. randn-scale NA block: the defect the metric exists to expose.
    w = _weight(lo, hi, fan_in, 0.0208, 1.0)
    before = [list(r) for r in w.rows]
    m = method(None, _Module([(f"model.{HEAD_KEY}", w)]))
    assert set(m) == {"prot_head_row_norm", "na_head_row_norm", "na_over_prot_head_norm"}, m
    assert math.isclose(m["prot_head_row_norm"], p_exp, rel_tol=1e-9), m
    assert math.isclose(m["na_head_row_norm"], n_exp, rel_tol=1e-9), m
    assert math.isclose(m["na_over_prot_head_norm"], n_exp / p_exp, rel_tol=1e-9), m
    assert m["na_over_prot_head_norm"] > 40, m
    assert w.rows == before, "the metric mutated the weight"
    print(f"PASS  randn-scale NA block -> ratio {m['na_over_prot_head_norm']:.2f}x, "
          f"prot {m['prot_head_row_norm']:.4f}, NA {m['na_head_row_norm']:.4f}")

    # 2. zero-initialized NA block: ratio must be exactly 0, not nan/inf.
    w = _weight(lo, hi, fan_in, 0.0208, 0.0)
    m = method(None, _Module([(HEAD_KEY, w)]))
    assert m["na_head_row_norm"] == 0.0, m
    assert m["na_over_prot_head_norm"] == 0.0, m
    assert math.isclose(m["prot_head_row_norm"], p_exp, rel_tol=1e-9), m
    print("PASS  zero-initialized NA block -> NA norm 0, ratio 0 (no nan/inf)")

    # 3. healthy head: both blocks at the same scale -> ratio ~1.
    w = _weight(lo, hi, fan_in, 0.0208, 0.0208)
    m = method(None, _Module([(HEAD_KEY, w)]))
    assert 0.9 < m["na_over_prot_head_norm"] < 1.1, m
    print(f"PASS  matched-scale head -> ratio {m['na_over_prot_head_norm']:.4f}")

    # 4. protein-only build: no 'P' atom, so nothing is logged.
    m_prot_only = _load_method(_RC({"CA": 1}, n_prot))
    assert m_prot_only(None, _Module([(HEAD_KEY, _weight(0, lo, fan_in, 0.0208, 0))])) == {}
    print("PASS  protein-only build (no 'P' in atom_order) -> {}")

    # 5. head absent from the module.
    assert method(None, _Module([("encoder.lin.weight", _weight(0, lo, fan_in, 0.1, 0))])) == {}
    print("PASS  no struct head in named_parameters -> {}")

    # 6. row count disagrees with the constants (stale checkpoint layout).
    assert method(None, _Module([(HEAD_KEY, _weight(0, lo, fan_in, 0.0208, 0))])) == {}
    print(f"PASS  head with {lo} rows while constants expect {hi} -> {{}}")

    # 7. all-zero protein block: report the norms, omit the undefined ratio.
    w = _weight(lo, hi, fan_in, 0.0, 1.0)
    m = method(None, _Module([(HEAD_KEY, w)]))
    assert set(m) == {"prot_head_row_norm", "na_head_row_norm"}, m
    assert m["prot_head_row_norm"] == 0.0, m
    print("PASS  zero protein block -> ratio omitted instead of dividing by zero")

    # 8. the head is found among unrelated parameters, at any nesting depth.
    w = _weight(lo, hi, fan_in, 0.0208, 1.0)
    named = [
        ("encoder.blocks.0.lin.weight", _weight(0, 8, fan_in, 0.1, 0)),
        ("decoder.struct_linear.0.weight", _weight(0, 4, fan_in, 0.2, 0)),
        (f"net.model.{HEAD_KEY}", w),
        ("decoder.logit_linear.1.weight", _weight(0, 8, fan_in, 0.3, 0)),
    ]
    m = method(None, _Module(named))
    assert math.isclose(m["na_over_prot_head_norm"], n_exp / p_exp, rel_tol=1e-9), m
    print("PASS  head located among decoys at depth 3 (LayerNorm/logit siblings skipped)")

    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())