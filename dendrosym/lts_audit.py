"""Audit a config's evolution equations for local time stepping (LTS).

Three things decide whether a system can run under ts::ExplicitNUTS and what its
step ladder must respect. All three are read off the right-hand-side
expressions, which are not modified:

  stiff rates     parameters that multiply an evolution variable in its own
                  equation (BSSN's eta in dB/dt = ... - eta B; CCZ4's kappa1 in
                  dTheta/dt). Their size bounds the stable step of every block,
                  so compute_lts_ts_offset has to cap the ladder with them.
  constraint reads  right-hand-side terms that read a constraint variable (or
                  its derivative). The stepper refreshes constraints once per
                  coarse step, so inside it they are stale.
  dt terms        right-hand-side terms that read the step size. Blocks take
                  different steps under LTS, so no single dt exists.

Usage:
    audit = lts_audit.audit(config)
    print(audit.summary())

A rate is the coefficient d(rhs_u)/du with every derivative symbol set to zero
and every field set to a flat background (alpha = chi = 1, flat conformal
metric, everything else 0); only the parameter dependence of that coefficient is
kept. This finds the damping rates and the gauge/constraint-damping couplings;
it does not bound the Jacobian's off-diagonal growth, which the CFL step and the
LTS gates cover.
"""

import re
from dataclasses import dataclass, field

import sympy as sym
from sympy.core.function import AppliedUndef

_INDEXED = re.compile(r"^(?P<base>[A-Za-z_]\w*)\[(?P<idx>\d+)\]$")
_DT_NAME = re.compile(r"^(dt|dT|delta_?t|time_?step\w*|curr_?dt|th)$", re.I)
_DERIV_FUNC = re.compile(r"^(?:ko|a)?grad\d*$")
_DIAG_METRIC = re.compile(r"^gt_?(?:mat)?(\d)\1$")


@dataclass
class StiffRate:
    """The parameter-dependent part of one variable's own linearization.

    `expr` is d(rhs_u)/du on the flat background, a function of the parameters
    only (BSSN's B: -eta). Its magnitude is the rate that bounds the step.
    """

    var: str
    params: tuple
    expr: object


@dataclass
class LtsAudit:
    stiff_rates: list = field(default_factory=list)
    constraint_reads: dict = field(default_factory=dict)
    dt_terms: dict = field(default_factory=dict)
    indeterminate: list = field(default_factory=list)

    @property
    def rate_params(self):
        """Parameter names that bound the step, sorted."""
        return sorted({p for r in self.stiff_rates for p in r.params})

    @property
    def clean(self):
        """True when LTS can use the equations as they are."""
        return not self.constraint_reads and not self.dt_terms

    def summary(self):
        lines = []
        if self.stiff_rates:
            for r in self.stiff_rates:
                lines.append(f"stiff rate  {r.var}: {r.expr}")
        else:
            lines.append("stiff rate  none found")
        for var, syms in sorted(self.constraint_reads.items()):
            lines.append(f"constraint read  {var}: {', '.join(syms)}")
        for var, syms in sorted(self.dt_terms.items()):
            lines.append(f"dt term  {var}: {', '.join(syms)}")
        if self.indeterminate:
            lines.append(
                "indeterminate (background singular): " + ", ".join(self.indeterminate)
            )
        return "\n".join(lines)


def default_background(grid_bases):
    """Flat NR background: alpha = chi = 1, diagonal conformal metric 1, rest 0."""
    bg = {}
    for base in grid_bases:
        if base in ("alpha", "chi") or _DIAG_METRIC.match(base):
            bg[base] = 1
        else:
            bg[base] = 0
    return bg


def _strip_idx(name, idx_str):
    return name[: -len(idx_str)] if name.endswith(idx_str) else None


def _split(config, exprs, idx_str):
    """Free symbols of `exprs` sorted into grid quantities and parameters."""
    grid, params = {}, {}
    for e in exprs:
        for s in e.free_symbols:
            base = _strip_idx(s.name, idx_str)
            (grid if base is not None else params)[s] = base or s.name
    return grid, params


def audit(config, background=None):
    """Audit `config`'s evolution right-hand sides. Does not modify them.

    background: optional {grid base name: value} overriding default_background.
    """
    exprs, names, _, _, _ = config._extract_rhs_expressions(
        "evolution", append_rhs_to_var=False
    )
    idx_str = config.idx_str
    evo = set(names)
    cons = set(config.all_var_names.get("constraint", []))

    grid, params = _split(config, exprs, idx_str)

    def derivative_free(expr):
        """expr with every derivative application grad(i, f), grad2(i, j, f), ...
        set to 0: the flat background is uniform."""
        zero = {
            a: sym.Integer(0)
            for a in expr.atoms(AppliedUndef)
            if _DERIV_FUNC.match(a.func.__name__)
        }
        return expr.xreplace(zero)

    result = LtsAudit()

    for expr, name in zip(exprs, names):
        reads = sorted(
            f"{b}{idx_str}"
            for s, b in grid.items()
            if s in expr.free_symbols and b in cons
        )
        if reads:
            result.constraint_reads[name] = reads
        dts = sorted(
            s.name
            for s in expr.free_symbols
            if _DT_NAME.match(_strip_idx(s.name, idx_str) or s.name)
        )
        if dts:
            result.dt_terms[name] = dts

    bg_vals = default_background(set(grid.values()))
    if background:
        bg_vals.update(background)
    bg = {s: sym.Integer(bg_vals.get(b, 0)) for s, b in grid.items()}

    by_name = {_strip_idx(s.name, idx_str): s for s in grid}
    for expr, name in zip(exprs, names):
        su = by_name.get(name)
        e0 = derivative_free(expr)
        if su is None or su not in e0.free_symbols:
            continue
        coef = e0.diff(su).xreplace(bg)
        coef = sym.factor(sym.together(coef))
        if coef.has(sym.zoo, sym.oo, sym.nan, -sym.oo):
            result.indeterminate.append(name)
            continue
        ps = tuple(sorted(s.name for s in coef.free_symbols if s in params))
        if ps:
            result.stiff_rates.append(StiffRate(name, ps, coef))
    return result


def rate_cpp(rate, cpp_name):
    """C++ expression for |rate.expr|; cpp_name maps a parameter name to C++."""
    sub = {s: sym.Symbol(cpp_name(s.name)) for s in rate.expr.free_symbols}
    return f"std::fabs({sym.ccode(rate.expr.xreplace(sub))})"
