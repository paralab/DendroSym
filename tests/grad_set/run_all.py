#!/usr/bin/env python
"""run_all.py -- tests for the planned-derivative-set emitter.

    python tests/grad_set/run_all.py

Plain asserts, no pytest dependency (same shape as tests/cascade/run_all.py).
Covers what `emit_deriv_calc_grad_set` must never get wrong: the member each
call maps to, completeness against the per-call list it replaces, the masks,
and the two fallbacks (staged chains, advective calls).
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dendrosym import codegen  # noqa: E402

FAILURES = []


def test(name):
    def deco(fn):
        def run():
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e!r}")
                FAILURES.append(name)
        run.test_name = name
        return run
    return deco


def make_calc(first_only, full, obj="SOLVER_DERIVS"):
    """A per-call deriv list in the shape project_generator hands the emitter."""
    L = []
    for v in first_only + full:
        L.append(f"{obj}.grad_x(d.{v}_x, in.{v}, hx, sz, bflag);")
        if v in full:
            L.append(f"{obj}.grad_xx(d.{v}_xx, in.{v}, hx, sz, bflag);")
        L.append(f"{obj}.grad_y(d.{v}_y, in.{v}, hy, sz, bflag);")
        if v in full:
            L.append(f"{obj}.grad_yy(d.{v}_yy, in.{v}, hy, sz, bflag);")
        L.append(f"{obj}.grad_z(d.{v}_z, in.{v}, hz, sz, bflag);")
        if v in full:
            L.append(f"{obj}.grad_zz(d.{v}_zz, in.{v}, hz, sz, bflag);")
            L.append(f"{obj}.grad_y(d.{v}_xy, d.{v}_x, hy, sz, bflag);")
            L.append(f"{obj}.grad_z(d.{v}_xz, d.{v}_x, hz, sz, bflag);")
            L.append(f"{obj}.grad_z(d.{v}_yz, d.{v}_y, hz, sz, bflag);")
    return "\n".join(L) + "\n"


CCZ4_FIRST = ["At00", "At01", "Gammahat0", "Khat", "Theta", "gaugeB0"]
CCZ4_FULL = ["alpha", "beta0", "chi", "gt00", "gt01"]


def set_rows(emitted):
    """[(mask_expr, [row buffers, ...], [sources, ...])] parsed back out."""
    rows = []
    for blk in re.findall(r"\{\n(.*?)\n\}", emitted, re.S):
        members = [
            [t.strip() for t in m.group(1).split(",")]
            for m in re.finditer(r"^\s*\{ (.*) \},$", blk, re.M)
        ]
        srcs = re.search(r"__du\d+\[\] = \{ (.*) \};", blk).group(1).split(", ")
        mask = re.search(r"grad_set(?:_batch)?\(.*?, ((?:__DD::DM_\w+(?: \| )?)+),",
                         blk, re.S).group(1)
        rows.append((mask, members, srcs))
    return rows


@test("every buffer of the per-call list lands in exactly one set slot")
def t_complete():
    calc = make_calc(CCZ4_FIRST, CCZ4_FULL)
    want = {m.group(1) for m in re.finditer(r"grad_\w+\((d\.\w+),", calc)}
    got = []
    for _mask, members, _srcs in set_rows(codegen.emit_deriv_calc_grad_set(calc)):
        got += [t for row in members for t in row if t != "nullptr"]
    assert len(got) == len(set(got)), "a buffer was emitted twice"
    assert set(got) == want, want ^ set(got)


@test("members land in the DerivSet slot the buffer name implies")
def t_member_positions():
    calc = make_calc(CCZ4_FIRST, CCZ4_FULL)
    cols = codegen._DERIV_SET_MEMBERS
    for _mask, members, srcs in set_rows(codegen.emit_deriv_calc_grad_set(calc)):
        for row, src in zip(members, srcs):
            var = src.split(".", 1)[1]
            for i, tok in enumerate(row):
                if tok != "nullptr":
                    assert tok == f"d.{var}_{cols[i]}", (tok, cols[i])


@test("masks collapse to the engine's aliases; sources group by mask")
def t_masks():
    calc = make_calc(CCZ4_FIRST, CCZ4_FULL)
    rows = set_rows(codegen.emit_deriv_calc_grad_set(calc))
    assert len(rows) == 2, len(rows)
    assert rows[0][0] == "__DD::DM_FIRST"
    assert rows[0][2] == [f"in.{v}" for v in CCZ4_FIRST]
    assert rows[1][0] == "__DD::DM_ALL"
    assert rows[1][2] == [f"in.{v}" for v in CCZ4_FULL]


@test("underscored variable names are not mis-split")
def t_underscored_names():
    calc = make_calc(["At_mat00"], ["gt_mat01"])
    rows = set_rows(codegen.emit_deriv_calc_grad_set(calc))
    assert rows[0][2] == ["in.At_mat00"] and rows[1][2] == ["in.gt_mat01"]
    assert rows[1][1][0][6] == "d.gt_mat01_xy", rows[1][1][0]


@test("fused-reduced list -> one set of {x, y, mixed}")
def t_fused():
    calc = make_calc(CCZ4_FIRST, CCZ4_FULL)
    reduced = codegen.reduce_deriv_calc_for_fused(calc)[0]
    rows = set_rows(codegen.emit_deriv_calc_grad_set(reduced))
    assert len(rows) == 1, len(rows)
    assert rows[0][0] == "__DD::DM_MIXED | __DD::DM_X | __DD::DM_Y", rows[0][0]
    assert rows[0][2] == [f"in.{v}" for v in CCZ4_FULL]


@test("single-variable group emits grad_set, not grad_set_batch")
def t_single():
    out = codegen.emit_deriv_calc_grad_set(make_calc(["chi"], []))
    assert ".grad_set(__ds0[0], __du0[0]," in out, out
    assert "grad_set_batch" not in out


@test("advective calls pass through verbatim, after the sets")
def t_advective():
    calc = (make_calc(["chi"], [])
            + "adv_deriv_x(d.adv_chi_x, in.chi, hx, sz, beta0, bflag);\n")
    out = codegen.emit_deriv_calc_grad_set(calc)
    assert out.rstrip().endswith(
        "adv_deriv_x(d.adv_chi_x, in.chi, hx, sz, beta0, bflag);")


@test("unresolvable chain (staged buffer) -> None, caller falls back")
def t_staged_bails():
    assert codegen.emit_deriv_calc_grad_set(
        "SOLVER_DERIVS.grad_y(d.foo_xy, d.staged_7, hy, sz, bflag);\n") is None
    # a mixed second chained off a z-first has no DerivSet member
    assert codegen.emit_deriv_calc_grad_set(
        "SOLVER_DERIVS.grad_z(d.a_z, in.a, hz, sz, bflag);\n"
        "SOLVER_DERIVS.grad_y(d.a_yz, d.a_z, hy, sz, bflag);\n") is None
    # an unrecognized line that is not an advective call
    assert codegen.emit_deriv_calc_grad_set(
        "some_other_call(d.a_x, in.a);\n") is None


@test("emission is deterministic for a fixed input list")
def t_deterministic():
    calc = make_calc(CCZ4_FIRST, CCZ4_FULL)
    a = codegen.emit_deriv_calc_grad_set(calc)
    assert a == codegen.emit_deriv_calc_grad_set(calc)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    print(f"grad_set emitter: {len(tests)} tests")
    for t in tests:
        t()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
