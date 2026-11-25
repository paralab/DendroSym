#!/usr/bin/env python
"""run_all.py -- LTS codegen checks.

    python tests/lts/run_all.py            # fast checks + BSSN known answer
    python tests/lts/run_all.py --ccz4     # also audit CCZ4 from ../ccz4-gr
    python tests/lts/run_all.py --fast     # skip the BSSN equations (~1 min)

Plain asserts, no pytest. Each check asserts both the finding and its absence on
a clean input, so it fails if the piece it covers stops detecting (or starts
over-reporting).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)

import sympy as sym  # noqa: E402

import jinja2  # noqa: E402

from dendrosym import lts_audit, template_gate  # noqa: E402
from dendrosym.lts_names import LTS_NAMES  # noqa: E402
from dendrosym.project_generator import _TEMPLATES_DIR, build_template_map  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAILURES.append(name)
        print(f"FAIL  {name}\n      {exc}")
    else:
        print(f"PASS  {name}")


class _StubConfig:
    """Just enough of a DendroConfiguration for the audit."""

    idx_str = "[pp]"

    def __init__(self, rhs, constraints=()):
        self._rhs = rhs
        self.all_var_names = {"constraint": list(constraints)}

    def _extract_rhs_expressions(self, var_type, append_rhs_to_var=True):
        names = list(self._rhs)
        return [self._rhs[n] for n in names], names, [], [], len(names)


def _sym(name):
    return sym.Symbol(f"{name}[pp]")


grad = sym.Function("grad")
k, dt = sym.Symbol("k"), sym.Symbol("dt")
u, v, w, ham = _sym("u"), _sym("v"), _sym("w"), _sym("ham")


def test_stiff_rate_found():
    a = lts_audit.audit(_StubConfig({"u": -k * u + grad(0, v), "v": grad(1, u)}))
    assert [r.var for r in a.stiff_rates] == ["u"], a.summary()
    assert a.stiff_rates[0].expr == -k, a.stiff_rates[0].expr
    assert a.rate_params == ["k"]


def test_derivative_terms_are_not_rates():
    a = lts_audit.audit(_StubConfig({"u": k * grad(0, u), "v": u * k}))
    assert not a.stiff_rates, a.summary()


def test_constraint_read_found():
    a = lts_audit.audit(
        _StubConfig({"u": -k * u + ham, "v": grad(0, ham), "w": u}, ["ham"])
    )
    assert sorted(a.constraint_reads) == ["u", "v"], a.constraint_reads
    assert not a.clean
    clean = lts_audit.audit(_StubConfig({"u": -k * u + v, "w": u}, ["ham"]))
    assert clean.clean and not clean.constraint_reads, clean.summary()


def test_dt_term_found():
    a = lts_audit.audit(_StubConfig({"u": dt * u + v, "v": u}))
    assert sorted(a.dt_terms) == ["u"], a.dt_terms
    assert not a.clean
    assert lts_audit.audit(_StubConfig({"u": k * u})).clean


def test_rate_cpp_names_and_magnitude():
    a = lts_audit.audit(_StubConfig({"u": -2 * k * u}))
    code = lts_audit.rate_cpp(a.stiff_rates[0], lambda n: f"NS::P_{n.upper()}")
    assert code == "std::fabs(-2*NS::P_K)", code


def test_bssn_eta_is_the_known_stiff_rate():
    sys.path.insert(0, os.path.join(ROOT, "tests", "jax_emit"))
    import bssn_eqns

    a = lts_audit.audit(bssn_eqns.dendroConfigs)
    assert [r.var for r in a.stiff_rates] == ["B0", "B1", "B2"], a.summary()
    assert a.rate_params == ["eta"], a.summary()
    assert all(r.expr == -sym.Symbol("eta") for r in a.stiff_rates)
    assert a.clean, a.summary()


def test_ccz4_audit():
    sys.path.insert(0, os.path.join(ROOT, "..", "ccz4-gr"))
    import ccz4

    a = lts_audit.audit(ccz4.dendroConfigs)
    assert a.clean, a.summary()
    assert not a.indeterminate, a.indeterminate
    assert {"eta", "kappa[0]"} <= set(a.rate_params), a.rate_params


def _render(flags, only=None):
    """Render the mapped C++ templates for one stub context; {output path: text}."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=template_gate._StubUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    ctx = dict(
        project_name="demo",
        project_upper="DEMO",
        namespace="demo",
        evolution_num_vars=3,
        physics_params=[],
    )
    ctx.update(flags)
    out = {}
    for rel, tmpl in build_template_map(ctx).items():
        if not rel.endswith((".h", ".cpp", ".toml")):
            continue
        if only and not any(o in rel for o in only):
            continue
        out[rel] = env.get_template(tmpl).render(**ctx)
    return out


_LTS_ON = dict(
    enable_lts=True,
    lts=LTS_NAMES,
    lts_rates=["std::fabs(static_cast<double>(demo::DEMO_ETA))"],
    lts_value_boundary_ids=[2],
    boundary_modes=[{"name": "a", "id": 0, "summary": "", "code": ""}],
)
_LTS_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:lts|LTS)(?![a-z])|rhs_blk|block_view|enforce_blk|"
    r"slot_override|restored_checkpt|restoredSlot|time_stepper|TIME_STEPPER|"
    r"ExplicitNUTS|get_async_batch_sz|EVAR_DUMP|dump_evar_bin")


def test_flag_off_emits_nothing_lts():
    for extra in (dict(), dict(enable_bh_tracking=True, enable_ah=True,
                               enable_gw_extraction=True, enable_profiling=True)):
        for path, text in _render(extra).items():
            hit = _LTS_TOKEN.search(text)
            assert not hit, f"{path} mentions {hit.group(0)!r} with enable_lts off"


def test_flag_on_emits_the_contract():
    out = _render(_LTS_ON)
    ctx_h = out["solver/include/demoCtx.h"]
    ctx_cpp = out["solver/src/demoCtx.cpp"]
    for hook in ("rhs_blk", "pre_stage_blk", "post_stage_blk", "pre_timestep_blk",
                 "post_timestep_blk", "on_mesh_changed", "compute_lts_ts_offset"):
        assert hook in ctx_h, f"{hook} not declared"
        assert re.search(rf"DEMOCtx::{hook}\(", ctx_cpp), f"{hook} not defined"
    assert "unsigned int get_async_batch_sz() { return DEMO_ASYNC_COMM_K; }" in ctx_h, \
        "the CRTP base forwards get_async_batch_sz to the leaf: without it the stepper recurses forever"
    assert "ot::Mesh *get_mesh() const {" in ctx_h, "the stepper calls get_mesh() on a const ctx"
    assert "lts_thread_safe" not in ctx_cpp.replace("Not lts_thread_safe", "")
    assert "std::fabs(static_cast<double>(demo::DEMO_ETA))" in ctx_cpp, \
        "audit rate missing from compute_lts_ts_offset"
    assert "DEMO_LTS_RATE_C / rate" in ctx_cpp
    assert "meta[\"time_stepper\"]" in ctx_cpp and "refusing to restore" in ctx_cpp
    driver = out["solver/include/demo_lts_driver.h"]
    for piece in ("ExplicitNUTS", "remesh_if_needed", "checkpoint(", "write_carried_g",
                  "read_carried_g", "clk.resume", "checkpoint_slot", "LtsOptions"):
        assert piece in driver, f"driver lacks {piece}"
    assert "dump_evar_bin" in driver and "DENDRO_EVAR_DUMP" in driver
    assert "demo::dump_evar_bin(" in out["solver/demo_main.cpp"], "global path does not dump"
    assert "demo_lts_driver.h" in out["solver/demo_main.cpp"]
    main_cpp = out["solver/demo_main.cpp"]
    assert re.search(r"run_demo_lts\(ctx, comm\);\s*(//[^\n]*\n\s*)*mesh = ctx->get_mesh\(\);\s*delete ctx;\s*delete mesh;", main_cpp), \
        "main deletes a stale mesh after the driver remeshed"
    assert 'DEMO_TIME_STEPPER == "lts"' in out["solver/demo_main.cpp"]


def test_flag_on_refuses_what_lts_cannot_run():
    par = _render(_LTS_ON)["solver/src/parameters.cpp"]
    assert "{RKType::RK3, RKType::RK4}" in par, "rk_type refusal missing or widened"
    assert "{2})" in par.replace(" ", "") or "{2}" in par.replace(" ", ""), \
        "value-form boundary refusal missing"
    assert "value-form boundary modes" in par
    assert "[lts] partitioning_method" in par and "{0}" in par.replace(" ", "")
    bh = _render(dict(_LTS_ON, enable_bh_tracking=True))["solver/src/parameters.cpp"]
    assert "fixed_punctures" in bh and "Puncture tracking is not supported" in bh
    nobh = _render(_LTS_ON)["solver/src/parameters.cpp"]
    assert "fixed_punctures" not in nobh
    novalue = _render(dict(_LTS_ON, lts_value_boundary_ids=[]))["solver/src/parameters.cpp"]
    assert "value-form boundary modes" not in novalue


def test_names_come_from_the_table():
    renamed = dict(LTS_NAMES, stepper="RenamedStepper", enforced_nodes="renamed_nodes")
    out = _render(dict(_LTS_ON, lts=renamed))
    assert "RenamedStepper" in out["solver/include/demo_lts_driver.h"]
    assert "renamed_nodes" in out["solver/src/demoCtx.cpp"]
    assert "ExplicitNUTS" not in out["solver/include/demo_lts_driver.h"]


check("audit finds a stiff rate", test_stiff_rate_found)
check("audit ignores derivative terms", test_derivative_terms_are_not_rates)
check("audit finds constraint reads", test_constraint_read_found)
check("audit finds dt terms", test_dt_term_found)
check("rate_cpp prints names and magnitude", test_rate_cpp_names_and_magnitude)
check("enable_lts off: no LTS text in any template", test_flag_off_emits_nothing_lts)
check("enable_lts on: hooks, driver, rates, checkpoint tag", test_flag_on_emits_the_contract)
check("enable_lts on: refusals are emitted", test_flag_on_refuses_what_lts_cannot_run)
check("library names read from lts_names", test_names_come_from_the_table)
if "--fast" not in sys.argv:
    check("BSSN: eta in B is the stiff rate", test_bssn_eta_is_the_known_stiff_rate)
if "--ccz4" in sys.argv:
    check("CCZ4 audit: eta and kappa1 rates, no reads", test_ccz4_audit)

if FAILURES:
    print(f"\n{len(FAILURES)} failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("\nall LTS checks passed")
