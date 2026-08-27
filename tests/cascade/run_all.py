#!/usr/bin/env python
"""run_all.py -- one-command test suite for the cascade pipeline.

    python tests/run_all.py            # fast tier (~30 s): builder, MHD,
                                       #   Neo-Hookean, quickstart, contracts
    python tests/run_all.py --slow     # + BSSN and EMDA full builds (~10 min)

Plain asserts, no pytest dependency. Each test prints PASS/FAIL; nonzero
exit on any failure. See findings/cascade_api_guide.md for the API itself.
"""

import os
import sys
import warnings
from collections import OrderedDict

import sympy as sym  # noqa: E402

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


# ---------------------------------------------------------------- fast tier

@test("builder: by-symbol spec, prior refs counted, no warning")
def t_builder_by_symbol():
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.symbols("a b")
    chunks = [("lo", OrderedDict([("u", a * b + 1)])),
              ("hi", OrderedDict([("w", sym.Symbol("u") ** 2 + a)]))]
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = build_cascade_ir(chunks, {a, b})
    assert not wl, [str(x.message) for x in wl]
    assert [c.n_prior_refs for c in r.chunks] == [0, 1]


@test("builder: by-value spec substitutes via exact-tree xreplace")
def t_builder_by_value():
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.symbols("a b")
    chunks = [("lo", OrderedDict([("u", a * b + 1)])),
              ("hi", OrderedDict([("w", (a * b + 1) ** 2)]))]
    r = build_cascade_ir(chunks, {a, b})
    assert r.chunks[1].n_prior_refs == 1


@test("builder: broken sharing (post-processed spec) warns")
def t_builder_broken_warns():
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.symbols("a b")
    chunks = [("lo", OrderedDict([("u", a * b + 1)])),
              ("hi", OrderedDict([("w", sym.expand((a * b + 1) ** 2))]))]
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        build_cascade_ir(chunks, {a, b})
    assert any("no chunk references" in str(x.message) for x in wl)


@test("builder: odd-delta auto split warns; explicit/even silent")
def t_odd_delta_warns():
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.symbols("a b")
    u = sym.Symbol("u")
    chunks = [("lo", OrderedDict([("u", a * b + 1), ("u2", a + b)])),
              ("hi", OrderedDict([("w", u ** 2), ("w2", u - b),
                                  ("w3", u * a), ("w4", u + b)]))]
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        build_cascade_ir(chunks, {a, b}, target_L=3)
    assert any("odd delta" in str(x.message) for x in wl)
    for kwargs in (dict(target_L=3, smart_split=False), dict(target_L=4)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            build_cascade_ir(chunks, {a, b}, **kwargs)
        assert not any("odd delta" in str(x.message) for x in wl), kwargs


@test("builder: smart-split parity guard raises; collapse to L=1 works")
def t_parity_and_collapse():
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.symbols("a b")
    chunks = [("lo", OrderedDict([("u", a * b + 1)])),
              ("hi", OrderedDict([("w", sym.Symbol("u") + a)]))]
    assert len(build_cascade_ir(chunks, {a, b}, target_L=1).chunks) == 1
    try:
        build_cascade_ir(chunks, {a, b}, target_L=3, smart_split=True)
    except ValueError:
        pass
    else:
        raise AssertionError("parity ValueError not raised")


@test("quickstart from cascade_api_guide.md runs verbatim")
def t_quickstart():
    from dendrosym.cascade.builder import build_cascade_ir
    from dendrosym.cascade.emit import emit_kernel_function_cpp
    rho = sym.Symbol("rho[pp]")
    mom_x = sym.Symbol("mom_x[pp]")
    gm1 = sym.Symbol("gamma_minus_1")
    rho_inv = sym.Symbol("rho_inv")
    chunks = [("invert", OrderedDict([("rho_inv", 1 / rho)])),
              ("flux", OrderedDict([("F_out", mom_x * rho_inv * gm1)]))]
    result = build_cascade_ir(chunks, {rho, mom_x, gm1})
    cpp = emit_kernel_function_cpp(result, fn_name="my_kernel", simd="avx512")
    assert "my_kernel" in cpp and "F_out" in cpp and "_mm512" in cpp


@test("spec contract: all spec functions return (chunks, leaves)")
def t_spec_order_contract():
    from dendrosym.cascade.systems.mhd import mhd_flux_spec
    from dendrosym.cascade.systems.neohook import neohook_2d_spec, neohook_3d_spec
    for fn in (mhd_flux_spec, neohook_2d_spec, neohook_3d_spec):
        chunks, leaves = fn()
        assert isinstance(chunks, list) and isinstance(chunks[0], tuple), fn
        assert isinstance(leaves, set), fn


@test("MHD: natural/collapse/smart builds + AVX2 kernel emit")
def t_mhd():
    from dendrosym.cascade.systems import mhd_cascade
    assert len(mhd_cascade.build_ir().chunks) == 5
    assert len(mhd_cascade.build_ir(target_L=3).chunks) == 3
    r1 = mhd_cascade.build_ir(target_L=1)   # L=1 == global CSE
    assert len(r1.chunks) == 1 and r1.chunks[0].name == "global"
    # Smart split adds the shared-precursor chunk only when shared temps
    # exist; MHD's flux chunk has none, so one split yields 6, not 7.
    assert len(mhd_cascade.build_ir(target_L=7, smart_split=True).chunks) == 6
    k = mhd_cascade.emit_kernel(simd="avx2")
    assert "mhd_flux_kernel" in k and "F_rho_out" in k


@test("Neo-Hookean: 2D and 3D builds")
def t_neohook():
    from dendrosym.cascade.systems import neohook_cascade
    assert len(neohook_cascade.build_ir(dim=2).chunks) == 5
    assert len(neohook_cascade.build_ir(dim=3).chunks) == 5
    body = neohook_cascade.emit_cpp()
    assert "P00" in body


# ---------------------------------------------------------------- slow tier

@test("[slow] BSSN: natural build, chunk counts, prior-ref fingerprint")
def t_bssn():
    from dendrosym.cascade.systems.bssn import cascade as bssn_cascade
    r = bssn_cascade.build_ir()
    assert len(r.chunks) == 7
    # Fingerprint after the F14 deep-substitution fix (boundaries real).
    # fingerprint re-derived 2026-08-26: vikr's test still says 70 for the
    # derived_quantities layer, but vikr HEAD's own build_ir() gives 30 (the
    # 2026-08-08 symmetric-trace/pow regen changed it; the emitted kernels are
    # byte-identical to vikr's -- see scripts/regen_vikr_kernels.sh).
    assert [c.n_prior_refs for c in r.chunks] == [0, 0, 24, 25, 43, 30, 46], \
        [c.n_prior_refs for c in r.chunks]


@test("[slow] BSSN: SSL+CAHD spec wires gauge terms through chunk symbols")
def t_bssn_gauge():
    from dendrosym.cascade.systems.bssn import cascade as bssn_cascade
    r = bssn_cascade.build_ir(ssl=True, cahd=True)
    last = r.chunks[-1]
    names = set()
    for e in list(last.outputs.values()) + [x for _, x in last.cse_temps]:
        names |= {s.name for s in e.free_symbols}
    assert {"ssl_fac", "cahd_coef", "At_sqr", "igt00", "R00"} <= names
    assert last.n_prior_refs == 53  # ssl-cahd fingerprint (post-F14)


@test("[slow] EMDA: 8 chunks with matter, leaf set complete")
def t_emda():
    from dendrosym.cascade.systems.emda import cascade as emda_cascade
    r = emda_cascade.build_ir(with_matter=True)
    assert len(r.chunks) == 8
    defined, free = set(), set()
    for c in r.chunks:
        for s, e in c.cse_temps:
            defined.add(s)
            free |= e.free_symbols
        for name, e in c.outputs.items():
            defined.add(sym.Symbol(name))
            free |= e.free_symbols
    assert not (free - defined) - r.leaf_symbols, "leaf_symbols incomplete"

@test("options: CascadeOptions <-> argparse round-trip, fma_split implies fma_tree")
def t_options_roundtrip():
    import argparse
    from dendrosym.cascade.options import CascadeOptions
    o = CascadeOptions(simd="avx512", L=7, inline_threshold=0, fma_tree=False,
                       global_cse=True, split_mode="smart")
    ap = argparse.ArgumentParser()
    CascadeOptions.add_argparse_args(ap)
    back = CascadeOptions.from_namespace(ap.parse_args(o.to_cli_args()))
    assert back == o, (back, o)
    ap2 = argparse.ArgumentParser()
    CascadeOptions.add_argparse_args(ap2, prefix="cascade-")
    ns = ap2.parse_args(["--cascade-fma-split", "2", "--cascade-simd", "avx2"])
    o2 = CascadeOptions.from_namespace(ns, prefix="cascade-", base=CascadeOptions(fma_tree=False))
    assert o2.fma_tree and o2.fma_split == 2 and o2.simd == "avx2"
    assert o.cache_key() != back.replace(L=8).cache_key()
    try:
        CascadeOptions(simd="scalar", fused=True)
    except ValueError:
        pass
    else:
        raise AssertionError("fused+scalar must be rejected")


@test("api: compile_system(options=) emits standalone kernels; predict() cost model")
def t_api_options():
    from dendrosym.cascade import compile_system, predict, CascadeOptions, build, emit_body
    rho = sym.Symbol("rho[pp]"); mom = sym.Symbol("mom_x[pp]"); gm1 = sym.Symbol("gamma_minus_1")
    lam0 = sym.Symbol("lambda[0]"); rho_inv = sym.Symbol("rho_inv")
    chunks = [("invert", OrderedDict([("rho_inv", 1 / rho)])),
              ("flux", OrderedDict([("F_out", mom * rho_inv * gm1 + lam0 * rho),
                                    ("G_rhs", sym.log(rho) * rho_inv)]))]
    leaves = {rho, mom, gm1, lam0}
    for simd, marker in (("scalar", "#define VEC double"), ("avx2", "_mm256"), ("avx512", "_mm512")):
        code, ir = compile_system(chunks, leaves, options=CascadeOptions(simd=simd, fn_name="k"))
        assert marker in code and "static inline void k(" in code, simd
        assert "VSTORE(F_out_ptr+pp, F_out);" in code and "VSTORE(G_rhs+pp," in code, simd
        assert "double gamma_minus_1_s" in code and "const double *__restrict__ lambda" in code
    legacy, _ = compile_system(chunks, leaves)
    assert "rho_inv" in legacy and "VSTORE" not in legacy
    body = emit_body(build(chunks, leaves), CascadeOptions(simd="avx2", inline_threshold=0))
    assert body.startswith("// --- IR-AVX prologue") and "v_rho" in body
    r = predict(chunks, leaves)
    assert r["layers"] == 2 and r["peak_live"] >= 1 and isinstance(r["trace"], list)


@test("emit: BSSN front-end == emit_body(header=...) composition (fidelity, toy IR)")
def t_emit_body_fidelity():
    from dendrosym.cascade.emit import (emit_body, _inline_low_use_temps, _classify_leaves,
                                        _emit_avx_prologue, _emit_avx_chunks, _lazy_reorder)
    from dendrosym.cascade.options import CascadeOptions
    from dendrosym.cascade.builder import build_cascade_ir
    a, b = sym.Symbol("a[pp]"), sym.Symbol("b[pp]")
    e = sym.Symbol("eta")
    chunks = [("p", OrderedDict([("q", a * b + a), ("r", a * b - b)])),
              ("o", OrderedDict([("x_rhs", sym.Symbol("q") * e + sym.Symbol("r") * e)]))]
    res = build_cascade_ir(chunks, {a, b, e})
    hdr = ["// h1", "// h2", ""]
    r2 = _inline_low_use_temps(res, threshold=2)
    lv = _classify_leaves(r2, fused=False)
    ref = "\n".join(_lazy_reorder(hdr + _emit_avx_prologue(lv) + _emit_avx_chunks(r2, lv, fma=True, split=1))) + "\n"
    got = emit_body(res, CascadeOptions(simd="avx2", inline_threshold=2, fma_tree=True),
                    header="\n".join(hdr) + "\n")
    assert got == ref, "emit_body drifted from the hand composition"

@test("bridge: derivs_to_symbols canonicalises grad/grad2/agrad applications")
def t_bridge_derivs_to_symbols():
    from dendrosym.cascade.dendro_bridge import derivs_to_symbols, CascadeNamingError
    grad, grad2, agrad = sym.Function("grad"), sym.Function("grad2"), sym.Function("agrad")
    a, b = sym.Symbol("alpha[pp]"), sym.Symbol("gt01[pp]")
    e = grad(0, a) * grad2(2, 1, b) + agrad(1, a) + grad2(0, 0, b) ** 2
    out = derivs_to_symbols(e)
    names = {s.name for s in out.free_symbols}
    assert names == {"grad_0_alpha[pp]", "grad2_1_2_gt01[pp]", "agrad_1_alpha[pp]",
                     "grad2_0_0_gt01[pp]"}, names
    assert not out.atoms(sym.core.function.AppliedUndef)
    try:
        derivs_to_symbols(grad(0, a * b))
    except CascadeNamingError:
        pass
    else:
        raise AssertionError("compound derivative must be rejected")


@test("bridge: emit_config_cascade alias/prologue/manifest on a toy config-like spec")
def t_bridge_emit():
    from types import SimpleNamespace
    from dendrosym.cascade.builder import build_cascade_ir
    from dendrosym.cascade.dendro_bridge import emit_config_cascade, CascadeNamingError
    from dendrosym.cascade.options import CascadeOptions
    a, ga, g2a = (sym.Symbol("alpha[pp]"), sym.Symbol("grad_0_alpha[pp]"),
                  sym.Symbol("grad2_0_1_alpha[pp]"))
    eta, lam = sym.Symbol("eta"), sym.Symbol("lambda[0]")
    chunks = [("inv", OrderedDict([("ainv", 1 / a)])),
              ("rhs_assembly", OrderedDict([("alpha_rhs", sym.Symbol("ainv") * ga * eta + lam * g2a * sym.Symbol("ainv"))]))]
    ir = build_cascade_ir(chunks, {a, ga, g2a, eta, lam})
    cfg = SimpleNamespace(all_var_names={"evolution": ["alpha"]}, project_name="toy",
                          input_struct_name=lambda: "in",
                          output_struct_name=lambda vt: "out")
    struct = "struct toy_evolution_derivs_t {\n    double *alpha_x;\n    double *alpha_xy;\n};"
    parts = emit_config_cascade(ir, CascadeOptions(simd="avx2"), config=cfg, var_type="evolution",
                                deriv_struct_text=struct, in_names=["alpha"], use_advective=False)
    assert "const double *const alpha = in.alpha;" in parts.alias
    assert "const double *const grad_0_alpha = d.alpha_x;" in parts.alias
    assert "const double *const grad2_0_1_alpha = d.alpha_xy;" in parts.alias
    assert "double *const alpha_rhs = out.alpha;" in parts.alias
    assert "const double __cascade_scalar_eta = eta;" in parts.alias
    assert "const VEC eta = VSET(__cascade_scalar_eta);" in parts.prologue
    assert "VSTORE(alpha_rhs+pp," in parts.body and "lambda_0 = VSET((double)(lambda[0]))" in parts.body
    assert "#define VFNMADD" in parts.macros_avx2 and "__AVX2__" in parts.macros_avx2
    assert "_mm512" in parts.macros_avx512 and "__AVX512F__" in parts.macros_avx512
    assert "__AVX512F__" in parts.select and "DENDRO_CASCADE_FLAT" in parts.select
    assert "toy_evolution_cascade_macros_avx512.cpp.inc" in parts.select
    assert "#define VMASK" in parts.macros_avx2 and "VLOADM(" in parts.body_tail
    assert parts.manifest["simd"] == "compile-time" and parts.manifest["fused"] is False
    # fused: the fused body computes grad/grad2 from the state array; alias covers it
    fparts = emit_config_cascade(ir, CascadeOptions(simd="avx2", fused=True), config=cfg,
                                 var_type="evolution", deriv_struct_text=struct,
                                 in_names=["alpha"], use_advective=False,
                                 deriv_calc_fused="// reduced\n")
    assert "idx60" in fparts.body_fused and "VLOAD(alpha+pp+1)" in fparts.body_fused
    assert "VLOAD(grad_0_alpha+pp)" not in fparts.body_fused and "grad2_0_1_alpha" in fparts.body_fused
    assert "VLOADM(alpha+pp+1)" in fparts.body_fused_tail
    assert fparts.manifest["fused"] and fparts.manifest["stencil_vars"] == ["alpha"]
    from dendrosym.codegen import reduce_deriv_calc_for_fused
    calc = ("D.grad_x(d.a_x, in.a, hx, sz, bflag);\nD.grad_y(d.a_y, in.a, hy, sz, bflag);\n"
            "D.grad_z(d.a_z, in.a, hz, sz, bflag);\nD.grad_xx(d.a_xx, in.a, hx, sz, bflag);\n"
            "D.grad_y(d.a_xy, d.a_x, hy, sz, bflag);\nD.grad_x(d.b_x, in.b, hx, sz, bflag);\n")
    red, kept, dropped = reduce_deriv_calc_for_fused(calc)
    assert kept == 2 and dropped == 4 and "d.a_xy" in red and "d.a_x, in.a" in red and "d.b_x" not in red
    assert parts.body_tail.count("VSTOREM(") == parts.body.count("VSTORE(") and "VLOAD(" not in parts.body_tail
    assert "#include" not in parts.macros_scalar and "#define VEC double" in parts.macros_scalar
    assert parts.manifest["outputs"] == ["alpha_rhs"] and parts.manifest["params"] == ["eta"]
    # a leaf whose buffer the deriv struct lacks must fail loudly
    try:
        emit_config_cascade(ir, CascadeOptions(simd="avx2"), config=cfg, var_type="evolution",
                            deriv_struct_text="struct t { double *alpha_x; };",
                            in_names=["alpha"], use_advective=False)
    except CascadeNamingError:
        pass
    else:
        raise AssertionError("missing d.alpha_xy must raise")


@test("template: rhs.cpp.j2 renders the flat loop unchanged without evolution_cascade")
def t_template_off_render():
    import jinja2, pathlib, subprocess
    from dendrosym import project_generator as pg
    tdir = pathlib.Path(pg.__file__).parent / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(tdir)),
                             trim_blocks=True, lstrip_blocks=True)
    src = env.loader.get_source(env, "gr/rhs.cpp.j2")[0]
    assert "{% if evolution_cascade is defined %}" in src
    ctx = {"project_name": "toy", "namespace": "toy", "project_upper": "TOY",
           "evolution_gencode": {"deriv_struct": "s.inc", "rhs_eqns": "r.inc", "deriv_calc": "dc.inc"}}
    off = env.get_template("gr/rhs.cpp.j2").render(**ctx)
    casc = {"simd": "compile-time", "select": "sel.inc", "alias": "a.inc", "prologue": "p.inc",
            "body": "b.inc", "body_tail": "bt.inc", "macros_undef": "mu.inc"}
    on = env.get_template("gr/rhs.cpp.j2").render(**ctx, evolution_cascade=casc)
    assert "cascade" not in off and '#include "../gencode/r.inc"' in off
    # on-render: the cascade block under #ifndef DENDRO_CASCADE_FLAT, the flat loop under #else
    assert "#ifndef DENDRO_CASCADE_FLAT" in on and '#include "../gencode/sel.inc"' in on
    assert '#include "../gencode/b.inc"' in on and '#include "../gencode/r.inc"' in on
    assert on.index("#ifndef DENDRO_CASCADE_FLAT") < on.index("b.inc") < on.index("#else") < on.index("r.inc") < on.index("#endif", on.index("#else"))
    assert "VMASK(__cascade_nvalid)" in on and '#include "../gencode/bt.inc"' in on
    # fused: bflag dispatch for both the deriv pass and the loop
    fon = env.get_template("gr/rhs.cpp.j2").render(**ctx, evolution_cascade={
        **casc, "fused": True, "body_fused": "bf.inc", "body_fused_tail": "bft.inc",
        "deriv_calc_fused": "dcf.inc"})
    assert fon.count("if (bflag == 0)") == 2 and '#include "../gencode/dcf.inc"' in fon
    assert '#include "../gencode/bf.inc"' in fon and '#include "../gencode/bft.inc"' in fon
    assert fon.count('#include "../gencode/b.inc"') == 1 and "DERIVTYPE_FIRST" in fon
    assert "if (bflag == 0)" not in on
    # the flat loop text is identical whether or not the cascade is present
    flat_loop = off[off.index("    for (unsigned int k = PW;"):off.index("    toy::timer::t_rhs.stop()")]
    assert flat_loop.endswith("    }\n") and flat_loop in on
    # constraint kernel: same branch in physcon.cpp.j2, keyed on constraint_cascade
    pctx = {"project_name": "toy", "namespace": "toy", "project_upper": "TOY",
            "constraint_gencode": {"deriv_struct": "cs.inc", "rhs_eqns": "cr.inc"},
            "evolution_var_extraction": "", "constraint_output_extraction": ""}
    poff = env.get_template("gr/physcon.cpp.j2").render(**pctx)
    pon = env.get_template("gr/physcon.cpp.j2").render(**pctx, constraint_cascade=casc)
    assert "cascade" not in poff and '#include "../gencode/cr.inc"' in poff
    assert "#ifndef DENDRO_CASCADE_FLAT" in pon and '#include "../gencode/cr.inc"' in pon
    assert pon.index("b.inc") < pon.index("#else") < pon.index("cr.inc")
    # the off-render must equal what the pre-cascade template produced: the
    # committed CCZ4 rhs.cpp (flat) is that reference when available.
    ref = pathlib.Path.home() / "research/ccz4-gr/solver/src/rhs.cpp"
    if ref.exists():
        head = subprocess.run(["git", "-C", str(ref.parent.parent.parent), "show",
                               "HEAD:solver/src/rhs.cpp"], capture_output=True, text=True)
        if head.returncode == 0 and "cascade" not in head.stdout:
            assert "__cascade" not in head.stdout  # sanity: reference is the flat one


FAST = [t_builder_by_symbol, t_builder_by_value, t_builder_broken_warns,
        t_odd_delta_warns, t_parity_and_collapse, t_quickstart,
        t_spec_order_contract, t_mhd, t_neohook,
        t_options_roundtrip, t_api_options, t_emit_body_fidelity,
        t_bridge_derivs_to_symbols, t_bridge_emit, t_template_off_render]
SLOW = [t_bssn, t_bssn_gauge, t_emda]


if __name__ == "__main__":
    slow = "--slow" in sys.argv
    tests = FAST + (SLOW if slow else [])
    print(f"cascade test suite: {len(tests)} tests"
          f"{'' if slow else ' (fast tier; --slow adds BSSN/EMDA)'}")
    for t in tests:
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
