"""cascade_codegen.py -- Generate C++ and CUDA code for the BSSN cascade RHS.

Translates the 7-layer quadratic cascade (as implemented in bench.cpp and
bssn_jax.py) into DendroGR-compatible C++ include files and CUDA kernels.

The generated code is a drop-in replacement for the CSE-based bssneqs_*.cpp
files, gated by a BSSN_USE_CASCADE* preprocessor flag.

Module split:
    cascade_common.py
        --- backend-agnostic infrastructure: SIMD macro headers, 6th-order
        centered stencils (scalar + SIMD), symmetric-index helpers, small
        code-emission utilities. Reusable for other PDE systems (EMDA, ...).
    cascade_codegen.py  (this file)
        --- BSSN-specific naming (gt/At/grad_*/grad2_*/agrad_*), layer
        emitters emit_L1..L7, state/deriv unpack, and the top-level
        generate_cascade_{cpu,cpu_unrolled,avx2,cuda} entry points + CLI.

Top-level generators:
    generate_cascade_cpu(config)          --- scalar tensor-loop
                                             (unified emitters + scalar macros)
    generate_cascade_cpu_unrolled(config) --- scalar fully-unrolled CSE-like
                                             (standalone; supports SSL, CAHD,
                                             Rochester gauge)
    generate_cascade_avx2(config)         --- AVX2 or AVX-512 (config["simd"])
                                             (unified emitters + AVX macros)
    generate_cascade_cuda(config)         --- CUDA __device__ kernel
                                             (unified emitters + scalar macros,
                                             since CUDA threads are scalar)

Limitation: SSL, CAHD, and Rochester gauge modifiers are currently only
supported by generate_cascade_cpu_unrolled(). Porting them to the unified
emitters would require vectorized sqrt / exp for SSL's exp(-t^2/sigma^2)
term (AVX has sqrt but exp needs SVML or a per-lane fallback).

CLI: python cascade_codegen.py --target {cpu,avx2,avx512,cuda,both}
                               [--unroll] [--fused-derivs] [--fuse-mixed]
                               [--ssl] [--cahd] [--gauge {standard,rochester}]
                               [--eta {const,func,RIT}] --output DIR
"""

from dendrosym.cascade.common import (
    sym,
    sym_label,
    stencil_grad as _stencil_grad,
    stencil_grad2_pure as _stencil_grad2_pure,
    stencil_grad2_mixed as _stencil_grad2_mixed,
    simd_stencil_1st as _avx_stencil_1st,
    simd_stencil_2nd_pure as _avx_stencil_2nd_pure,
    SCALAR_MACROS_HEADER as _SCALAR_MACROS_HEADER,
    AVX_MACROS_HEADER as _AVX_MACROS_HEADER,
    AVX512_MACROS_HEADER as _AVX512_MACROS_HEADER,
    _STENCIL_COEF_SETUP as _AVX_STENCIL_COEF_SETUP,
    indent as _indent,
    comment as _comment,
)


# ---------------------------------------------------------------------------
# DendroGR variable naming (BSSN-specific)
# ---------------------------------------------------------------------------

def _gt(i, j):
    return f"gt{sym(i,j)}[pp]"

def _At(i, j):
    return f"At{sym(i,j)}[pp]"

def _grad(d, var):
    return f"grad_{d}_{var}[pp]"

def _grad2(d1, d2, var):
    a, b = min(d1, d2), max(d1, d2)
    return f"grad2_{a}_{b}_{var}[pp]"

def _agrad(d, var):
    return f"agrad_{d}_{var}[pp]"


# Array-vs-stencil dispatch for derivatives of BSSN fields. When fused=True,
# the per-point loop has the stencil coefficient constants from
# _AVX_STENCIL_COEF_SETUP in scope and we inline a stencil expression instead
# of loading from a pre-computed grad_* / grad2_* / agrad_* array.

def _grad_expr(d, var, fused):
    """Return either array lookup or fused stencil for 1st derivative."""
    return _stencil_grad(d, var) if fused else _grad(d, var)


def _grad2_expr(d1, d2, var, fused, fuse_mixed=False):
    """Return expression for 2nd derivative.

    In fused mode:
      pure 2nd derivs (d1==d2): 7-point centered stencil (always fused)
      mixed 2nd derivs (d1!=d2): 36-term tensor-product stencil only if
        `fuse_mixed=True`, otherwise fall back to array lookup.

    The mixed stencil is 36 multiplies per output; with 33 mixed 2nd derivs
    in the BSSN cascade, that's ~1200 extra multiplies per grid point. In
    practice this is slower than keeping mixed derivs in a small workspace
    (~66 arrays pre-computed via the 2-pass scheme).
    """
    if fused:
        if d1 == d2:
            return _stencil_grad2_pure(d1, var)
        if fuse_mixed:
            return _stencil_grad2_mixed(d1, d2, var)
    return _grad2(d1, d2, var)


def _agrad_expr(d, var, fused):
    """Return expression for advective derivative.

    dendrogr uses centered grad for advective (not upwinded), so in fused mode
    we emit the same centered stencil.
    """
    return _stencil_grad(d, var) if fused else _agrad(d, var)


# ---------------------------------------------------------------------------
# Layer emitters -- each returns a list of C++ code lines
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Top-level generators
# ---------------------------------------------------------------------------

def generate_cascade_cpu(config=None):
    """Generate a scalar C++ include file for the cascade RHS (tensor-loop form).

    Uses the unified VEC-macro-based emitters with a trivial scalar macros
    header (VEC = double, VMUL(a,b) = a*b, etc.). The same emitters produce
    AVX2 / AVX-512 output when given the AVX macro headers.

    Note: SSL / CAHD / Rochester gauge are not yet ported to the unified
    emitters. For those, use generate_cascade_cpu_unrolled() which has its
    own (fully inline) emission path supporting all gauge modifiers.
    """
    config = config or {}
    fused_derivs = config.get("fused_derivs", False)

    lines = [
        _SCALAR_MACROS_HEADER,
        f"// BSSN RHS via polynomial cascade (scalar, tensor-loop form)",
        f"// Generated by cascade_codegen.py -- do not edit",
        "// This file is included inside the per-point (i,j,k) loop in rhs.cpp.",
        "// All derivative arrays (grad_*, grad2_*, agrad_*) and state arrays",
        "// (alpha, chi, gt0-gt5, ...) are already in scope.",
        "{",
    ]

    body = []
    if fused_derivs:
        body += _comment("--- Stencil coefficient constants (per block) ---")
        body += _AVX_STENCIL_COEF_SETUP
        body += [""]
    body += emit_unpack_state()
    body += [""]
    body += emit_unpack_derivs(fused_derivs=fused_derivs)
    body += [""]
    body += emit_L1()
    body += [""]
    body += emit_L2()
    body += [""]
    body += emit_L3()
    body += [""]
    body += emit_L4()
    body += [""]
    body += emit_L5()
    body += [""]
    body += emit_L6()
    body += [""]
    body += emit_L7(config)

    lines += _indent(body)
    lines.append("}")

    return "\n".join(lines) + "\n"


def generate_cascade_cpu_unrolled(config=None):
    """Generate a fully-unrolled C++ include file for the cascade RHS.

    Every tensor loop is expanded into explicit scalar assignments.
    No arrays, no for-loops inside the per-point code — just flat scalars
    like the CSE code. This enables the compiler to vectorize the outer
    loop and keep everything in registers.

    Config options:
      fused_derivs : bool
        If True, inline finite-difference stencils instead of reading from
        pre-computed grad_*/agrad_* arrays. Mixed 2nd derivs (grad2_0_1 etc.)
        still come from arrays (reduced workspace). Default False.
    """
    config = config or {}
    gauge = config.get("gauge", "standard")
    ssl = config.get("ssl", False)
    cahd = config.get("cahd", False)
    fused = config.get("fused_derivs", False)
    fuse_mixed = config.get("fuse_mixed_derivs", False)
    eta_mode = config.get("eta_mode", "const")
    eta_expr = "eta[pp]" if eta_mode == "func" else "eta"

    L = []  # accumulate lines
    def emit(s=""):
        L.append(s)

    emit("// BSSN RHS via polynomial cascade (UNROLLED)")
    emit("// Generated by cascade_codegen.py -- do not edit")
    emit("// Fully unrolled: no inner loops, all scalar variables")
    if fused:
        emit("// FUSED DERIVATIVES: 1st derivs and pure 2nd derivs from inline stencils")
        if fuse_mixed:
            emit("// FULLY FUSED: mixed 2nd derivs also from inline 36-term stencils")
            emit("// (no grad2_* workspace reads at all)")
        else:
            emit("// (mixed 2nd derivs still read from grad2_* arrays -- shrunk workspace)")
    emit("")

    if fused:
        emit("// Stencil coefficients (computed once per block invocation)")
        emit("const double idx60    = (1.0/60.0)  / hx;")
        emit("const double idy60    = (1.0/60.0)  / hy;")
        emit("const double idz60    = (1.0/60.0)  / hz;")
        emit("const double idx2_180 = (1.0/180.0) / (hx*hx);")
        emit("const double idy2_180 = (1.0/180.0) / (hy*hy);")
        emit("const double idz2_180 = (1.0/180.0) / (hz*hz);")
        emit("// Mixed 2nd-deriv coefficient: 1/(60*60*h1*h2) = 1/(3600*h1*h2)")
        emit("const double idxy_3600 = 1.0 / (3600.0 * hx * hy);")
        emit("const double idxz_3600 = 1.0 / (3600.0 * hx * hz);")
        emit("const double idyz_3600 = 1.0 / (3600.0 * hy * hz);")
        emit("")

    # --- Unpack state ---
    emit("const double al = alpha[pp];")
    emit("const double be0 = beta0[pp];")
    emit("const double be1 = beta1[pp];")
    emit("const double be2 = beta2[pp];")
    for i in range(3):
        for j in range(i, 3):
            s = sym(i, j)
            emit(f"const double gt{i}{j} = gt{s}[pp];")
            if i != j:
                emit(f"const double gt{j}{i} = gt{i}{j};")
    emit("const double ch = chi[pp];")
    for i in range(3):
        for j in range(i, 3):
            s = sym(i, j)
            emit(f"const double At{i}{j} = At{s}[pp];")
            if i != j:
                emit(f"const double At{j}{i} = At{i}{j};")
    emit("const double Kv = K[pp];")
    for i in range(3):
        emit(f"const double Gtv{i} = Gt{i}[pp];")
    for i in range(3):
        emit(f"const double Bv{i} = B{i}[pp];")

    # --- Unpack derivatives (flat scalars) ---
    emit("")
    if fused:
        emit("// 1st derivs: inline stencils (no workspace reads)")
    # First derivs of scalars
    for var, short in [("alpha", "al"), ("chi", "ch"), ("K", "K")]:
        for d in range(3):
            emit(f"const double d_{short}{d} = {_grad_expr(d, var, fused)};")
    # First derivs of beta, Gt
    for var, short in [("beta", "be"), ("Gt", "Gt")]:
        for c in range(3):
            for d in range(3):
                emit(f"const double d_{short}{c}{d} = {_grad_expr(d, var+str(c), fused)};")
    # First derivs of gt -> d_gt_DIR_I_J
    for dr in range(3):
        for i in range(3):
            for j in range(i, 3):
                emit(f"const double d_gt{dr}{i}{j} = {_grad_expr(dr, 'gt'+str(sym(i,j)), fused)};")
                if i != j:
                    emit(f"const double d_gt{dr}{j}{i} = d_gt{dr}{i}{j};")
    # Second derivs
    if fused:
        emit("// 2nd derivs: pure (xx/yy/zz) from inline stencils, mixed from arrays")
    for var, short in [("alpha", "al"), ("chi", "ch")]:
        for d1 in range(3):
            for d2 in range(d1, 3):
                emit(f"const double d2_{short}{d1}{d2} = {_grad2_expr(d1, d2, var, fused, fuse_mixed)};")
                if d1 != d2:
                    emit(f"const double d2_{short}{d2}{d1} = d2_{short}{d1}{d2};")
    for c in range(3):
        for d1 in range(3):
            for d2 in range(d1, 3):
                emit(f"const double d2_be{c}{d1}{d2} = {_grad2_expr(d1, d2, 'beta'+str(c), fused, fuse_mixed)};")
                if d1 != d2:
                    emit(f"const double d2_be{c}{d2}{d1} = d2_be{c}{d1}{d2};")
    for s in range(6):
        for i in range(3):
            for j in range(i, 3):
                if sym(i, j) == s:
                    si, sj = i, j
                    break
        for d1 in range(3):
            for d2 in range(d1, 3):
                emit(f"const double d2_gt{si}{sj}{d1}{d2} = {_grad2_expr(d1, d2, 'gt'+str(s), fused, fuse_mixed)};")
                if d1 != d2:
                    emit(f"const double d2_gt{si}{sj}{d2}{d1} = d2_gt{si}{sj}{d1}{d2};")
                if si != sj:
                    emit(f"const double d2_gt{sj}{si}{d1}{d2} = d2_gt{si}{sj}{d1}{d2};")
                    if d1 != d2:
                        emit(f"const double d2_gt{sj}{si}{d2}{d1} = d2_gt{si}{sj}{d1}{d2};")

    # Advective derivs (dendrogr uses centered, not upwinded)
    # In fused mode, these are IDENTICAL to the d_* values we already computed,
    # so just alias them rather than re-evaluating stencils.
    if fused:
        emit("// Advective derivs (dendrogr: centered = non-advective): alias existing locals")
        for var, short in [("alpha", "al"), ("chi", "ch"), ("K", "K")]:
            for d in range(3):
                emit(f"const double ad_{short}{d} = d_{short}{d};")
        for var, short in [("beta", "be"), ("Gt", "Gt")]:
            for c in range(3):
                for d in range(3):
                    emit(f"const double ad_{short}{c}{d} = d_{short}{c}{d};")
        # B and At don't have d_* locals emitted — compute stencils for those
        for c in range(3):
            for d in range(3):
                emit(f"const double ad_B{c}{d} = {_stencil_grad(d, 'B'+str(c))};")
        for dr in range(3):
            for i in range(3):
                for j in range(i, 3):
                    emit(f"const double ad_gt{dr}{i}{j} = d_gt{dr}{i}{j};")
                    if i != j:
                        emit(f"const double ad_gt{dr}{j}{i} = ad_gt{dr}{i}{j};")
        for dr in range(3):
            for i in range(3):
                for j in range(i, 3):
                    s = sym(i, j)
                    emit(f"const double ad_At{dr}{i}{j} = {_stencil_grad(dr, 'At'+str(s))};")
                    if i != j:
                        emit(f"const double ad_At{dr}{j}{i} = ad_At{dr}{i}{j};")
    else:
        for var, short in [("alpha", "al"), ("chi", "ch"), ("K", "K")]:
            for d in range(3):
                emit(f"const double ad_{short}{d} = {_agrad(d, var)};")
        for var, short in [("beta", "be"), ("Gt", "Gt"), ("B", "B")]:
            for c in range(3):
                for d in range(3):
                    emit(f"const double ad_{short}{c}{d} = {_agrad(d, var+str(c))};")
        for dr in range(3):
            for i in range(3):
                for j in range(i, 3):
                    s = sym(i, j)
                    emit(f"const double ad_gt{dr}{i}{j} = {_agrad(dr, 'gt'+str(s))};")
                    if i != j:
                        emit(f"const double ad_gt{dr}{j}{i} = ad_gt{dr}{i}{j};")
        for dr in range(3):
            for i in range(3):
                for j in range(i, 3):
                    s = sym(i, j)
                    emit(f"const double ad_At{dr}{i}{j} = {_agrad(dr, 'At'+str(s))};")
                    if i != j:
                        emit(f"const double ad_At{dr}{j}{i} = ad_At{dr}{i}{j};")

    # === L1: Inverse metric (fully unrolled) ===
    emit("")
    emit("// === L1: Inverse metric ===")
    emit("const double det_gt = gt00*(gt11*gt22 - gt12*gt12) - gt01*(gt01*gt22 - gt12*gt02) + gt02*(gt01*gt12 - gt11*gt02);")
    emit("const double inv_det = 1.0/det_gt;")
    emit(f"const double igt00 = (gt11*gt22-gt12*gt12)*inv_det;")
    emit(f"const double igt01 = (gt02*gt12-gt01*gt22)*inv_det;")
    emit(f"const double igt02 = (gt01*gt12-gt02*gt11)*inv_det;")
    emit(f"const double igt10 = igt01;")
    emit(f"const double igt11 = (gt00*gt22-gt02*gt02)*inv_det;")
    emit(f"const double igt12 = (gt01*gt02-gt00*gt12)*inv_det;")
    emit(f"const double igt20 = igt02;")
    emit(f"const double igt21 = igt12;")
    emit(f"const double igt22 = (gt00*gt11-gt01*gt01)*inv_det;")
    emit("const double chi_inv = 1.0/ch;")

    # === L2: First Christoffel C1[k][i][j] -- symmetric in (i,j); 18 unique ===
    emit("")
    emit("// === L2: First Christoffel C1[k][i][j] ===")
    for k in range(3):
        for i in range(3):
            for j in range(i, 3):
                emit(f"const double C1_{k}{i}{j} = 0.5*(d_gt{j}{k}{i} + d_gt{i}{k}{j} - d_gt{k}{i}{j});")
                if i != j:
                    emit(f"const double C1_{k}{j}{i} = C1_{k}{i}{j};")

    # === L3: Second Christoffel C2[i][j][k] -- symmetric in (j,k); 18 unique ===
    emit("")
    emit("// === L3: Second Christoffel C2[i][j][k] ===")
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                terms = " + ".join(f"igt{i}{l}*C1_{l}{j}{k}" for l in range(3))
                emit(f"const double C2_{i}{j}{k} = {terms};")
                if j != k:
                    emit(f"const double C2_{i}{k}{j} = C2_{i}{j}{k};")

    # === L4: Complete Christoffel C3[i][j][k] -- symmetric in (j,k); 18 unique ===
    emit("")
    emit("// === L4: Complete Christoffel C3[i][j][k] ===")
    emit("// hoist 0.5*chi_inv once (reused in L5).")
    emit("const double half_chi_inv = 0.5*chi_inv;")
    for i in range(3):
        terms = " + ".join(f"igt{i}{m}*d_ch{m}" for m in range(3))
        emit(f"const double igt_dchi{i} = {terms};")
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                dij = "1.0" if i == j else "0.0"
                dik = "1.0" if i == k else "0.0"
                emit(f"const double C3_{i}{j}{k} = C2_{i}{j}{k} - half_chi_inv*({dij}*d_ch{k} + {dik}*d_ch{j} - gt{j}{k}*igt_dchi{i});")
                if j != k:
                    emit(f"const double C3_{i}{k}{j} = C3_{i}{j}{k};")

    # === L5: Ricci ===
    emit("")
    emit("// === L5: Ricci tensor ===")
    # CalGt
    for i in range(3):
        terms = " + ".join(f"igt{k}{l}*C2_{i}{k}{l}" for k in range(3) for l in range(3))
        emit(f"const double CalGt{i} = {terms};")

    # Rt (symmetric, compute upper triangle)
    emit("")
    for i in range(3):
        for j in range(i, 3):
            # t1: -0.5 * igt[l][m]*d2_gt[i][j][l][m]
            t1_terms = " + ".join(f"igt{l}{m}*d2_gt{i}{j}{l}{m}" for l in range(3) for m in range(3))
            emit(f"const double Rt_{i}{j}_t1 = -0.5*({t1_terms});")
            # t2: 0.5*(gt[k][i]*d_Gt[k][j] + gt[k][j]*d_Gt[k][i])
            t2_terms = " + ".join(f"gt{k}{i}*d_Gt{k}{j} + gt{k}{j}*d_Gt{k}{i}" for k in range(3))
            emit(f"const double Rt_{i}{j}_t2 = 0.5*({t2_terms});")
            # t3: 0.5*CalGt[k]*(C1[i][j][k] + C1[j][i][k])
            t3_terms = " + ".join(f"CalGt{k}*(C1_{i}{j}{k} + C1_{j}{i}{k})" for k in range(3))
            emit(f"const double Rt_{i}{j}_t3 = 0.5*({t3_terms});")
            # t4: igt[l][m]*(C2[k][l][i]*C1[j][k][m] + C2[k][l][j]*C1[i][k][m] + C2[k][i][m]*C1[k][l][j])
            t4_parts = []
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        t4_parts.append(f"igt{l}{m}*(C2_{k}{l}{i}*C1_{j}{k}{m} + C2_{k}{l}{j}*C1_{i}{k}{m} + C2_{k}{i}{m}*C1_{k}{l}{j})")
            # Break into multiple lines for readability
            emit(f"const double Rt_{i}{j}_t4 = " + " + ".join(t4_parts) + ";")
            emit(f"const double Rt_{i}{j} = Rt_{i}{j}_t1 + Rt_{i}{j}_t2 + Rt_{i}{j}_t3 + Rt_{i}{j}_t4;")
            if i != j:
                emit(f"const double Rt_{j}{i} = Rt_{i}{j};")

    # Rphi -- C2_dc is sym(i,j) since C2 is sym(i,j) in last two; xRphi inherits
    emit("")
    emit("// Rphi  (half_chi_inv hoisted in L4; C2_dc/xRphi are symmetric in (i,j))")
    emit("const double chi_inv2 = chi_inv*chi_inv;")
    emit("const double quarter_chi_inv2 = 0.25*chi_inv2;")
    for i in range(3):
        for j in range(i, 3):
            terms = " + ".join(f"C2_{k}{j}{i}*d_ch{k}" for k in range(3))
            emit(f"const double C2_dc{i}{j} = {terms};")
            if i != j:
                emit(f"const double C2_dc{j}{i} = C2_dc{i}{j};")
    for i in range(3):
        for j in range(i, 3):
            emit(f"const double xRphi{i}{j} = half_chi_inv*(d2_ch{i}{j} - C2_dc{i}{j}) - quarter_chi_inv2*d_ch{i}*d_ch{j};")
            if i != j:
                emit(f"const double xRphi{j}{i} = xRphi{i}{j};")

    igt_d2chi_terms = " + ".join(f"igt{k}{l}*d2_ch{k}{l}" for k in range(3) for l in range(3))
    igt_dchi_dchi_terms = " + ".join(f"igt{k}{l}*d_ch{k}*d_ch{l}" for k in range(3) for l in range(3))
    calgt_dchi_terms = " + ".join(f"CalGt{k}*d_ch{k}" for k in range(3))
    emit(f"const double igt_d2chi = {igt_d2chi_terms};")
    emit(f"const double igt_dchi_dchi = {igt_dchi_dchi_terms};")
    emit(f"const double CalGt_dchi = {calgt_dchi_terms};")
    emit("const double scalar_part = igt_d2chi - 1.5*chi_inv*igt_dchi_dchi - CalGt_dchi;")
    emit("const double half_chi_inv_scalar = half_chi_inv*scalar_part;")

    # R = Rt + xRphi + (half_chi_inv * scalar_part) * gt; symmetric in (i,j)
    emit("")
    for i in range(3):
        for j in range(i, 3):
            emit(f"const double R{i}{j} = Rt_{i}{j} + xRphi{i}{j} + half_chi_inv_scalar*gt{i}{j};")
            if i != j:
                emit(f"const double R{j}{i} = R{i}{j};")

    # === L6: Derived quantities ===
    emit("")
    emit("// === L6: Derived quantities ===")
    # At_UU
    for i in range(3):
        for j in range(i, 3):
            terms = " + ".join(f"igt{i}{k}*igt{j}{l}*At{k}{l}" for k in range(3) for l in range(3))
            emit(f"const double At_UU{i}{j} = {terms};")
            if i != j:
                emit(f"const double At_UU{j}{i} = At_UU{i}{j};")

    # AikAkj: At[i][k] * (igt[k][l]*At[l][j])
    for k in range(3):
        for j in range(3):
            terms = " + ".join(f"igt{k}{l}*At{l}{j}" for l in range(3))
            emit(f"const double A_ud{k}{j} = {terms};")
    for i in range(3):
        for j in range(i, 3):
            terms = " + ".join(f"At{i}{k}*A_ud{k}{j}" for k in range(3))
            emit(f"const double AikAkj{i}{j} = {terms};")
            if i != j:
                emit(f"const double AikAkj{j}{i} = AikAkj{i}{j};")

    # DiDj_a -- symmetric in (i,j); emit 6 unique then mirror.
    for i in range(3):
        for j in range(i, 3):
            c3_terms = " + ".join(f"C3_{l}{i}{j}*d_al{l}" for l in range(3))
            emit(f"const double DiDj_a{i}{j} = d2_al{i}{j} - ({c3_terms});")
            if i != j:
                emit(f"const double DiDj_a{j}{i} = DiDj_a{i}{j};")

    # trace_free(al*R - DiDj_a) -- hoist (1/3)*trace; tf is sym in (i,j).
    trace_terms = " + ".join(f"igt{i}{j}*(al*R{i}{j} - DiDj_a{i}{j})" for i in range(3) for j in range(3))
    emit(f"const double trace_val = {trace_terms};")
    emit(f"const double tf_scalar = (1.0/3.0)*trace_val;")
    for i in range(3):
        for j in range(i, 3):
            emit(f"const double tf{i}{j} = (al*R{i}{j} - DiDj_a{i}{j}) - gt{i}{j}*tf_scalar;")
            if i != j:
                emit(f"const double tf{j}{i} = tf{i}{j};")

    # At_sqr
    terms = " + ".join(f"At{i}{j}*At_UU{i}{j}" for i in range(3) for j in range(3))
    emit(f"const double At_sqr = {terms};")

    # laplacian(alpha) = sum_{ij} ch*igt[i][j] * DiDj_a[i][j]; reuse DiDj_a.
    lap_parts = " + ".join(f"igt{i}{j}*DiDj_a{i}{j}" for i in range(3) for j in range(3))
    emit(f"const double lap_a = ch*({lap_parts});")

    emit(f"const double div_be = d_be00 + d_be11 + d_be22;")
    emit(f"const double w = -2.0/3.0;")

    # === L7: RHS outputs ===
    emit("")
    emit("// === L7: RHS outputs ===")
    # a_rhs (+ SSL correction if enabled; #ifdef so one file serves both builds)
    emit("a_rhs[pp] = lambda[0]*(be0*ad_al0+be1*ad_al1+be2*ad_al2) - 2.0*al*Kv;")
    if ssl:
        emit("#ifdef BSSN_ENABLE_SSL_HD")
        emit("    // SSL: a_rhs += -sqrt(chi)*h_ssl*(alpha-sqrt(chi))*exp(-0.5*t^2/sig^2)")
        emit("    {")
        emit("        const double _ssl_fac = -h_ssl * std::exp(-0.5*t*t/(sig_ssl*sig_ssl));")
        emit("        const double _sqrt_chi = std::sqrt(ch);")
        emit("        a_rhs[pp] += _ssl_fac * _sqrt_chi * (al - _sqrt_chi);")
        emit("    }")
        emit("#endif")

    # b_rhs
    if gauge == "standard":
        for i in range(3):
            emit(f"b_rhs{i}[pp] = 0.75*(lambda_f[0]+lambda_f[1]*al)*Bv{i} + lambda[1]*(be0*ad_be{i}0+be1*ad_be{i}1+be2*ad_be{i}2);")

    # gt_rhs
    for i in range(3):
        for j in range(i, 3):
            label = sym_label(i, j)
            lie_terms = " + ".join(f"(k=={kk}?be{kk}:0.0)*ad_gt{kk}{i}{j}" for kk in range(3))
            # Simpler: just use be0/be1/be2 directly
            lie = f"be0*ad_gt0{i}{j}+be1*ad_gt1{i}{j}+be2*ad_gt2{i}{j}"
            for k in range(3):
                lie += f" + gt{i}{k}*d_be{k}{j} + gt{k}{j}*d_be{k}{i}"
            lie += f" + w*gt{i}{j}*div_be"
            emit(f"gt_rhs{label}[pp] = {lie} - 2.0*al*At{i}{j};")

    # chi_rhs (+ CAHD correction if enabled)
    emit("chi_rhs[pp] = (be0*ad_ch0+be1*ad_ch1+be2*ad_ch2) + w*ch*div_be + (2.0/3.0)*ch*al*Kv;")
    if cahd:
        rscal = " + ".join(f"igt{i}{j}*R{i}{j}" for i in range(3) for j in range(3))
        emit("#ifdef BSSN_ENABLE_SSL_HD")
        emit("    // CAHD: chi_rhs += cahd_coef*ham*chi;  ham = chi*R_scalar - At_sqr + (2/3)K^2")
        emit("    // coef POSITIVE (no leading -); see memory/project_cahd_sign_bug.md")
        emit("    {")
        emit(f"        const double _R_scalar = {rscal};")
        emit("        const double _ham = ch*_R_scalar + (2.0/3.0)*Kv*Kv - At_sqr;")
        emit("        const double _cahd_coef = BSSN_CAHD_C * dx_i*dx_i / (1.0 + 10.0*dx_i*dx_i) / dt;")
        emit("        chi_rhs[pp] += _cahd_coef * _ham * ch;")
        emit("    }")
        emit("#endif")

    # At_rhs
    for i in range(3):
        for j in range(i, 3):
            label = sym_label(i, j)
            lie = f"be0*ad_At0{i}{j}+be1*ad_At1{i}{j}+be2*ad_At2{i}{j}"
            for k in range(3):
                lie += f" + At{i}{k}*d_be{k}{j} + At{k}{j}*d_be{k}{i}"
            lie += f" + w*At{i}{j}*div_be"
            emit(f"At_rhs{label}[pp] = ({lie}) + ch*tf{i}{j} + al*(Kv*At{i}{j} - 2.0*AikAkj{i}{j});")

    # K_rhs
    emit("K_rhs[pp] = (be0*ad_K0+be1*ad_K1+be2*ad_K2) - lap_a + al*(Kv*Kv/3.0 + At_sqr);")

    # Gt_rhs
    for i in range(3):
        t1 = f"be0*ad_Gt{i}0+be1*ad_Gt{i}1+be2*ad_Gt{i}2"
        t2 = " + ".join(f"CalGt{j}*d_be{i}{j}" for j in range(3))
        t3 = f"(2.0/3.0)*CalGt{i}*div_be"
        t4_parts = []
        for j in range(3):
            for k in range(3):
                t4_parts.append(f"igt{j}{k}*d2_be{i}{j}{k} + igt{i}{j}*d2_be{k}{j}{k}/3.0")
        t4 = " + ".join(t4_parts)
        t5 = " + ".join(f"At_UU{i}{j}*d_al{j}" for j in range(3))
        t6_parts = []
        for j in range(3):
            for k in range(3):
                t6_parts.append(f"C2_{i}{j}{k}*At_UU{j}{k}")
        t6 = " + ".join(t6_parts)
        t7_parts = []
        for j in range(3):
            t7_parts.append(f"3.0*chi_inv*At_UU{i}{j}*d_ch{j} + (4.0/3.0)*igt{i}{j}*d_K{j}")
        t7 = " + ".join(t7_parts)
        emit(f"const double Gt_rhs_v{i} = ({t1}) - ({t2}) + {t3} + ({t4}) - 2.0*({t5}) + 2.0*al*({t6}) - al*({t7});")
    for i in range(3):
        emit(f"Gt_rhs{i}[pp] = Gt_rhs_v{i};")

    # B_rhs
    if gauge == "standard":
        for i in range(3):
            emit(f"B_rhs{i}[pp] = Gt_rhs_v{i} - {eta_expr}*Bv{i} + lambda[2]*(be0*ad_B{i}0+be1*ad_B{i}1+be2*ad_B{i}2) - lambda[3]*(be0*ad_Gt{i}0+be1*ad_Gt{i}1+be2*ad_Gt{i}2);")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Unified emitters: produce VEC-macro C++ that compiles as scalar, AVX2, or
# AVX-512 depending on which macro header was prepended.
#
# The SIMD macro headers (_SCALAR_MACROS_HEADER / _AVX_MACROS_HEADER /
# _AVX512_MACROS_HEADER) and the SIMD stencil helpers (_avx_stencil_1st /
# _avx_stencil_2nd_pure) are defined in cascade_common.py and imported at
# the top of this file.
# ---------------------------------------------------------------------------


def emit_unpack_state():
    """Load state from arrays into VEC tensor structure at offset pp."""
    lines = _comment("--- Unpack state (4 lanes at pp..pp+3) ---")
    lines += [
        "const VEC al = VLOAD(alpha+pp);",
        "const VEC ch = VLOAD(chi+pp);",
        "const VEC Kv = VLOAD(K+pp);",
        "const VEC be[3] = {VLOAD(beta0+pp), VLOAD(beta1+pp), VLOAD(beta2+pp)};",
        "const VEC Bv[3] = {VLOAD(B0+pp), VLOAD(B1+pp), VLOAD(B2+pp)};",
        "const VEC Gtv[3] = {VLOAD(Gt0+pp), VLOAD(Gt1+pp), VLOAD(Gt2+pp)};",
        "const VEC gt_sym6[6] = {VLOAD(gt0+pp), VLOAD(gt1+pp), VLOAD(gt2+pp),",
        "                         VLOAD(gt3+pp), VLOAD(gt4+pp), VLOAD(gt5+pp)};",
        "const VEC gt[3][3] = {{gt_sym6[0], gt_sym6[1], gt_sym6[2]},",
        "                       {gt_sym6[1], gt_sym6[3], gt_sym6[4]},",
        "                       {gt_sym6[2], gt_sym6[4], gt_sym6[5]}};",
        "const VEC At_sym6[6] = {VLOAD(At0+pp), VLOAD(At1+pp), VLOAD(At2+pp),",
        "                         VLOAD(At3+pp), VLOAD(At4+pp), VLOAD(At5+pp)};",
        "const VEC At[3][3] = {{At_sym6[0], At_sym6[1], At_sym6[2]},",
        "                       {At_sym6[1], At_sym6[3], At_sym6[4]},",
        "                       {At_sym6[2], At_sym6[4], At_sym6[5]}};",
    ]
    return lines


def emit_unpack_derivs(fused_derivs=False):
    """Load derivatives as VEC tensors.

    If fused_derivs is False: all derivs are VLOAD'd from precomputed arrays.
    If True: 1st derivs and pure 2nd derivs (xx, yy, zz) are computed inline via
    4-wide AVX stencils on the state arrays; mixed 2nd derivs (xy, xz, yz) still
    come from precomputed arrays (2D stencils are expensive).
    """
    lines = _comment("--- Unpack derivatives ---")
    if fused_derivs:
        # Map scalar name -> state array name
        scalar_vars = [("al", "alpha"), ("ch", "chi"), ("K", "K")]
        # Map symmetric tensor slot -> gt{s} array, At{s} array
        gt_vars = [f"gt{s}" for s in range(6)]
        at_vars = [f"At{s}" for s in range(6)]
        be_vars = ["beta0", "beta1", "beta2"]
        Gt_vars = ["Gt0", "Gt1", "Gt2"]
        B_vars = ["B0", "B1", "B2"]

        # 1st derivs of scalars: d_al[d], d_ch[d], d_K[d]
        lines.append("VEC d_al[3], d_ch[3], d_K[3];")
        for d in range(3):
            lines.append(f"d_al[{d}] = {_avx_stencil_1st(d, 'alpha')};")
            lines.append(f"d_ch[{d}] = {_avx_stencil_1st(d, 'chi')};")
            lines.append(f"d_K[{d}]  = {_avx_stencil_1st(d, 'K')};")

        # 1st derivs of vector fields: d_be, d_Gt, d_B
        for short, arrs in [("be", be_vars), ("Gt", Gt_vars), ("B", B_vars)]:
            lines.append(f"VEC d_{short}[3][3];")
            for c in range(3):
                for d in range(3):
                    lines.append(f"d_{short}[{c}][{d}] = {_avx_stencil_1st(d, arrs[c])};")

        # 1st derivs of symmetric tensors gt, At -> d_gt[d][i][j], d_At[d][i][j]
        lines.append("VEC d_gt[3][3][3];")
        lines.append("VEC d_At[3][3][3];")
        for d in range(3):
            for i in range(3):
                for j in range(i, 3):
                    s = i if i == j else (i + j) if not (i == 0 and j == 2) else 2
                    # Map (i,j) pair to symmetric index 0..5
                    sym_idx = {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(i,j)]
                    lines.append(f"d_gt[{d}][{i}][{j}] = {_avx_stencil_1st(d, gt_vars[sym_idx])};")
                    if i != j:
                        lines.append(f"d_gt[{d}][{j}][{i}] = d_gt[{d}][{i}][{j}];")
                    lines.append(f"d_At[{d}][{i}][{j}] = {_avx_stencil_1st(d, at_vars[sym_idx])};")
                    if i != j:
                        lines.append(f"d_At[{d}][{j}][{i}] = d_At[{d}][{i}][{j}];")

        # Pure 2nd derivs (xx, yy, zz) inline; mixed 2nds from arrays
        # d2_al, d2_ch
        lines.append("VEC d2_al[3][3], d2_ch[3][3];")
        for i in range(3):
            for j in range(i, 3):
                sym_idx = {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(i,j)]
                if i == j:
                    lines.append(f"d2_al[{i}][{j}] = {_avx_stencil_2nd_pure(i, 'alpha')};")
                    lines.append(f"d2_ch[{i}][{j}] = {_avx_stencil_2nd_pure(i, 'chi')};")
                else:
                    lines.append(f"d2_al[{i}][{j}] = VLOAD(d2_al_p[{sym_idx}]+pp);")
                    lines.append(f"d2_ch[{i}][{j}] = VLOAD(d2_ch_p[{sym_idx}]+pp);")
                if i != j:
                    lines.append(f"d2_al[{j}][{i}] = d2_al[{i}][{j}];")
                    lines.append(f"d2_ch[{j}][{i}] = d2_ch[{i}][{j}];")

        # d2_be[c][i][j]
        lines.append("VEC d2_be[3][3][3];")
        for c in range(3):
            for i in range(3):
                for j in range(i, 3):
                    sym_idx = {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(i,j)]
                    if i == j:
                        lines.append(f"d2_be[{c}][{i}][{j}] = {_avx_stencil_2nd_pure(i, be_vars[c])};")
                    else:
                        lines.append(f"d2_be[{c}][{i}][{j}] = VLOAD(d2_be_p[{c}][{sym_idx}]+pp);")
                    if i != j:
                        lines.append(f"d2_be[{c}][{j}][{i}] = d2_be[{c}][{i}][{j}];")

        # d2_gt[k][l][i][j] -- indexed by (gt_sym, kl_sym)
        lines.append("VEC d2_gt[3][3][3][3];")
        for i in range(3):
            for j in range(i, 3):
                gt_s = {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(i,j)]
                for k in range(3):
                    for l in range(k, 3):
                        kl_s = {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(k,l)]
                        if k == l:
                            lines.append(f"d2_gt[{k}][{l}][{i}][{j}] = {_avx_stencil_2nd_pure(k, gt_vars[gt_s])};")
                        else:
                            lines.append(f"d2_gt[{k}][{l}][{i}][{j}] = VLOAD(d2_gt_p[{gt_s}][{kl_s}]+pp);")
                        if k != l:
                            lines.append(f"d2_gt[{l}][{k}][{i}][{j}] = d2_gt[{k}][{l}][{i}][{j}];")
                        if i != j:
                            lines.append(f"d2_gt[{k}][{l}][{j}][{i}] = d2_gt[{k}][{l}][{i}][{j}];")
                            if k != l:
                                lines.append(f"d2_gt[{l}][{k}][{j}][{i}] = d2_gt[{k}][{l}][{i}][{j}];")

        lines += [
            "// Advective = centered in this harness/dendrogr convention",
            "const VEC (* const ad_al) = d_al;",
            "const VEC (* const ad_ch) = d_ch;",
            "const VEC (* const ad_K)  = d_K;",
            "const VEC (* const ad_be)[3] = d_be;",
            "const VEC (* const ad_Gt)[3] = d_Gt;",
            "const VEC (* const ad_B)[3]  = d_B;",
            "const VEC (* const ad_gt)[3][3] = d_gt;",
            "const VEC (* const ad_At)[3][3] = d_At;",
        ]
        return lines

    # Non-fused (original) path

    # 1st derivs of scalars
    lines += [
        "VEC d_al[3], d_ch[3], d_K[3];",
        "for (int d = 0; d < 3; d++) {",
        "    d_al[d] = VLOAD(d_al_p[d]+pp);",
        "    d_ch[d] = VLOAD(d_ch_p[d]+pp);",
        "    d_K[d]  = VLOAD(d_K_p[d]+pp);",
        "}",
        "VEC d_be[3][3], d_Gt[3][3], d_B[3][3];",
        "for (int c = 0; c < 3; c++)",
        "    for (int d = 0; d < 3; d++) {",
        "        d_be[c][d] = VLOAD(d_be_p[c][d]+pp);",
        "        d_Gt[c][d] = VLOAD(d_Gt_p[c][d]+pp);",
        "        d_B[c][d]  = VLOAD(d_B_p[c][d]+pp);",
        "    }",
        "VEC d_gt[3][3][3];",
        "for (int d = 0; d < 3; d++)",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = 0; j < 3; j++)",
        "            d_gt[d][i][j] = VLOAD(d_gt_p[cascade_sym_idx(i,j)][d]+pp);",
        "VEC d_At[3][3][3];",
        "for (int d = 0; d < 3; d++)",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = 0; j < 3; j++)",
        "            d_At[d][i][j] = VLOAD(d_At_p[cascade_sym_idx(i,j)][d]+pp);",
        "VEC d2_al[3][3], d2_ch[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++) {",
        "        d2_al[i][j] = VLOAD(d2_al_p[cascade_sym_idx(i,j)]+pp);",
        "        d2_ch[i][j] = VLOAD(d2_ch_p[cascade_sym_idx(i,j)]+pp);",
        "    }",
        "VEC d2_be[3][3][3];",
        "for (int c = 0; c < 3; c++)",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = 0; j < 3; j++)",
        "            d2_be[c][i][j] = VLOAD(d2_be_p[c][cascade_sym_idx(i,j)]+pp);",
        "VEC d2_gt[3][3][3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++)",
        "        for (int k = 0; k < 3; k++)",
        "            for (int l = 0; l < 3; l++)",
        "                d2_gt[k][l][i][j] = VLOAD(d2_gt_p[cascade_sym_idx(i,j)][cascade_sym_idx(k,l)]+pp);",
        "// Advective = centered in this harness/dendrogr convention",
        "const VEC (* const ad_al) = d_al;",
        "const VEC (* const ad_ch) = d_ch;",
        "const VEC (* const ad_K)  = d_K;",
        "const VEC (* const ad_be)[3] = d_be;",
        "const VEC (* const ad_Gt)[3] = d_Gt;",
        "const VEC (* const ad_B)[3]  = d_B;",
        "const VEC (* const ad_gt)[3][3] = d_gt;",
        "const VEC (* const ad_At)[3][3] = d_At;",
    ]
    return lines


def emit_L1():
    return _comment("=== L1: Inverse metric ===") + [
        "const VEC _half = VSET(0.5);",
        "const VEC _nhalf = VSET(-0.5);",
        "const VEC _inv3  = VSET(1.0/3.0);",
        "const VEC _w23   = VSET(-2.0/3.0);",
        "const VEC _two = VSET(2.0);",
        "const VEC det = VSUB(VADD(",
        "    VMUL(gt[0][0], VSUB(VMUL(gt[1][1],gt[2][2]), VMUL(gt[1][2],gt[1][2]))),",
        "    VMUL(gt[0][2], VSUB(VMUL(gt[0][1],gt[1][2]), VMUL(gt[1][1],gt[0][2])))),",
        "    VMUL(gt[0][1], VSUB(VMUL(gt[0][1],gt[2][2]), VMUL(gt[1][2],gt[0][2]))));",
        "const VEC di = VDIV(VSET(1.0), det);",
        "VEC igt[3][3];",
        "igt[0][0] = VMUL(VSUB(VMUL(gt[1][1],gt[2][2]),VMUL(gt[1][2],gt[1][2])),di);",
        "igt[0][1] = VMUL(VSUB(VMUL(gt[0][2],gt[1][2]),VMUL(gt[0][1],gt[2][2])),di);",
        "igt[0][2] = VMUL(VSUB(VMUL(gt[0][1],gt[1][2]),VMUL(gt[0][2],gt[1][1])),di);",
        "igt[1][0] = igt[0][1];",
        "igt[1][1] = VMUL(VSUB(VMUL(gt[0][0],gt[2][2]),VMUL(gt[0][2],gt[0][2])),di);",
        "igt[1][2] = VMUL(VSUB(VMUL(gt[0][1],gt[0][2]),VMUL(gt[0][0],gt[1][2])),di);",
        "igt[2][0] = igt[0][2];",
        "igt[2][1] = igt[1][2];",
        "igt[2][2] = VMUL(VSUB(VMUL(gt[0][0],gt[1][1]),VMUL(gt[0][1],gt[0][1])),di);",
        "const VEC chi_inv = VDIV(VSET(1.0), ch);",
    ]


def emit_L2():
    return _comment("=== L2: First Christoffel ===") + [
        "VEC C1[3][3][3];",
        "for (int k = 0; k < 3; k++)",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = 0; j < 3; j++)",
        "            C1[k][i][j] = VMUL(_half, VSUB(VADD(d_gt[j][k][i], d_gt[i][k][j]), d_gt[k][i][j]));",
    ]


def emit_L3():
    return _comment("=== L3: Second Christoffel ===") + [
        "VEC C2[3][3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++)",
        "        for (int k = 0; k < 3; k++) {",
        "            C2[i][j][k] = VFMA(igt[i][0], C1[0][j][k],",
        "                           VFMA(igt[i][1], C1[1][j][k], VMUL(igt[i][2], C1[2][j][k])));",
        "        }",
    ]


def emit_L4():
    return _comment("=== L4: Complete Christoffel ===") + [
        "VEC igt_dchi[3];",
        "for (int i = 0; i < 3; i++)",
        "    igt_dchi[i] = VFMA(igt[i][0], d_ch[0],",
        "                    VFMA(igt[i][1], d_ch[1], VMUL(igt[i][2], d_ch[2])));",
        "VEC C3[3][3][3];",
        "const VEC nhalf_chi_inv = VMUL(_nhalf, chi_inv);",
        "const VEC _zero = VSET(0.0);",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++)",
        "        for (int k = 0; k < 3; k++) {",
        "            VEC t_ij = (i==j) ? d_ch[k] : _zero;",
        "            VEC t_ik = (i==k) ? d_ch[j] : _zero;",
        "            VEC corr = VSUB(VADD(t_ij, t_ik), VMUL(gt[j][k], igt_dchi[i]));",
        "            C3[i][j][k] = VFMA(nhalf_chi_inv, corr, C2[i][j][k]);",
        "        }",
    ]


def emit_L5():
    return _comment("=== L5: Ricci tensor ===") + [
        "VEC CalGt[3];",
        "for (int i = 0; i < 3; i++) {",
        "    CalGt[i] = _zero;",
        "    for (int k = 0; k < 3; k++)",
        "        for (int l = 0; l < 3; l++)",
        "            CalGt[i] = VFMA(igt[k][l], C2[i][k][l], CalGt[i]);",
        "}",
        "VEC Rt[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        VEC t1=_zero, t2=_zero, t3=_zero, t4=_zero;",
        "        for (int l = 0; l < 3; l++)",
        "            for (int m = 0; m < 3; m++)",
        "                t1 = VFMA(igt[l][m], d2_gt[l][m][i][j], t1);",
        "        t1 = VMUL(_nhalf, t1);",
        "        for (int k = 0; k < 3; k++)",
        "            t2 = VADD(t2, VADD(VMUL(gt[k][i],d_Gt[k][j]), VMUL(gt[k][j],d_Gt[k][i])));",
        "        t2 = VMUL(_half, t2);",
        "        for (int k = 0; k < 3; k++)",
        "            t3 = VFMA(CalGt[k], VADD(C1[i][j][k],C1[j][i][k]), t3);",
        "        t3 = VMUL(_half, t3);",
        "        for (int k = 0; k < 3; k++)",
        "            for (int l = 0; l < 3; l++)",
        "                for (int m = 0; m < 3; m++)",
        "                    t4 = VFMA(igt[l][m],",
        "                          VADD(VADD(VMUL(C2[k][l][i],C1[j][k][m]),",
        "                                     VMUL(C2[k][l][j],C1[i][k][m])),",
        "                               VMUL(C2[k][i][m],C1[k][l][j])), t4);",
        "        Rt[i][j] = VADD(VADD(t1,t2), VADD(t3,t4));",
        "        Rt[j][i] = Rt[i][j];",
        "    }",
        "VEC C2_dc[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++) {",
        "        C2_dc[i][j] = _zero;",
        "        for (int k = 0; k < 3; k++)",
        "            C2_dc[i][j] = VFMA(C2[k][j][i], d_ch[k], C2_dc[i][j]);",
        "    }",
        "const VEC chi_inv2 = VMUL(chi_inv, chi_inv);",
        "const VEC half_chi_inv = VMUL(_half, chi_inv);",
        "const VEC quarter_chi_inv2 = VMUL(VSET(0.25), chi_inv2);",
        "VEC xRphi[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        xRphi[i][j] = VSUB(VMUL(half_chi_inv, VSUB(d2_ch[i][j], C2_dc[i][j])),",
        "                            VMUL(quarter_chi_inv2, VMUL(d_ch[i], d_ch[j])));",
        "        if (i != j) xRphi[j][i] = xRphi[i][j];",
        "    }",
        "VEC igt_d2chi = _zero, igt_dchi_dchi = _zero, CalGt_dchi = _zero;",
        "for (int k = 0; k < 3; k++) {",
        "    CalGt_dchi = VFMA(CalGt[k], d_ch[k], CalGt_dchi);",
        "    for (int l = 0; l < 3; l++) {",
        "        igt_d2chi = VFMA(igt[k][l], d2_ch[k][l], igt_d2chi);",
        "        igt_dchi_dchi = VFMA(igt[k][l], VMUL(d_ch[k], d_ch[l]), igt_dchi_dchi);",
        "    }",
        "}",
        "const VEC scalar_part = VSUB(VSUB(igt_d2chi, VMUL(VSET(1.5), VMUL(chi_inv, igt_dchi_dchi))), CalGt_dchi);",
        "const VEC half_chi_inv_scalar = VMUL(half_chi_inv, scalar_part);",
        "VEC R[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        R[i][j] = VADD(Rt[i][j], VADD(xRphi[i][j], VMUL(half_chi_inv_scalar, gt[i][j])));",
        "        if (i != j) R[j][i] = R[i][j];",
        "    }",
    ]


def emit_L6():
    return _comment("=== L6: Derived quantities ===") + [
        "// At_UU = igt . At . igt (symmetric); compute 6 unique then mirror",
        "VEC At_UU[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        At_UU[i][j] = _zero;",
        "        for (int k = 0; k < 3; k++)",
        "            for (int l = 0; l < 3; l++)",
        "                At_UU[i][j] = VFMA(VMUL(igt[i][k], igt[j][l]), At[k][l], At_UU[i][j]);",
        "        if (i != j) At_UU[j][i] = At_UU[i][j];",
        "    }",
        "VEC Aup[3][3];",
        "for (int k = 0; k < 3; k++)",
        "    for (int j = 0; j < 3; j++) {",
        "        Aup[k][j] = _zero;",
        "        for (int l = 0; l < 3; l++)",
        "            Aup[k][j] = VFMA(igt[k][l], At[l][j], Aup[k][j]);",
        "    }",
        "// AikAkj = At . Aup (symmetric); compute 6 unique then mirror",
        "VEC AikAkj[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        AikAkj[i][j] = _zero;",
        "        for (int k = 0; k < 3; k++)",
        "            AikAkj[i][j] = VFMA(At[i][k], Aup[k][j], AikAkj[i][j]);",
        "        if (i != j) AikAkj[j][i] = AikAkj[i][j];",
        "    }",
        "// DiDj_a is symmetric in (i,j) since d2 is sym and C3[l][i][j]=C3[l][j][i]",
        "VEC DiDj_a[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        DiDj_a[i][j] = d2_al[i][j];",
        "        for (int l = 0; l < 3; l++)",
        "            DiDj_a[i][j] = VSUB(DiDj_a[i][j], VMUL(C3[l][i][j], d_al[l]));",
        "        if (i != j) DiDj_a[j][i] = DiDj_a[i][j];",
        "    }",
        "VEC trace_val = _zero;",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++)",
        "        trace_val = VFMA(igt[i][j], VSUB(VMUL(al, R[i][j]), DiDj_a[i][j]), trace_val);",
        "// hoist (1/3)*trace once; reuse for all 6 unique tf components.",
        "const VEC tf_scalar = VMUL(trace_val, _inv3);",
        "VEC tf[3][3];",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = i; j < 3; j++) {",
        "        tf[i][j] = VSUB(VSUB(VMUL(al, R[i][j]), DiDj_a[i][j]),",
        "                         VMUL(gt[i][j], tf_scalar));",
        "        if (i != j) tf[j][i] = tf[i][j];",
        "    }",
        "VEC At_sqr = _zero;",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++)",
        "        At_sqr = VFMA(At[i][j], At_UU[i][j], At_sqr);",
        "VEC lap_a = _zero;",
        "for (int i = 0; i < 3; i++)",
        "    for (int j = 0; j < 3; j++) {",
        "        VEC dd = d2_al[i][j];",
        "        for (int l = 0; l < 3; l++)",
        "            dd = VSUB(dd, VMUL(C3[l][i][j], d_al[l]));",
        "        lap_a = VFMA(VMUL(ch, igt[i][j]), dd, lap_a);",
        "    }",
        "const VEC div_be = VADD(VADD(d_be[0][0], d_be[1][1]), d_be[2][2]);",
    ]


def emit_L7(config=None):
    """Layer 7 -- RHS assembly.

    config options:
      gauge   : "standard" (default) or "rochester"
      ssl     : bool -- shock-avoiding-lapse correction on a_rhs
      cahd    : bool -- constraint-added Hamiltonian damping on chi_rhs

    SSL/CAHD require extra scalars in scope at include point: h_ssl, sig_ssl,
    t (current time), BSSN_CAHD_C, dx_i, dt. dendrogr's bssnrhs() defines all
    of these unconditionally. For harness use, either set ssl/cahd=False, or
    add the scalars to the wrapper.
    """
    config = config or {}
    gauge = config.get("gauge", "standard")
    ssl = config.get("ssl", False)
    cahd = config.get("cahd", False)

    lines = _comment("=== L7: RHS assembly (stores to output arrays) ===")

    # a_rhs (+ SSL correction if enabled at codegen time)
    # SSL code is always wrapped in #ifdef BSSN_ENABLE_SSL_HD so one generated
    # file works for both builds (cmake flag toggles runtime behavior).
    a_rhs_block = [
        "{",
        "    VEC beadal = VFMA(be[0], ad_al[0], VFMA(be[1], ad_al[1], VMUL(be[2], ad_al[2])));",
        "    VEC out = VSUB(VMUL(VSET((double)lambda[0]), beadal), VMUL(_two, VMUL(al, Kv)));",
    ]
    if ssl:
        a_rhs_block += [
            "#ifdef BSSN_ENABLE_SSL_HD",
            "    // SSL: shock-avoiding lapse correction",
            "    //   a_rhs += -sqrt(chi) * h_ssl * (alpha - sqrt(chi)) * exp(-0.5*t^2/sig^2)",
            "    // Constant factor per RHS call (t, h_ssl, sig_ssl are scalars in scope).",
            "    const VEC _ssl_fac = VSET(-h_ssl * std::exp(-0.5*t*t/(sig_ssl*sig_ssl)));",
            "    const VEC _sqrt_chi = VSQRT(ch);",
            "    out = VFMA(_ssl_fac, VMUL(_sqrt_chi, VSUB(al, _sqrt_chi)), out);",
            "#endif",
        ]
    a_rhs_block += [
        "    VSTORE(a_rhs+pp, out);",
        "}",
    ]
    lines += a_rhs_block

    # b_rhs[i]
    if gauge == "standard":
        lines += [
            "{",
            "    VEC fac = VMUL(VSET(0.75), VADD(VSET(lambda_f[0]), VMUL(VSET(lambda_f[1]), al)));",
            "    double * const b_rhs_arr[3] = {b_rhs0, b_rhs1, b_rhs2};",
            "    for (int i = 0; i < 3; i++) {",
            "        VEC badbi = VFMA(be[0], ad_be[i][0], VFMA(be[1], ad_be[i][1], VMUL(be[2], ad_be[i][2])));",
            "        VEC out = VADD(VMUL(fac, Bv[i]), VMUL(VSET((double)lambda[1]), badbi));",
            "        VSTORE(b_rhs_arr[i]+pp, out);",
            "    }",
            "}",
        ]
    else:  # rochester
        lines += [
            "// Rochester gauge: b_rhs = XI[1]*(beta.grad beta) + 0.75*XI[2]*Gt - eta*beta",
            "{",
            "    double * const b_rhs_arr[3] = {b_rhs0, b_rhs1, b_rhs2};",
            "    for (int i = 0; i < 3; i++) {",
            "        VEC badbi = VFMA(be[0], ad_be[i][0], VFMA(be[1], ad_be[i][1], VMUL(be[2], ad_be[i][2])));",
            "        VEC out = VSUB(VFMA(VSET((double)BSSN_XI[1]), badbi,",
            "                            VMUL(VSET(0.75 * (double)BSSN_XI[2]), Gtv[i])),",
            "                       VMUL(eta, be[i]));",
            "        VSTORE(b_rhs_arr[i]+pp, out);",
            "    }",
            "}",
        ]
    # gt_rhs
    lines += [
        "{",
        "    double * const gt_rhs_arr[6] = {gt_rhs00, gt_rhs01, gt_rhs02, gt_rhs11, gt_rhs12, gt_rhs22};",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = i; j < 3; j++) {",
        "            VEC lie = VMUL(_w23, VMUL(gt[i][j], div_be));",
        "            for (int k = 0; k < 3; k++)",
        "                lie = VADD(lie, VADD(VADD(VMUL(be[k], ad_gt[k][i][j]), VMUL(gt[i][k], d_be[k][j])), VMUL(gt[k][j], d_be[k][i])));",
        "            VEC out = VSUB(lie, VMUL(_two, VMUL(al, At[i][j])));",
        "            VSTORE(gt_rhs_arr[cascade_sym_idx(i,j)]+pp, out);",
        "        }",
        "}",
    ]

    # chi_rhs (+ CAHD correction if enabled)
    chi_rhs_block = [
        "{",
        "    VEC beadch = VFMA(be[0], ad_ch[0], VFMA(be[1], ad_ch[1], VMUL(be[2], ad_ch[2])));",
        "    VEC term = VADD(VMUL(_w23, VMUL(ch, div_be)), VMUL(VSET(2.0/3.0), VMUL(ch, VMUL(al, Kv))));",
        "    VEC chi_rhs_out = VADD(beadch, term);",
    ]
    if cahd:
        chi_rhs_block += [
            "#ifdef BSSN_ENABLE_SSL_HD",
            "    // CAHD: constraint-added Hamiltonian damping",
            "    //   chi_rhs += -cahd_fac * ham / dt * chi",
            "    //   ham = chi*R_scalar - At_sqr + (2/3)*K^2",
            "    //   cahd_fac = BSSN_CAHD_C * dx^2 / (1 + 10*dx^2) (scalar per call)",
            "    VEC _R_scalar = _zero;",
            "    for (int i = 0; i < 3; i++)",
            "        for (int j = 0; j < 3; j++)",
            "            _R_scalar = VFMA(igt[i][j], R[i][j], _R_scalar);",
            "    VEC _ham = VSUB(VFMA(ch, _R_scalar, VMUL(VSET(2.0/3.0), VMUL(Kv, Kv))), At_sqr);",
            "    // NOTE: coefficient is POSITIVE (no leading -). The production CSE",
            "    // writes `chi_rhs += -BSSN_CAHD_C * (chi/12) * H_full / ...` where",
            "    // H_full ~ -12 * ham_computation; the minus, the chi/12, and H_full's",
            "    // sign cancel. See memory/project_cahd_sign_bug.md. Verified at machine",
            "    // precision against ssl_cahd CSE in the harness pseudo-verify.",
            "    const VEC _cahd_coef = VSET(BSSN_CAHD_C * dx_i * dx_i / (1.0 + 10.0 * dx_i * dx_i) / dt);",
            "    chi_rhs_out = VFMA(_cahd_coef, VMUL(_ham, ch), chi_rhs_out);",
            "#endif",
        ]
    chi_rhs_block += [
        "    VSTORE(chi_rhs+pp, chi_rhs_out);",
        "}",
    ]
    lines += chi_rhs_block

    # At_rhs
    lines += [
        "{",
        "    double * const At_rhs_arr[6] = {At_rhs00, At_rhs01, At_rhs02, At_rhs11, At_rhs12, At_rhs22};",
        "    for (int i = 0; i < 3; i++)",
        "        for (int j = i; j < 3; j++) {",
        "            VEC lie = VMUL(_w23, VMUL(At[i][j], div_be));",
        "            for (int k = 0; k < 3; k++)",
        "                lie = VADD(lie, VADD(VADD(VMUL(be[k], ad_At[k][i][j]), VMUL(At[i][k], d_be[k][j])), VMUL(At[k][j], d_be[k][i])));",
        "            VEC algeb = VADD(VMUL(ch, tf[i][j]), VMUL(al, VSUB(VMUL(Kv, At[i][j]), VMUL(_two, AikAkj[i][j]))));",
        "            VSTORE(At_rhs_arr[cascade_sym_idx(i,j)]+pp, VADD(lie, algeb));",
        "        }",
        "}",
    ]

    # K_rhs
    lines += [
        "{",
        "    VEC beadK = VFMA(be[0], ad_K[0], VFMA(be[1], ad_K[1], VMUL(be[2], ad_K[2])));",
        "    VEC out = VADD(VSUB(beadK, lap_a), VMUL(al, VADD(VMUL(Kv, VMUL(Kv, _inv3)), At_sqr)));",
        "    VSTORE(K_rhs+pp, out);",
        "}",
    ]

    # Gt_rhs (and keep as VEC for B_rhs)
    lines += [
        "VEC Gt_rhs_v[3];",
        "for (int i = 0; i < 3; i++) {",
        "    VEC t1 = VFMA(be[0], ad_Gt[i][0], VFMA(be[1], ad_Gt[i][1], VMUL(be[2], ad_Gt[i][2])));",
        "    VEC t2 = _zero;",
        "    for (int j = 0; j < 3; j++) t2 = VFMA(CalGt[j], d_be[i][j], t2);",
        "    t2 = VSUB(_zero, t2);",
        "    VEC t3 = VMUL(VSET(2.0/3.0), VMUL(CalGt[i], div_be));",
        "    VEC t4 = _zero;",
        "    for (int j = 0; j < 3; j++)",
        "        for (int k = 0; k < 3; k++)",
        "            t4 = VADD(t4, VADD(VMUL(igt[j][k], d2_be[i][j][k]), VMUL(VMUL(igt[i][j], d2_be[k][j][k]), _inv3)));",
        "    VEC t5 = _zero;",
        "    for (int j = 0; j < 3; j++) t5 = VFMA(At_UU[i][j], d_al[j], t5);",
        "    t5 = VMUL(VSET(-2.0), t5);",
        "    VEC t6 = _zero;",
        "    for (int j = 0; j < 3; j++)",
        "        for (int k = 0; k < 3; k++)",
        "            t6 = VFMA(C2[i][j][k], At_UU[j][k], t6);",
        "    t6 = VMUL(VMUL(_two, al), t6);",
        "    VEC t7 = _zero;",
        "    for (int j = 0; j < 3; j++)",
        "        t7 = VADD(t7, VADD(VMUL(VMUL(VSET(3.0), chi_inv), VMUL(At_UU[i][j], d_ch[j])),",
        "                            VMUL(VSET(4.0/3.0), VMUL(igt[i][j], d_K[j]))));",
        "    t7 = VMUL(VSUB(_zero, al), t7);",
        "    Gt_rhs_v[i] = VADD(VADD(VADD(t1,t2), VADD(t3,t4)), VADD(VADD(t5,t6), t7));",
        "}",
        "VSTORE(Gt_rhs0+pp, Gt_rhs_v[0]);",
        "VSTORE(Gt_rhs1+pp, Gt_rhs_v[1]);",
        "VSTORE(Gt_rhs2+pp, Gt_rhs_v[2]);",
    ]

    # B_rhs
    if gauge == "standard":
        lines += [
            "{",
            "    double * const B_rhs_arr[3] = {B_rhs0, B_rhs1, B_rhs2};",
            "    for (int i = 0; i < 3; i++) {",
            "        VEC adB = VFMA(be[0], ad_B[i][0], VFMA(be[1], ad_B[i][1], VMUL(be[2], ad_B[i][2])));",
            "        VEC adGt = VFMA(be[0], ad_Gt[i][0], VFMA(be[1], ad_Gt[i][1], VMUL(be[2], ad_Gt[i][2])));",
            "        VEC out = VADD(VSUB(Gt_rhs_v[i], VMUL(eta, Bv[i])),",
            "                        VSUB(VMUL(VSET((double)lambda[2]), adB), VMUL(VSET((double)lambda[3]), adGt)));",
            "        VSTORE(B_rhs_arr[i]+pp, out);",
            "    }",
            "}",
        ]
    else:  # rochester: B_rhs = 0
        lines += [
            "// Rochester gauge: B_rhs = 0",
            "{",
            "    double * const B_rhs_arr[3] = {B_rhs0, B_rhs1, B_rhs2};",
            "    const VEC _zero_b = VSET(0.0);",
            "    VSTORE(B_rhs_arr[0]+pp, _zero_b);",
            "    VSTORE(B_rhs_arr[1]+pp, _zero_b);",
            "    VSTORE(B_rhs_arr[2]+pp, _zero_b);",
            "}",
        ]

    return lines


def generate_cascade_avx2(config=None):
    """Generate AVX2-batched cascade body for include'ing into a per-batch harness wrapper.

    The wrapper is expected to:
      - Provide array pointers (alpha, beta0..2, ..., a_rhs, chi_rhs, ...)
      - Pack deriv pointers into arrays d_al_p[3], d_be_p[3][3], d_gt_p[6][3], d_At_p[6][3],
        d2_al_p[6], d2_ch_p[6], d2_be_p[3][6], d2_gt_p[6][6]  (if fused_derivs=False)
        or d2_al_p[6], d2_ch_p[6], d2_be_p[3][6], d2_gt_p[6][6]  (if fused_derivs=True,
        only mixed 2nd derivs needed at runtime)
      - Provide a `cascade_sym_idx(i,j)` inline helper (sym 3x3 -> 0..5)
      - Provide an integer `pp` (starting offset for the 4-batch) and a `VEC eta`
      - Provide `const unsigned int lambda[4]` and `const double lambda_f[2]`
      - If fused_derivs: provide `nx`, `ny`, `hx`, `hy`, `hz` in scope
    """
    config = config or {}
    fused_derivs = config.get("fused_derivs", False)
    simd = config.get("simd", "avx2")   # "avx2" or "avx512"
    lanes = 8 if simd == "avx512" else 4
    macros = _AVX512_MACROS_HEADER if simd == "avx512" else _AVX_MACROS_HEADER

    avx_typedef = "__m512d" if simd == "avx512" else "__m256d"

    lines = [
        macros,
        f"// BSSN RHS via polynomial cascade, {simd.upper()}-batched ({lanes} grid points per batch)",
        f"// Generated by cascade_codegen.py -- do not edit{'  [FUSED-DERIVS]' if fused_derivs else ''}",
        "// Included inside a harness wrapper that provides pointer setup + outer loops.",
        "{",
        f"    typedef {avx_typedef} VEC;  // scoped to this body; coexists with other-width VEC in same TU",
    ]

    body = []
    if fused_derivs:
        body += _comment("--- Stencil coefficient constants (per block) ---")
        body += _AVX_STENCIL_COEF_SETUP
        body += [""]
    body += emit_unpack_state()
    body += [""]
    body += emit_unpack_derivs(fused_derivs=fused_derivs)
    body += [""]
    body += emit_L1()
    body += [""]
    body += emit_L2()
    body += [""]
    body += emit_L3()
    body += [""]
    body += emit_L4()
    body += [""]
    body += emit_L5()
    body += [""]
    body += emit_L6()
    body += [""]
    body += emit_L7(config)

    lines += _indent(body)
    lines.append("}")

    return "\n".join(lines) + "\n"


def generate_cascade_cuda(config=None):
    """Generate a CUDA __device__ function for the cascade RHS.

    This emits a single fused kernel function that computes all 24 RHS
    outputs per thread, replacing the 8 separate __compute_*_rhs calls.
    Uses the unified VEC-macro emitters with a trivial scalar macros header
    (CUDA threads operate on scalar doubles).
    """
    config = config or {}

    lines = [
        _SCALAR_MACROS_HEADER,
        "// BSSN RHS cascade -- CUDA fused kernel",
        "// Generated by cascade_codegen.py -- do not edit",
        "//",
        "// Call this instead of the 8 separate __compute_*_rhs functions.",
        "// Each thread computes all 24 RHS outputs for one grid point.",
        "",
        "__device__ void __compute_cascade_rhs(",
        "    double **__unzipOutVar,",
        "    const double **__unzipInVar,",
        "    MemoryDerivs* __derivWorkspace,",
        "    const cuda::_Block* dblock,",
        "    const unsigned int * __gpuBlockMap,",
        "    const cuda::BSSNComputeParams * __bssnParams,",
        "    const cudaDeviceProp* __deviceProperties,",
        "    double* __sm_base,",
        "    unsigned int stream_id)",
        "{",
    ]

    # Kernel boilerplate: extract block info
    cuda_setup = [
        "const unsigned int offset = dblock->getOffset();",
        "const unsigned int *sz = dblock->getSz();",
        "const unsigned int nx = sz[0];",
        "const unsigned int ny = sz[1];",
        "const unsigned int nz = sz[2];",
        "const unsigned int PW = dblock->getBflag() ? 0 : 3;  // padding",
        "",
        "// Parameters",
        "const double lambda[4] = {__bssnParams->BSSN_LAMBDA[0], __bssnParams->BSSN_LAMBDA[1],",
        "                          __bssnParams->BSSN_LAMBDA[2], __bssnParams->BSSN_LAMBDA[3]};",
        "const double lambda_f[2] = {__bssnParams->BSSN_LAMBDA_F[0], __bssnParams->BSSN_LAMBDA_F[1]};",
        "const double eta = __bssnParams->ETA_CONST;",
        "",
        "// Thread-to-point mapping",
        "const unsigned int tx = threadIdx.x;",
        "const unsigned int ty = threadIdx.y;",
        "",
        "// Iterate over tiles in z",
        "for (unsigned int iz = PW; iz < nz - PW; iz++) {",
        "    const unsigned int ix = tx + PW;",
        "    const unsigned int iy = ty + PW;",
        "    if (ix >= nx - PW || iy >= ny - PW) continue;",
        "",
        "    const unsigned int pp = ix + nx * (iy + ny * iz);",
        "",
        "    // --- Pointer aliases into unzipped arrays ---",
        "    const double *const alpha = &__unzipInVar[VAR::U_ALPHA][offset];",
        "    const double *const chi   = &__unzipInVar[VAR::U_CHI][offset];",
        "    const double *const K     = &__unzipInVar[VAR::U_K][offset];",
        "    const double *const gt0   = &__unzipInVar[VAR::U_SYMGT0][offset];",
        "    const double *const gt1   = &__unzipInVar[VAR::U_SYMGT1][offset];",
        "    const double *const gt2   = &__unzipInVar[VAR::U_SYMGT2][offset];",
        "    const double *const gt3   = &__unzipInVar[VAR::U_SYMGT3][offset];",
        "    const double *const gt4   = &__unzipInVar[VAR::U_SYMGT4][offset];",
        "    const double *const gt5   = &__unzipInVar[VAR::U_SYMGT5][offset];",
        "    const double *const beta0 = &__unzipInVar[VAR::U_BETA0][offset];",
        "    const double *const beta1 = &__unzipInVar[VAR::U_BETA1][offset];",
        "    const double *const beta2 = &__unzipInVar[VAR::U_BETA2][offset];",
        "    const double *const At0   = &__unzipInVar[VAR::U_SYMAT0][offset];",
        "    const double *const At1   = &__unzipInVar[VAR::U_SYMAT1][offset];",
        "    const double *const At2   = &__unzipInVar[VAR::U_SYMAT2][offset];",
        "    const double *const At3   = &__unzipInVar[VAR::U_SYMAT3][offset];",
        "    const double *const At4   = &__unzipInVar[VAR::U_SYMAT4][offset];",
        "    const double *const At5   = &__unzipInVar[VAR::U_SYMAT5][offset];",
        "    const double *const Gt0   = &__unzipInVar[VAR::U_GT0][offset];",
        "    const double *const Gt1   = &__unzipInVar[VAR::U_GT1][offset];",
        "    const double *const Gt2   = &__unzipInVar[VAR::U_GT2][offset];",
        "    const double *const B0    = &__unzipInVar[VAR::U_B0][offset];",
        "    const double *const B1    = &__unzipInVar[VAR::U_B1][offset];",
        "    const double *const B2    = &__unzipInVar[VAR::U_B2][offset];",
        "",
        "    double *const a_rhs      = &__unzipOutVar[VAR::U_ALPHA][offset];",
        "    double *const chi_rhs    = &__unzipOutVar[VAR::U_CHI][offset];",
        "    double *const K_rhs      = &__unzipOutVar[VAR::U_K][offset];",
        "    double *const gt_rhs00   = &__unzipOutVar[VAR::U_SYMGT0][offset];",
        "    double *const gt_rhs01   = &__unzipOutVar[VAR::U_SYMGT1][offset];",
        "    double *const gt_rhs02   = &__unzipOutVar[VAR::U_SYMGT2][offset];",
        "    double *const gt_rhs11   = &__unzipOutVar[VAR::U_SYMGT3][offset];",
        "    double *const gt_rhs12   = &__unzipOutVar[VAR::U_SYMGT4][offset];",
        "    double *const gt_rhs22   = &__unzipOutVar[VAR::U_SYMGT5][offset];",
        "    double *const b_rhs0     = &__unzipOutVar[VAR::U_BETA0][offset];",
        "    double *const b_rhs1     = &__unzipOutVar[VAR::U_BETA1][offset];",
        "    double *const b_rhs2     = &__unzipOutVar[VAR::U_BETA2][offset];",
        "    double *const At_rhs00   = &__unzipOutVar[VAR::U_SYMAT0][offset];",
        "    double *const At_rhs01   = &__unzipOutVar[VAR::U_SYMAT1][offset];",
        "    double *const At_rhs02   = &__unzipOutVar[VAR::U_SYMAT2][offset];",
        "    double *const At_rhs11   = &__unzipOutVar[VAR::U_SYMAT3][offset];",
        "    double *const At_rhs12   = &__unzipOutVar[VAR::U_SYMAT4][offset];",
        "    double *const At_rhs22   = &__unzipOutVar[VAR::U_SYMAT5][offset];",
        "    double *const Gt_rhs0    = &__unzipOutVar[VAR::U_GT0][offset];",
        "    double *const Gt_rhs1    = &__unzipOutVar[VAR::U_GT1][offset];",
        "    double *const Gt_rhs2    = &__unzipOutVar[VAR::U_GT2][offset];",
        "    double *const B_rhs0     = &__unzipOutVar[VAR::U_B0][offset];",
        "    double *const B_rhs1     = &__unzipOutVar[VAR::U_B1][offset];",
        "    double *const B_rhs2     = &__unzipOutVar[VAR::U_B2][offset];",
        "",
        "    // Derivative pointers from pre-computed workspace",
        "    // (these are filled by __compute_derivatives before this call)",
        "    const unsigned int BLK_SZ = nx * ny * nz;",
        "    double *const deriv_base = __derivWorkspace->getBase(stream_id);",
        "#include \"bssnrhs_evar_derivs.h\"",
        "",
    ]

    # The cascade body is the same as CPU (unified VEC-macro emitters).
    # CUDA threads are scalar so fused_derivs stays off here.
    body = []
    body += emit_unpack_state()
    body += [""]
    body += emit_unpack_derivs(fused_derivs=False)
    body += [""]
    body += emit_L1()
    body += [""]
    body += emit_L2()
    body += [""]
    body += emit_L3()
    body += [""]
    body += emit_L4()
    body += [""]
    body += emit_L5()
    body += [""]
    body += emit_L6()
    body += [""]
    body += emit_L7(config)

    lines += _indent(cuda_setup)
    lines += _indent(_indent(body))
    lines += _indent([""] + ["} // end z-loop"])
    lines.append("}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Generate cascade BSSN RHS code")
    parser.add_argument("--target", choices=["cpu", "cuda", "avx2", "avx512", "both"], default="cpu")
    parser.add_argument("--gauge", choices=["standard", "rochester"], default="standard")
    parser.add_argument("--eta", choices=["const", "func", "RIT"], default="const")
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--cahd", action="store_true")
    parser.add_argument("--unroll", action="store_true", help="Fully unroll all tensor loops into scalar code")
    parser.add_argument("--fused-derivs", action="store_true",
                        help="Inline FD stencils for 1st and pure 2nd derivs (reduces memory workspace)")
    parser.add_argument("--fuse-mixed", action="store_true",
                        help="Also inline mixed 2nd derivs via 36-term tensor-product stencil (slower in practice — 1188 extra muls/point)")
    parser.add_argument("--output", default="generated")
    args = parser.parse_args()

    config = {
        "gauge": args.gauge,
        "eta_mode": args.eta,
        "ssl": args.ssl,
        "cahd": args.cahd,
        "fused_derivs": args.fused_derivs,
        "fuse_mixed_derivs": args.fuse_mixed,
    }

    os.makedirs(args.output, exist_ok=True)

    if args.target in ("cpu", "both"):
        if args.unroll:
            code = generate_cascade_cpu_unrolled(config)
        else:
            code = generate_cascade_cpu(config)
        suffix_parts = []
        if args.ssl:
            suffix_parts.append("SSL")
        if args.cahd:
            suffix_parts.append("HD")
        if args.fused_derivs:
            suffix_parts.append("allfused" if args.fuse_mixed else "fused")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        fname = os.path.join(args.output, f"bssneqs_cascade{suffix}.cpp")
        with open(fname, "w") as f:
            f.write(code)
        print(f"Wrote CPU cascade: {fname}")

    if args.target in ("avx2", "both"):
        config["simd"] = "avx2"
        code = generate_cascade_avx2(config)
        avx_suffix = "_fused" if args.fused_derivs else ""
        fname = os.path.join(args.output, f"bssneqs_cascade_avx{avx_suffix}.cpp")
        with open(fname, "w") as f:
            f.write(code)
        print(f"Wrote AVX2 cascade: {fname}")

    if args.target in ("avx512", "both"):
        config["simd"] = "avx512"
        code = generate_cascade_avx2(config)
        avx_suffix = "_fused" if args.fused_derivs else ""
        fname = os.path.join(args.output, f"bssneqs_cascade_avx512{avx_suffix}.cpp")
        with open(fname, "w") as f:
            f.write(code)
        print(f"Wrote AVX-512 cascade: {fname}")

    if args.target in ("cuda", "both"):
        code = generate_cascade_cuda(config)
        fname = os.path.join(args.output, "rhs_cascade.cu")
        with open(fname, "w") as f:
            f.write(code)
        print(f"Wrote CUDA cascade: {fname}")
