"""cascade_common.py -- Backend-agnostic infrastructure for cascade codegen.

This module holds the pieces that are not tied to any particular PDE system
(BSSN, EMDA, etc.). A PDE-specific codegen module (e.g. cascade_codegen.py
for BSSN) imports from here and composes emit functions on top.

Contents:

    * Symmetric-tensor index helpers (3x3 -> 6 flat slots)
    * 6th-order centered finite-difference stencils (scalar and 4-wide SIMD)
    * Backend macro headers that map the VEC / VMUL / VFMA abstraction onto
      scalar double, AVX2 __m256d, or AVX-512 __m512d
    * Small code-emission utilities (indent, comment)

The scalar and SIMD paths share one emitter set via the macro headers; a
"scalar" build is produced by swapping in _SCALAR_MACROS_HEADER, which
defines VEC=double and VMUL(a,b)=a*b, etc.
"""


# ---------------------------------------------------------------------------
# Symmetric 3x3 tensor index helpers
# ---------------------------------------------------------------------------
# Flat slot layout: (0,0)->0 (0,1)->1 (0,2)->2 (1,1)->3 (1,2)->4 (2,2)->5
# This matches bench.cpp / bssn_jax.py / dendrogr convention.
_SYM_TBL = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
_SYM_NAMES = {0: "00", 1: "01", 2: "02", 3: "11", 4: "12", 5: "22"}


def sym(i, j):
    """Flat index for symmetric 3x3: (0,0)->0, (0,1)->1, ..., (2,2)->5."""
    return _SYM_TBL[min(i, j)][max(i, j)]


def sym_label(i, j):
    """String label for symmetric pair: (0,1) -> '01'."""
    return _SYM_NAMES[sym(i, j)]


# ---------------------------------------------------------------------------
# Finite-difference stencil data (6th-order centered, dendrogr deriv644)
# ---------------------------------------------------------------------------
#   f'  = [ -1, +9, -45,  0, +45, -9, +1 ] / 60
#   f'' = [  2,-27, 270,-490,270,-27, 2 ] / 180
# Stride for direction d: d=0 -> 1, d=1 -> nx, d=2 -> nx*ny
# The per-point loop must pre-compute idx60/idy60/idz60 = 1/(60 h) and
# idx2_180/idy2_180/idz2_180 = 1/(180 h^2). See _STENCIL_COEF_SETUP for the
# one-liner setup block.

_STRIDE_NAME = ["1", "nx", "(nx*ny)"]
_IDX_COEF_NAME = ["idx60", "idy60", "idz60"]
_IDX2_COEF_NAME = ["idx2_180", "idy2_180", "idz2_180"]

# 1st-derivative coefficients at offsets [-3, -2, -1, 0, +1, +2, +3]
# (center coeff is 0; mixed-stencil code skips it)
_D1_COEF = {-3: -1.0, -2: 9.0, -1: -45.0, 0: 0.0,
            1: 45.0, 2: -9.0, 3: 1.0}


# One-shot block setup: defines the per-block 1/h constants used by every
# stencil. Emit once at the top of each generated cascade body.
_STENCIL_COEF_SETUP = [
    "const double idx60    = (1.0/60.0)  / hx;",
    "const double idy60    = (1.0/60.0)  / hy;",
    "const double idz60    = (1.0/60.0)  / hz;",
    "const double idx2_180 = (1.0/180.0) / (hx*hx);",
    "const double idy2_180 = (1.0/180.0) / (hy*hy);",
    "const double idz2_180 = (1.0/180.0) / (hz*hz);",
]


# ---------------------------------------------------------------------------
# Scalar stencil emitters (used by the tensor-loop / unrolled CPU paths)
# ---------------------------------------------------------------------------

def stencil_grad(d, var):
    """6th-order centered 1st-derivative stencil expression (scalar)."""
    s = _STRIDE_NAME[d]
    coef = _IDX_COEF_NAME[d]
    if d == 0:
        return (f"(-{var}[pp-3] + 9.0*{var}[pp-2] - 45.0*{var}[pp-1] "
                f"+ 45.0*{var}[pp+1] - 9.0*{var}[pp+2] + {var}[pp+3]) * {coef}")
    return (f"(-{var}[pp-3*{s}] + 9.0*{var}[pp-2*{s}] - 45.0*{var}[pp-{s}] "
            f"+ 45.0*{var}[pp+{s}] - 9.0*{var}[pp+2*{s}] + {var}[pp+3*{s}]) * {coef}")


def stencil_grad2_pure(d, var):
    """6th-order centered pure 2nd-derivative stencil (scalar)."""
    s = _STRIDE_NAME[d]
    coef = _IDX2_COEF_NAME[d]
    if d == 0:
        return (f"(2.0*{var}[pp-3] - 27.0*{var}[pp-2] + 270.0*{var}[pp-1] - 490.0*{var}[pp] "
                f"+ 270.0*{var}[pp+1] - 27.0*{var}[pp+2] + 2.0*{var}[pp+3]) * {coef}")
    return (f"(2.0*{var}[pp-3*{s}] - 27.0*{var}[pp-2*{s}] + 270.0*{var}[pp-{s}] - 490.0*{var}[pp] "
            f"+ 270.0*{var}[pp+{s}] - 27.0*{var}[pp+2*{s}] + 2.0*{var}[pp+3*{s}]) * {coef}")


def _offset_expr(d, k):
    """Return C++ expression for offset along direction d by k steps.

    Examples:
      _offset_expr(0, 1)  -> "1"
      _offset_expr(0, -3) -> "-3"
      _offset_expr(1, 2)  -> "2*nx"
      _offset_expr(2, -3) -> "-3*(nx*ny)"
    """
    if k == 0:
        return "0"
    s = _STRIDE_NAME[d]
    if s == "1":
        return f"{k}"
    sign = "-" if k < 0 else ""
    mag = abs(k)
    if mag == 1:
        return f"{sign}{s}"
    return f"{sign}{mag}*{s}"


def stencil_grad2_mixed(d1, d2, var):
    """6th-order centered MIXED 2nd-derivative stencil (d/d_d1 d/d_d2, d1 != d2).

    Tensor product of two 1st-deriv stencils. Center coefficient is zero, so
    the 7x7 grid collapses to 36 nonzero terms.

    Requires `idxy_3600`, `idxz_3600`, `idyz_3600` = 1/(3600 h1 h2) in scope.
    """
    if d1 > d2:
        d1, d2 = d2, d1
    pair = f"id{'xyz'[d1]}{'xyz'[d2]}_3600"

    terms = []
    for i in [-3, -2, -1, 1, 2, 3]:
        ci = _D1_COEF[i]
        for j in [-3, -2, -1, 1, 2, 3]:
            cj = _D1_COEF[j]
            coef = ci * cj
            if coef == 0.0:
                continue
            off_d1 = _offset_expr(d1, i)
            off_d2 = _offset_expr(d2, j)
            if off_d1 == "0":
                off_expr = off_d2
            elif off_d2 == "0":
                off_expr = off_d1
            else:
                if off_d2.startswith("-"):
                    off_expr = f"{off_d1}{off_d2}"
                else:
                    off_expr = f"{off_d1}+{off_d2}"
            if off_expr == "0":
                access = f"{var}[pp]"
            else:
                access = f"{var}[pp+({off_expr})]"
            if coef > 0:
                terms.append(f"+{coef:g}*{access}")
            else:
                terms.append(f"{coef:g}*{access}")
    body = " ".join(terms)
    if body.startswith("+"):
        body = body[1:]
    return f"({body}) * {pair}"


# ---------------------------------------------------------------------------
# SIMD stencil emitters (used by the AVX2 / AVX-512 path)
# ---------------------------------------------------------------------------
# Factored FMA form: cuts the multiply count by using antisymmetric / symmetric
# pairings. Same math as the scalar stencils -- compiler should produce the
# same result to the bit, modulo FP reassociation, which has been verified
# against the reference bench_avx.cpp.

_SIMD_STENCIL_STRIDE = {0: "1", 1: "nx", 2: "(nx*ny)"}
_SIMD_STENCIL_IDX_1ST = ["idx60", "idy60", "idz60"]
_SIMD_STENCIL_IDX_2ND = ["idx2_180", "idy2_180", "idz2_180"]


def simd_stencil_1st(d, var):
    """4-/8-wide SIMD expression for 6th-order centered 1st deriv of `var`.

    Antisymmetric factored form: half the multiplies vs. naive.
    """
    coef = _SIMD_STENCIL_IDX_1ST[d]

    def ofs(k):
        if d == 0:
            if k == 0:
                return "pp"
            return f"pp{'+' if k > 0 else ''}{k}"
        s = _SIMD_STENCIL_STRIDE[d]
        if k == 0:
            return "pp"
        return f"pp{'+' if k > 0 else ''}{k}*{s}"

    return (
        f"VMUL(VSET({coef}), "
        f"VFMA(VSET(45.0), VSUB(VLOAD({var}+{ofs(1)}), VLOAD({var}+{ofs(-1)})), "
        f"VFMA(VSET(-9.0), VSUB(VLOAD({var}+{ofs(2)}), VLOAD({var}+{ofs(-2)})), "
        f"VSUB(VLOAD({var}+{ofs(3)}), VLOAD({var}+{ofs(-3)})))))"
    )


def simd_stencil_2nd_pure(d, var):
    """SIMD expression for 6th-order centered pure 2nd deriv of `var`.

    Symmetric factored form: 4 FMAs (the paired sums + the -490 center).
    """
    coef = _SIMD_STENCIL_IDX_2ND[d]
    s = _SIMD_STENCIL_STRIDE[d]

    def ofs(k):
        if d == 0:
            if k == 0:
                return "pp"
            return f"pp{'+' if k > 0 else ''}{k}"
        if k == 0:
            return "pp"
        return f"pp{'+' if k > 0 else ''}{k}*{s}"

    return (
        f"VMUL(VSET({coef}), "
        f"VFMA(VSET(2.0), VADD(VLOAD({var}+{ofs(-3)}), VLOAD({var}+{ofs(3)})), "
        f"VFMA(VSET(-27.0), VADD(VLOAD({var}+{ofs(-2)}), VLOAD({var}+{ofs(2)})), "
        f"VFMA(VSET(270.0), VADD(VLOAD({var}+{ofs(-1)}), VLOAD({var}+{ofs(1)})), "
        f"VMUL(VSET(-490.0), VLOAD({var}+pp))))))"
    )


# ---------------------------------------------------------------------------
# Backend macro headers
# ---------------------------------------------------------------------------
# Emit one of these at the top of the generated cascade body. The emitter
# functions write VEC/VMUL/VFMA/... calls; the macro header picks the
# concrete width (scalar double, 4-wide AVX2, or 8-wide AVX-512).

SCALAR_MACROS_HEADER = """\
// ---------- Scalar macros (trivial VEC = double wrappers) ----------
// Lets the unified cascade body compile as plain scalar code. Same source
// can be compiled as AVX2 / AVX-512 by swapping macro headers.
#include <cmath>
#undef VSET
#undef VLOAD
#undef VSTORE
#undef VADD
#undef VSUB
#undef VMUL
#undef VDIV
#undef VFMA
#undef VSQRT
#define VSET(x)      ((double)(x))
#define VLOAD(p)     (*(p))
#define VSTORE(p,v)  (*(p) = (v))
#define VADD(a,b)    ((a) + (b))
#define VSUB(a,b)    ((a) - (b))
#define VMUL(a,b)    ((a) * (b))
#define VDIV(a,b)    ((a) / (b))
#define VFMA(a,b,c)  ((a) * (b) + (c))
#define VSQRT(a)     (std::sqrt(a))
// -------------------------------------------------------------------
"""

AVX_MACROS_HEADER = """\
// ---------- AVX2 macros (4 doubles per vector) ----------
// Designed to coexist with AVX-512 variant in same TU by #undef'ing names
// before redefinition; typedef VEC is scoped to the body {} below.
#include <immintrin.h>
#undef VSET
#undef VLOAD
#undef VSTORE
#undef VADD
#undef VSUB
#undef VMUL
#undef VDIV
#undef VFMA
#undef VSQRT
#define VSET(x)      _mm256_set1_pd(x)
#define VLOAD(p)     _mm256_loadu_pd(p)
#define VSTORE(p,v)  _mm256_storeu_pd((p),(v))
#define VADD(a,b)    _mm256_add_pd((a),(b))
#define VSUB(a,b)    _mm256_sub_pd((a),(b))
#define VMUL(a,b)    _mm256_mul_pd((a),(b))
#define VDIV(a,b)    _mm256_div_pd((a),(b))
#define VFMA(a,b,c)  _mm256_fmadd_pd((a),(b),(c))
#define VSQRT(a)     _mm256_sqrt_pd(a)
// --------------------------------------------------------
"""

AVX512_MACROS_HEADER = """\
// ---------- AVX-512 macros (8 doubles per vector) ----------
// Designed to coexist with AVX2 variant in same TU by #undef'ing names
// before redefinition; typedef VEC is scoped to the body {} below.
#include <immintrin.h>
#undef VSET
#undef VLOAD
#undef VSTORE
#undef VADD
#undef VSUB
#undef VMUL
#undef VDIV
#undef VFMA
#undef VSQRT
#define VSET(x)      _mm512_set1_pd(x)
#define VLOAD(p)     _mm512_loadu_pd(p)
#define VSTORE(p,v)  _mm512_storeu_pd((p),(v))
#define VADD(a,b)    _mm512_add_pd((a),(b))
#define VSUB(a,b)    _mm512_sub_pd((a),(b))
#define VMUL(a,b)    _mm512_mul_pd((a),(b))
#define VDIV(a,b)    _mm512_div_pd((a),(b))
#define VFMA(a,b,c)  _mm512_fmadd_pd((a),(b),(c))
#define VSQRT(a)     _mm512_sqrt_pd(a)
// -----------------------------------------------------------
"""


def macros_header(width):
    """Return the macros header for a given SIMD width name.

    width: one of "scalar", "avx2", "avx512".
    """
    if width == "scalar":
        return SCALAR_MACROS_HEADER
    if width == "avx2":
        return AVX_MACROS_HEADER
    if width == "avx512":
        return AVX512_MACROS_HEADER
    raise ValueError(f"unknown SIMD width: {width!r}")


# ---------------------------------------------------------------------------
# Code emission utilities
# ---------------------------------------------------------------------------

def indent(lines, n=1):
    """Indent a list of code lines by n levels (4 spaces each)."""
    prefix = "    " * n
    return [prefix + l for l in lines]


def comment(text):
    """Return a single-line C++ comment as a one-element list."""
    return [f"// {text}"]


# Backwards-compatible private aliases (older call sites used the leading
# underscore). New code should prefer the public names above.
_indent = indent
_comment = comment
