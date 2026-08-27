"""cascade_vec_printer.py -- SymPy → VEC-macro C++ printer.

Translates a SymPy expression tree into nested VMUL/VADD/VSUB/VFMA/VDIV/VSET
calls, so the same emitted body compiles under SCALAR / AVX2 / AVX-512 macro
headers from cascade_common.py without source changes.

Used by cascade_emit when --simd is avx2 / avx512 (and useful for scalar too,
since SCALAR_MACROS_HEADER defines VMUL(a,b) = a*b etc.).
"""

from sympy import Add, Mul, Pow, Rational, Integer, Float, Number, S, Symbol
from sympy.printing.codeprinter import CodePrinter


class VecPrinter(CodePrinter):
    """Emit SymPy expressions as nested VEC-macro calls.

    Output uses VSET(c) for scalar constants, VMUL/VADD/VSUB/VDIV/VFMA for
    arithmetic. Leaves (Symbol leaves like `gt0[pp]`, `igt00`, `grad_0_alpha[pp]`)
    must already be VEC-typed in the surrounding C++ scope.
    """

    printmethod = "_vec"
    language = "C"
    _default_settings = dict(CodePrinter._default_settings)
    _default_settings.update({"order": None, "reserved_word_suffix": "_"})

    # ----- numbers -----

    def _print_Integer(self, expr):
        return f"VSET({int(expr)}.0)"

    def _print_Float(self, expr):
        return f"VSET({float(expr)!r})"

    def _print_Rational(self, expr):
        return f"VSET(({int(expr.p)}.0/{int(expr.q)}.0))"

    def _print_NumberSymbol(self, expr):
        return f"VSET({float(expr)!r})"

    def _print_Symbol(self, expr):
        return str(expr.name)

    # ----- arithmetic -----

    def _is_negative_term(self, t):
        """t is `-something`? Either a negative Number, or Mul whose leading
        coefficient is negative."""
        if t.is_Number:
            return t.is_negative
        if t.is_Mul:
            c = t.args[0]
            return c.is_Number and c.is_negative
        return False

    def _negate_term(self, t):
        """Return -t with the leading sign stripped."""
        if t.is_Number:
            return -t
        if t.is_Mul:
            c = t.args[0]
            if c.is_Number and c.is_negative:
                rest = t.args[1:]
                if c == S.NegativeOne:
                    if len(rest) == 1:
                        return rest[0]
                    return Mul(*rest, evaluate=False)
                return Mul(-c, *rest, evaluate=False)
        return -t

    def _fold_vadd_vsub(self, pos_parts, neg_parts):
        """Build VADD/VSUB from already-printed pieces."""
        if not pos_parts and not neg_parts:
            return "VSET(0.0)"
        if not pos_parts:
            # all negative: VSUB(0, sum)
            inner = self._fold_left("VADD", neg_parts)
            return f"VSUB(VSET(0.0), {inner})"
        s = self._fold_left("VADD", pos_parts)
        for n in neg_parts:
            s = f"VSUB({s}, {n})"
        return s

    @staticmethod
    def _fold_left(op, parts):
        """Left-fold a 2-arg op over a non-empty list of strings."""
        result = parts[0]
        for p in parts[1:]:
            result = f"{op}({result}, {p})"
        return result

    def _print_Add(self, expr):
        if getattr(self, "_emit_fma", False):
            return self._print_Add_fma(expr)
        pos, neg = [], []
        for t in expr.args:
            if self._is_negative_term(t):
                neg.append(self._print(self._negate_term(t)))
            else:
                pos.append(self._print(t))
        return self._fold_vadd_vsub(pos, neg)

    def _fma_classify(self, t):
        """Classify a (sign-stripped) term for FMA folding.

        Returns ('prod', mul_a, mul_b) for a fusable product a*b (so it can
        become VFMA/VFNMADD), or ('atom', s) for anything else (symbol, number,
        Pow, or a Mul carrying a division -- those stay VADD/VSUB operands)."""
        if t.is_Mul:
            has_div = any(a.is_Pow and a.exp.is_Number and a.exp.is_negative
                          for a in t.args)
            if not has_div and len(t.args) >= 2:
                args = list(t.args)
                m1 = self._print(args[0])
                rest = args[1:]
                m2 = (self._print(rest[0]) if len(rest) == 1
                      else self._print(Mul(*rest, evaluate=False)))
                return ("prod", m1, m2)
        return ("atom", self._print(t))

    def _fold_terms(self, pos, neg):
        """Fold classified positive/negative terms into one serial accumulator
        chain (VFMA/VFNMADD for products, VADD/VSUB for atoms). Seeds with a
        positive atom when available so the chain has no leading VADD(0, .)."""
        pos = list(pos)
        acc = None
        for i, item in enumerate(pos):
            if item[0] == "atom":
                acc = item[1]
                pos = pos[:i] + pos[i + 1:]
                break
        if acc is None:
            if pos:
                _, m1, m2 = pos[0]
                acc = f"VMUL({m1}, {m2})"
                pos = pos[1:]
            else:
                acc = "VSET(0.0)"
        for item in pos:
            acc = (f"VFMA({item[1]}, {item[2]}, {acc})" if item[0] == "prod"
                   else f"VADD({acc}, {item[1]})")
        for item in neg:
            acc = (f"VFNMADD({item[1]}, {item[2]}, {acc})" if item[0] == "prod"
                   else f"VSUB({acc}, {item[1]})")
        return acc

    @staticmethod
    def _balanced_vadd(parts):
        """Combine partial sums with a balanced VADD reduction tree (log-depth),
        so the join doesn't reintroduce a long serial chain."""
        parts = list(parts)
        while len(parts) > 1:
            nxt = [f"VADD({parts[i]}, {parts[i + 1]})"
                   for i in range(0, len(parts) - 1, 2)]
            if len(parts) % 2:
                nxt.append(parts[-1])
            parts = nxt
        return parts[0]

    def _print_Add_fma(self, expr):
        """FMA-aware sum. Every product term folds into a VFMA (positive) or
        VFNMADD (negative), so the source carries the fused ops explicitly --
        matching gcc -ffp-contract=fast but guaranteeing it under clang/icpx,
        which don't contract across intrinsic calls. Requires VFNMADD.

        With _fma_split = k > 1, a long sum is partitioned into k independent
        accumulator chains (round-robin, so each gets a similar product load)
        joined by a balanced VADD tree. A serial FMA chain of N terms is
        latency-bound (~N * fmadd_latency); k independent chains cut the
        critical path to ~N/k + log2(k), trading the idle FMA-throughput
        headroom (latency >> 1/throughput) for speed. Pure reassociation -- no
        system-specific knowledge; applies to any Add."""
        pos, neg = [], []
        for t in expr.args:
            if self._is_negative_term(t):
                neg.append(self._fma_classify(self._negate_term(t)))
            else:
                pos.append(self._fma_classify(t))
        k = getattr(self, "_fma_split", 1)
        nterms = len(pos) + len(neg)
        if k <= 1 or nterms < getattr(self, "_fma_split_min", 8):
            return self._fold_terms(pos, neg)
        # Round-robin into k buckets so chain lengths stay balanced; each bucket
        # is a self-contained signed partial sum, joined by a balanced tree.
        bpos = [[] for _ in range(k)]
        bneg = [[] for _ in range(k)]
        for i, item in enumerate(pos):
            bpos[i % k].append(item)
        for i, item in enumerate(neg):
            bneg[i % k].append(item)
        parts = [self._fold_terms(bpos[j], bneg[j])
                 for j in range(k) if bpos[j] or bneg[j]]
        return self._balanced_vadd(parts)

    def _print_Mul(self, expr):
        args = list(expr.args)
        # Detect leading -1: emit VSUB(0, rest) so callers can compose into VSUB.
        if args and args[0] == S.NegativeOne:
            rest = args[1:]
            if not rest:
                return "VSET(-1.0)"
            inner = self._print(Mul(*rest, evaluate=False)) if len(rest) > 1 else self._print(rest[0])
            return f"VSUB(VSET(0.0), {inner})"
        # Detect leading negative numeric: split sign so it composes via VSUB.
        if args and args[0].is_Number and args[0].is_negative:
            pos_coeff = -args[0]
            new_args = list(args)
            new_args[0] = pos_coeff
            inner = self._print(Mul(*new_args, evaluate=False))
            return f"VSUB(VSET(0.0), {inner})"
        # Detect division: split into numerator and denominator-Pow(_, -1) factors
        num_args = []
        den_args = []
        for a in args:
            if a.is_Pow and a.exp.is_Number and a.exp.is_negative:
                den_args.append(Pow(a.base, -a.exp, evaluate=False))
            else:
                num_args.append(a)
        if den_args:
            num = self._print(Mul(*num_args, evaluate=False)) if len(num_args) > 1 else (
                self._print(num_args[0]) if num_args else "VSET(1.0)"
            )
            den = self._print(Mul(*den_args, evaluate=False)) if len(den_args) > 1 else self._print(den_args[0])
            return f"VDIV({num}, {den})"
        parts = [self._print(a) for a in args]
        return self._fold_left("VMUL", parts)

    def _print_Function(self, expr):
        """Generic unary/binary functions: log, exp, sin, etc. become V<UPPER>(args).
        The caller must define a matching macro (VLOG, VEXP, etc.). The kernel
        emitter in cascade_emit.py provides VLOG/VEXP/VSQRT for the SIMD dialects.
        """
        fname = type(expr).__name__
        macro = "V" + fname.upper()
        args = ", ".join(self._print(a) for a in expr.args)
        return f"{macro}({args})"

    def _print_Pow(self, expr):
        base, exp = expr.base, expr.exp
        # Treat float exponents with integral value (e.g. chi**-2.0 from
        # external specs) as integers so they get the VMUL/VDIV expansion.
        if exp.is_Number and exp.is_Float and float(exp).is_integer():
            exp = Integer(int(float(exp)))
        if exp.is_Number and exp.is_Integer:
            n = int(exp)
            if n == 1:
                return self._print(base)
            if n == 2:
                b = self._print(base)
                return f"VMUL({b}, {b})"
            if n > 2:
                b = self._print(base)
                result = b
                for _ in range(n - 1):
                    result = f"VMUL({result}, {b})"
                return result
            if n == -1:
                return f"VDIV(VSET(1.0), {self._print(base)})"
            if n < 0:
                # 1 / base^|n|
                pos_pow = Pow(base, -exp, evaluate=False)
                return f"VDIV(VSET(1.0), {self._print(pos_pow)})"
        if exp.is_Number and exp == Rational(1, 2):
            return f"VSQRT({self._print(base)})"
        # half-integer exponents p/2 (|p| odd, e.g. -1/2, 3/2 from 1/sqrt(...) and
        # tetrad normalisations): x^(p/2) = sqrt(x) * x^((|p|-1)/2), reciprocal if p<0.
        # No vikr kernel reaches this branch (they contained no pow()), so the
        # checked-in kernels are unaffected.
        if exp.is_Number and exp.is_Rational and exp.q == 2:
            b = self._print(base)
            body = f"VSQRT({b})"
            for _ in range((abs(exp.p) - 1) // 2):
                body = f"VMUL({body}, {b})"
            return body if exp.p > 0 else f"VDIV(VSET(1.0), {body})"
        # anything else: per-lane pow via the VPOW macro (the wrapper's macro set
        # defines it, like VLOG/VEXP).
        return f"VPOW({self._print(base)}, {self._print(exp)})"

    # ----- CodePrinter scaffolding -----

    def _rate_index_position(self, p):
        return p

    def _get_statement(self, codestring):
        return f"{codestring};"

    def _get_comment(self, text):
        return f"// {text}"

    def _declare_number_const(self, name, value):
        return f"const VEC {name} = {self._print(value)};"

    def _format_code(self, lines):
        return lines


def to_vec_cpp(expr, fma: bool = False, split: int = 1) -> str:
    """One-shot helper: print a SymPy expression as a VEC-macro string.

    fma=True folds every product term into VFMA/VFNMADD at the tree level
    (robust superset of the regex fold_vfma; needs VFNMADD in the header).
    split=k>1 partitions long sums into k independent FMA accumulator chains
    joined by a balanced VADD tree (ILP/latency win; pure reassociation)."""
    p = VecPrinter()
    p._emit_fma = fma or split > 1
    p._fma_split = split
    return p.doprint(expr)


# Optional: rewrite-after pass that turns `VADD(VMUL(a, b), c)` into
# `VFMA(a, b, c)`. Saves one instruction on AVX2/AVX-512 with FMA. The
# compiler usually does this for us at -O3 with -mfma, but doing it in
# source guarantees it under any flags.

import re as _re

_VFMA_PAT = _re.compile(r"VADD\(\s*VMUL\(([^,()]*(?:\([^()]*\)[^,()]*)*),\s*([^,()]*(?:\([^()]*\)[^,()]*)*)\),\s*([^,()]*(?:\([^()]*\)[^,()]*)*)\)")


def fold_vfma(text: str, max_passes: int = 0) -> str:
    """Optional: fold VADD(VMUL(a,b), c) -> VFMA(a,b,c).

    Off by default (max_passes=0). With FMA-capable -O3 compilation, the
    compiler does this for us. Enable for source-level guarantee.
    """
    for _ in range(max_passes):
        new = _VFMA_PAT.sub(lambda m: f"VFMA({m.group(1)}, {m.group(2)}, {m.group(3)})", text)
        if new == text:
            break
        text = new
    return text
