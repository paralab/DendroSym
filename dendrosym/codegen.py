"""codegen.py

This file contains the functions that generate the C++ or CUDA code
that can be used by Dendro to run the programmed simulations.

TODO: Please note that there are a few functions missing from the original
dendro.py script currently. This includes the GPU code and other small
currently unused functions. They will be added soon.
"""

# import enum
import heapq
import re as regex
import sys
from typing import List, Tuple, Union

import sympy as sym

# from sympy.core.evalf import N
# from sympy.core.symbol import var
# from sympy.utilities.iterables import uniq
from dendrosym import nr
import dendrosym


def extract_expression(expression, is_symmetric_matrix=True):
    """This extracts out individual expressions

    This is meant to be used with other functions to try and pull out
    individual variables from matrices and vectors and lists
    """

    mi = [0, 1, 2, 4, 5, 8]

    num_e = 0
    list_expressions = []
    # check the typing
    if type(expression) == list:
        num_e += len(expression)
        for ii, exp_individual in enumerate(expression):
            list_expressions.append(exp_individual)
    elif type(expression) == sym.Matrix:
        # 1D matrix from sympy using our 3vec notation
        if expression.shape[1] == 1:
            num_e += len(expression)
            for ii in range(expression.shape[0]):
                list_expressions.append(expression[ii])
        # otherwise it's a matrix
        else:
            if is_symmetric_matrix:
                # NOTE: original method, does *not* check for symmetry
                for (
                    j,
                    k,
                ) in enumerate(mi):
                    list_expressions.append(sym.sympify(expression[k]))
                    num_e += 1
            else:
                for j, k in expression.shape:
                    list_expressions.append(sym.sympify(expression[j, k]))
                    num_e += 1

            # NOTE: my implementation if there's symmetry is currently
            # really slow and broken, need to consider more info...
            # check for matrix symmetry
            # if sym.simplify(e) == sym.simplify(e.T):
            #     # print("Symmetric matrix found")
            #     for jj in range(e.shape[0]):
            #         for kk in range(jj, e.shape[1]):
            #             lname.append(vnames[i] + repr(jj) + repr(kk) + idx)
            #             lexp.append(e[jj, kk])
            # else:
            #     for jj in range(e.shape[0]):
            #         for kk in range(e.shape[1]):
            #             lname.append(vnames[i] + repr(jj) + repr(kk) + idx)
            #             lexp.append(e[jj, kk])

    elif type(expression) == float or type(expression) == int:
        num_e += 1
        list_expressions.append(sym.sympify(expression))

    else:
        num_e += 1
        list_expressions.append(expression)

    return list_expressions, num_e


def construct_expression_list(
    ex: Union[list, sym.Matrix, sym.Expr], vnames: List[str], idx: str = "[pp]"
):
    # NOTE: there seems to be an issue with the symmetric stuff
    mi = [0, 1, 2, 4, 5, 8]
    midx = ["00", "01", "02", "11", "12", "22"]

    # total number of expressions
    # print("--------------------------------------------------------")

    num_e = 0
    lexp = []
    lname = []
    for i, e in enumerate(ex):
        if type(e) == list:
            num_e = num_e + len(e)
            for j, ev in enumerate(e):
                lexp.append(ev)
                lname.append(vnames[i] + repr(j) + idx)
        elif type(e) == sym.Matrix:
            num_e = num_e + len(e)

            # NOTE: Original
            for j, k in enumerate(mi):
                lexp.append(e[k])
                lname.append(vnames[i] + midx[j] + idx)

            # NOTE: my implementation if there's symmetry is currently
            # really slow and broken, need to consider more info...
            # check for matrix symmetry
            # if sym.simplify(e) == sym.simplify(e.T):
            #     # print("Symmetric matrix found")
            #     for jj in range(e.shape[0]):
            #         for kk in range(jj, e.shape[1]):
            #             lname.append(vnames[i] + repr(jj) + repr(kk) + idx)
            #             lexp.append(e[jj, kk])
            # else:
            #     for jj in range(e.shape[0]):
            #         for kk in range(e.shape[1]):
            #             lname.append(vnames[i] + repr(jj) + repr(kk) + idx)
            #             lexp.append(e[jj, kk])
        else:
            num_e = num_e + 1
            lexp.append(e)
            lname.append(vnames[i] + idx)

    return lexp, lname, num_e


def custom_numbered_symbols(prefix="DENDRO_", start=0, num_digits=4):
    while True:
        num_str = str(start).zfill(num_digits)
        name = f"{prefix}{num_str}"

        s = sym.Symbol(name)

        yield s

        start += 1


def construct_cse(
    ex: Union[list, sym.Matrix, sym.Expr], vnames: List[str], idx: str
) -> Tuple[list, int]:
    """Construct the common sub-expression ellimination tree

    TODO: detailed explanation

    Parameters
    ----------
    ex : list, sympy.Matrix, sympy.Expr
        The expression to parse for the corresponding varibles in the next
        parameter. e.g. (as Sympy expressions) [x + 1, y + x**2]
    vnames : list
        The variable names corresponding to each expression, e.g.
        ['alpha_rhs', 'beta_rhs']
    idx : str
        The string used for indexing. e.g. '[pp]'

    Returns
    -------


    """

    lexp, lname, num_e = construct_expression_list(ex, vnames, idx)

    # ''.join(random.choice(string.ascii_uppercase) for _ in range(5))
    ee_name = "DENDRO_"
    ee_syms = sym.numbered_symbols(prefix=ee_name)
    print("Now generating cse", file=sys.stderr)
    # optimizations=None (was "basic"): the basic preprocessor rewrites
    # neg-powers to division, often inflating tree size and slowing CSE.
    # order="none" skips canonical-arg ordering.
    _v = sym.cse(lexp, symbols=ee_syms, optimizations=None, order="none")
    print("Finished generating cse", file=sys.stderr)

    return _v, sym.count_ops(lexp)


def construct_cse_from_list(
    expression_list, temp_var_prefix="DENDRO_", ignore_symbols=[], optimizations=None
):
    temp_var_gen = custom_numbered_symbols(temp_var_prefix)

    print("Now generating cse!", file=sys.stderr)

    if optimizations is not None:
        print(
            "    WARNING: Optimizations are set, this could take a while!",
            file=sys.stderr,
        )

    # optimizations=None (was "basic"): the basic preprocessor rewrites
    # neg-powers to division, often inflating tree size before CSE runs.
    cse_out = sym.cse(
        expression_list,
        symbols=temp_var_gen,
        optimizations=None,
        order="none",
        ignore=ignore_symbols,
    )
    print("Finished generating cse!", file=sys.stderr)

    return cse_out


def generate_cpu_preextracted(
    cse_list,
    rhs_var_names,
    idx,
    orig_ops,
    dtype="double",
    use_const=False,
    return_stats=False,
    input_names=None,
    input_struct=None,
    interleave_outputs=False,
):
    custom_functions = {
        "grad": "grad",
        "grad2": "grad2",
        "agrad": "agrad",
        "kograd": "kograd",
    }

    cprinter = dendrosym.code_printer.DendroCPrinter(
        additional_user_funcs=custom_functions
    )

    output_str = "// Dendro: C++ Equation Code Generation {{{{ \n"

    # count_ops walks the full expression tree -- only do it when callers
    # explicitly need the stat (return_stats=True)
    want_stats = return_stats
    reduced_ops = 0

    # printer + change_deriv_names are now both batched: emit all lines into
    # output_str first, then run the regex pass once over the whole block
    # (the regex matches grad(...)/grad2(...) call sites independent of
    # surrounding context, so per-expression vs one-shot is equivalent).
    prefix = f"{'const ' if use_const else ''}{dtype} "

    if interleave_outputs:
        # Outputs that are also INPUTS to other statements (a staged block whose
        # quantities are defined in terms of each other) cannot use the
        # temporaries-then-outputs layout below: cse() hoists every temporary to
        # the front, so a temporary reading an output would be emitted above that
        # output's declaration and the block would not compile. Emit one flat,
        # dependency-ordered block instead. Ties keep the original index, so the
        # output stays deterministic and, when no output feeds another statement,
        # identical in content to the default layout.
        stmts = []  # (defined symbol or None, text, referenced symbols)
        for v1, v2 in cse_list[0]:
            stmts.append((v1, prefix + cprinter.doprint(v2, assign_to=v1) + "\n",
                          v2.free_symbols))
            if want_stats:
                reduced_ops += sym.count_ops(v2)
        for i, e in enumerate(cse_list[1]):
            name = str(rhs_var_names[i])
            # names arrive already carrying their declaration ("double DDth00")
            bare = name.split()[-1]
            stmts.append((sym.Symbol(bare),
                          "\n//--\n" + cprinter.doprint(e, assign_to=name + idx) + "\n",
                          e.free_symbols))
            if want_stats:
                reduced_ops += sym.count_ops(e)

        defined = {}
        for k, (dsym, _t, _r) in enumerate(stmts):
            if dsym is not None:
                defined.setdefault(dsym, k)
        children = [[] for _ in stmts]
        indeg = [0] * len(stmts)
        for k, (_d, _t, refs) in enumerate(stmts):
            deps = {defined[r] for r in refs if r in defined and defined[r] != k}
            indeg[k] = len(deps)
            for j in deps:
                children[j].append(k)

        ready = [k for k in range(len(stmts)) if indeg[k] == 0]
        heapq.heapify(ready)
        n_emitted = 0
        while ready:
            k = heapq.heappop(ready)
            output_str += stmts[k][1]
            n_emitted += 1
            for c in children[k]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(ready, c)
        if n_emitted != len(stmts):
            stuck = [str(stmts[k][0]) for k in range(len(stmts)) if indeg[k] > 0]
            raise ValueError(
                "cyclic dependency among the emitted statements; "
                f"{len(stmts) - n_emitted} unresolved, e.g. {stuck[:8]}"
            )
    else:
        output_str += "// Dendro: TEMPORARY VARIABLES\n"
        for v1, v2 in cse_list[0]:
            output_str += prefix + cprinter.doprint(v2, assign_to=v1) + "\n"
            if want_stats:
                reduced_ops += sym.count_ops(v2)

        output_str += "// Dendro: END TEMPORARY VARIABLES\n"
        output_str += "\n// Dendro: MAIN VARIABLES"
        for i, e in enumerate(cse_list[1]):
            output_str += "\n//--\n" + cprinter.doprint(e, assign_to=str(rhs_var_names[i]) + idx) + "\n"
            if want_stats:
                reduced_ops += sym.count_ops(e)

        output_str += "// Dendro: END MAIN VARIABLES\n\n"
    output_str = change_deriv_names(output_str)
    if input_struct and input_names:
        output_str = apply_input_struct(output_str, input_names, input_struct)

    if not return_stats:
        output_str += "// Dendro: }}}} End Code Generation \n"
        return output_str
    else:
        return output_str, reduced_ops


def generate_cpu(
    ex: Union[list, sym.Matrix, sym.Expr], vnames: List[str], idx: str
) -> Tuple[list, int]:
    """Generate the CPU C++ code by simplifying the expressions

    TODO: expand the documentation

    Parameters
    ----------
    ex : list, sympy.Matrix, sympy.Expr
        The expression to parse for the corresponding varibles in the next
        parameter. e.g. (as Sympy expressions) [x + 1, y + x**2]
    vnames : list
        The variable names corresponding to each expression, e.g.
        ['alpha_rhs', 'beta_rhs']
    idx : str
        The string used for indexing. e.g. '[pp]'
    """

    output_string = ""
    # print(ex)

    # total number of expressions
    # print("--------------------------------------------------------")

    lexp, lname, num_e = construct_expression_list(ex, vnames, idx)

    cse = construct_cse(ex, vnames, idx)
    _v = cse[0]

    output_string += "// Dendro: {{{ \n"
    output_string += "// Dendro: original ops: %d \n" % (cse[1])

    ee_name = "DENDRO_"
    ee_syms = sym.utilities.numbered_symbols(prefix=ee_name)

    custom_functions = {
        "grad": "grad",
        "grad2": "grad2",
        "agrad": "agrad",
        "kograd": "kograd",
    }

    cprinter = dendrosym.code_printer.DendroCPrinter(
        additional_user_funcs=custom_functions
    )

    rops = 0
    output_string += "// Dendro: printing temp variables\n"
    for v1, v2 in _v[0]:
        # TODO: add the potential for const???? They're not going to be modified, but might not be necessary
        temp_str = "double "

        temp_str += change_deriv_names(
            cprinter.doprint(v2, assign_to=v1)
            # sym.ccode(v2, assign_to=v1, user_functions=custom_functions)
        )
        output_string += temp_str + "\n"
        rops = rops + sym.count_ops(v2)

    output_string += "\n// Dendro: printing variables"
    for i, e in enumerate(_v[1]):
        output_string += "\n//--\n"
        output_string += change_deriv_names(
            cprinter.doprint(e, assign_to=lname[i])
            # sym.ccode(e, assign_to=lname[i], user_functions=custom_functions)
        )
        output_string += "\n"
        rops = rops + sym.count_ops(e)

    output_string += "// Dendro: reduced ops: %d \n" % (rops)
    output_string += "// Dendro: }}} \n"

    return output_string


_DERIV1_PAT = regex.compile(r"\b(agrad|grad|kograd)\((\d),\s*(\w+\[pp\])\)")
_DERIV2_PAT = regex.compile(r"\bgrad2\((\d),\s*(\d),\s*(\w+\[pp\])\)")


def _deriv1_repl(m):
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def _deriv2_repl(m):
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return f"grad2_{a}_{b}_{m.group(3)}"


def change_deriv_names(in_str: str) -> str:
    """Rewrite `grad(i, var[pp])` -> `grad_i_var[pp]` and the grad2 variant.

    Single-pass re.sub (was findall + per-match str.replace, O(N^2) on big
    CSE outputs).
    """
    out = _DERIV1_PAT.sub(_deriv1_repl, in_str)
    out = _DERIV2_PAT.sub(_deriv2_repl, out)
    return out


def apply_input_struct(code: str, input_names: list, struct: str) -> str:
    """Prefix bare input-variable reads `v` -> `struct.v` (e.g. `in.alpha`).

    Matches each field name as a whole token not preceded by a word char or '.'
    -- so it skips `grad_0_v`/`agrad_0_v` derivative buffers (preceded by '_')
    and `out.v` output writes (preceded by '.') -- and bounded by `\\b`, so both
    `v[pp]` value reads in the equations and bare `v` pointer args in the
    deriv-calc are caught. Keyed on the closed, enumerated input set, so the
    matches are exact. Must run AFTER the deriv-name rewrite.
    """
    if not input_names or not struct:
        return code
    for v in input_names:
        code = regex.sub(
            rf"(?<![\w.]){regex.escape(v)}\b",
            f"{struct}.{v}",
            code,
        )
    return code


# derivative-buffer tokens: grad_0_X, grad2_0_1_X (agrad already folded to grad).
# the digit after the prefix keeps it from matching the grad_x/grad_xy *methods*.
# Fallback shape, used only when no enumerated field set is supplied.
_DERIV_BUF_PAT = r"(grad2_\d+_\d+_\w+|grad_\d+_\w+)"

# direction-index -> axis letter, shared by the rename + the operator-name parse.
_DIR_AXIS = {"0": "x", "1": "y", "2": "z"}

# second-order axis pairs, longest-first so `xx` wins over `x` without relying
# on alternation backtracking; first-order axes follow.
_AXIS_SUFFIX = r"(?:xx|xy|xz|yy|yz|zz|x|y|z)"


def _fields_alt(field_names) -> str:
    """Regex alternation of the field names, longest-first so e.g. `Gammahat0`
    wins over a shorter field that is a prefix of it."""
    return "|".join(
        regex.escape(f) for f in sorted(field_names, key=len, reverse=True)
    )


def rename_deriv_buffers(code: str, field_names, use_advective: bool = False) -> str:
    """Rename canonical deriv-buffer tokens to operator form, keyed on fields.

    `grad_{i}_X -> X_{axis}`, `grad2_{i}_{j}_X -> X_{axis}{axis}`,
    `agrad_{i}_X -> adv_X_{axis}` (the advective form only survives when
    advection is on; otherwise the buffers were already folded to `grad_`).
    So `grad_0_chi -> chi_x`, `grad2_0_1_chi -> chi_xy`, `agrad_0_chi -> adv_chi_x`.

    Keyed on the closed, enumerated field set so the match is exact (the bare
    field reads `in.X` and the `grad_x(`/`deriv_x(` operator names are never
    touched). Applies to every emitted string -- the workspace carve, the
    deriv-calc, the equations, KO and BC -- so all references stay in lock-step.
    """
    if not field_names:
        return code
    fields = _fields_alt(field_names)
    # grad2 first: its tokens contain no literal `grad_` substring, but order it
    # ahead anyway so the intent is clear.
    code = regex.sub(
        rf"(?<![\w.])grad2_([012])_([012])_({fields})\b",
        lambda m: f"{m.group(3)}_{_DIR_AXIS[m.group(1)]}{_DIR_AXIS[m.group(2)]}",
        code,
    )
    code = regex.sub(
        rf"(?<![\w.])grad_([012])_({fields})\b",
        lambda m: f"{m.group(2)}_{_DIR_AXIS[m.group(1)]}",
        code,
    )
    code = regex.sub(
        rf"(?<![\w.])agrad_([012])_({fields})\b",
        lambda m: f"adv_{m.group(2)}_{_DIR_AXIS[m.group(1)]}",
        code,
    )
    return code


def apply_deriv_struct(
    code: str, field_names=None, struct: str = "d", extra_names=None
) -> str:
    """Prefix derivative-buffer reads `chi_x` -> `struct.chi_x` (`d.`).

    Run AFTER `rename_deriv_buffers`, so the buffers are in operator form. Keyed
    on the enumerated field set + the closed axis-suffix set, the lookbehind
    skips an already-prefixed `d.chi_x` and the bare/`in.`-grouped field reads.
    Apply to the deriv-calc/equations/KO/BC, NOT the struct definition (whose
    members are the bare names). With no field set, falls back to the canonical
    `grad_*` token shape (pre-rename behaviour).

    `extra_names` is an explicit set of staged/intermediate buffer names (e.g.
    `DENDRO_STAGED_GRAD_000`, `..._intermediate`) that are not field-derived
    operator names; they are prefixed by exact match (longest-first + word
    boundary so a name that is a prefix of another does not partial-match).
    """
    if not struct:
        return code
    if not field_names:
        code = regex.sub(rf"(?<![\w.]){_DERIV_BUF_PAT}", rf"{struct}.\1", code)
    else:
        pat = rf"((?:adv_)?(?:{_fields_alt(field_names)})_{_AXIS_SUFFIX})"
        code = regex.sub(rf"(?<![\w.]){pat}\b", rf"{struct}.\1", code)
    if extra_names:
        alt = "|".join(
            regex.escape(n) for n in sorted(extra_names, key=len, reverse=True)
        )
        code = regex.sub(rf"(?<![\w.])({alt})\b", rf"{struct}.\1", code)
    return code


def malloc_to_carve(malloc_code: str, start_offset: int = 0):
    """Convert malloc'd staged-buffer decls to `deriv_base` carve lines.

    `T *NAME = (T *)malloc(...);` -> `T * NAME = deriv_base + k * BLK_SZ;`,
    numbering from `start_offset`. Returns `(carve_code, names)`. Non-matching
    lines (comments, blanks) are dropped -- the carve is appended to the
    deriv-workspace memalloc so the staged/intermediate buffers join the `d.`
    struct + `count()` alongside the ordinary derivative buffers.
    """
    carve, names = [], []
    k = start_offset
    for line in malloc_code.splitlines():
        m = regex.match(
            r"\s*(\w[\w ]*?)\s*\*\s*(\w+)\s*=\s*\([^)]*\)\s*malloc\([^;]*;\s*$",
            line,
        )
        if not m:
            continue
        dtype, name = m.group(1), m.group(2)
        carve.append(f"{dtype} * {name} = deriv_base + {k} * BLK_SZ;")
        names.append(name)
        k += 1
    return ("\n".join(carve) + ("\n" if carve else "")), names


def gen_deriv_struct(memalloc_code: str, struct_name: str) -> str:
    """Build a derivative-workspace struct from the memalloc carve.

    Parses `T *NAME = deriv_base + k*BLK_SZ;` lines into a struct holding the
    pointer members, a `count()` (auto -- replaces the hand-set NUM_DERIVATIVES),
    and a `bind()` that performs the carve. The kernel then reads `d.NAME`.
    """
    members, binds = [], []
    for line in memalloc_code.splitlines():
        m = regex.match(
            r"\s*\w[\w ]*\*\s*(\w+)\s*=\s*(deriv_base\s*\+\s*\d+\s*\*\s*BLK_SZ);",
            line,
        )
        if m:
            members.append(m.group(1))
            binds.append((m.group(1), m.group(2)))
    out = f"// derivative workspace -- grouped + self-sizing (see d.bind/d.count)\n"
    out += f"struct {struct_name} {{\n"
    for name in members:
        out += f"    double *{name};\n"
    out += f"    static constexpr unsigned int count() {{ return {len(members)}; }}\n"
    out += "    void bind(double *deriv_base, unsigned int BLK_SZ) {\n"
    for name, expr in binds:
        out += f"        {name} = {expr};\n"
    out += "    }\n};\n"
    return out


# one deriv-calc call: OBJ.grad_<axis>(d.<dst>, <in.X|d.Y>, h<axis>, sz, bflag);
_DERIV_CALL_PAT = regex.compile(
    r"^\s*(\w+)\.grad_(xx|yy|zz|x|y|z)\("
    r"(d\.\w+),\s*((?:in|d)\.\w+),\s*(h[xyz]),\s*sz,\s*bflag\);\s*$"
)


def reduce_deriv_calc_for_fused(calc: str):
    """Keep only the derivative calls a FUSED cascade body still needs.

    Input: the per-call form (before `group_deriv_calc`). The fused body computes
    every 1st and pure-2nd derivative inline, so on interior blocks the deriv pass
    only has to produce (a) the mixed seconds and (b) the first-order buffers those
    mixed seconds read (`grad_y(d.X_xy, d.X_x)` needs `d.X_x`). Everything else --
    pure seconds, and firsts nobody downstream reads -- is dropped. KO uses the
    first-order buffers as scratch only and the BC table runs on boundary blocks,
    which keep the full pass.

    Returns (reduced_calc, kept, dropped) or None if a line is not a recognized
    deriv call (staged/intermediate solvers), in which case fusion must be refused.
    """
    lines = [ln for ln in calc.splitlines() if ln.strip()]
    parsed = []
    for ln in lines:
        m = _DERIV_CALL_PAT.match(ln)
        if not m:
            return None
        parsed.append((ln, m.groups()))
    needed = {src for _ln, (_o, op, _dst, src, _h) in parsed
              if op in ("x", "y", "z") and src.startswith("d.")}
    kept = []
    for ln, (_o, op, dst, src, _h) in parsed:
        if op in ("x", "y", "z") and src.startswith("d."):
            kept.append(ln)                      # mixed second
        elif op in ("x", "y", "z") and dst in needed:
            kept.append(ln)                      # first feeding a mixed second
    tail = "\n" if calc.endswith("\n") else ""
    return "\n".join(kept) + tail, len(kept), len(parsed) - len(kept)


def group_deriv_calc(calc: str) -> str:
    """Regroup the deriv-calc into batched `grad_*_batch` dispatches.

    Every per-block deriv call is bucketed by the operator applied, into three
    ordered phases so dependencies hold:
      1. first derivatives  -- `grad_x/y/z` reading `in.X`  -> grad_{x,y,z}_batch
      2. pure seconds       -- `grad_xx/yy/zz` reading `in.X` -> grad_{xx,yy,zz}_batch
      3. mixed seconds      -- `grad_y/z` reading a first-order `d.X_*` buffer
                               -> grad_{y,z}_batch, AFTER phase 1 fills those buffers
    Buckets are keyed (phase, operator); within a bucket every call shares the
    same step `h`. Reordering calls within a phase is bit-identical (each writes
    a distinct buffer, no aliasing); phase order preserves the mixed-second deps.
    On an explicit engine each `grad_*_batch` loops the raw stencil fn, so the
    result is identical to the per-call form; matrix/compact engines share the
    operator across the batch (the actual win).

    Bails (returns `calc` unchanged) unless every non-blank line is a recognized
    deriv call, so a solver with intermediate/staged lines is left untouched.
    """
    lines = [ln for ln in calc.splitlines() if ln.strip()]
    if not lines:
        return calc
    obj = None
    groups = {}  # (phase, operator) -> [h, [(dst, src), ...]]
    order = []   # first-seen key order (stable-sorted by phase for emission)
    for ln in lines:
        m = _DERIV_CALL_PAT.match(ln)
        if not m:
            return calc
        this_obj, op, dst, src, h = m.groups()
        obj = obj or this_obj
        if op in ("x", "y", "z"):
            phase = 1 if src.startswith("in.") else 3  # 1st vs mixed-2nd
        else:
            phase = 2  # pure 2nd (grad_xx/yy/zz)
        key = (phase, op)
        if key not in groups:
            groups[key] = [h, []]
            order.append(key)
        groups[key][1].append((dst, src))

    order.sort(key=lambda k: k[0])  # stable: phases grouped, intra-phase order kept
    phase_label = {
        1: "first derivatives",
        2: "pure second derivatives",
        3: "mixed second derivatives (read the first-order buffers above)",
    }
    out, k, last_phase = [], 0, None
    for key in order:
        phase, op = key
        h, pairs = groups[key]
        if phase != last_phase:
            out.append(f"// {phase_label[phase]} -- batched dispatch")
            last_phase = phase
        n = len(pairs)
        out_arr = ", ".join(d for d, _ in pairs)
        in_arr = ", ".join(s for _, s in pairs)
        out.append("{")
        out.append(f"    double *__db{k}_out[] = {{ {out_arr} }};")
        out.append(f"    const double *__db{k}_in[] = {{ {in_arr} }};")
        out.append(
            f"    {obj}.grad_{op}_batch("
            f"__db{k}_out, __db{k}_in, {n}, {h}, sz, bflag);"
        )
        out.append("}")
        k += 1
    tail = "\n" if calc.endswith("\n") else ""
    return "\n".join(out) + tail



# ---------------------------------------------------------------------------
# planned per-variable derivative sets (DendroDerivatives::grad_set)
# ---------------------------------------------------------------------------

_DERIVS_TYPE = "dendroderivs::DendroDerivatives"
# DerivSet aggregate-initializer positions. C++17 has no designated
# initializers, so unused slots are emitted as explicit nullptr.
_DERIV_SET_MEMBERS = ("x", "y", "z", "xx", "yy", "zz", "xy", "xz", "yz")
# a first derivative chained into a mixed second: (op, source axis) -> member
_MIXED_FROM_CHAIN = {("y", "x"): "xy", ("z", "x"): "xz", ("z", "y"): "yz"}
_ADV_CALL_PAT = regex.compile(r"\badv_deriv_[xyz]\(")


_MASK_ALIASES = (
    ("DM_ALL", _DERIV_SET_MEMBERS),
    ("DM_FIRST", ("x", "y", "z")),
    ("DM_SECOND", ("xx", "yy", "zz")),
    ("DM_MIXED", ("xy", "xz", "yz")),
)


def _mask_names(mask):
    """Name a mask with the engine's aliases where a whole family is present."""
    left = set(mask)
    names = []
    for alias, family in _MASK_ALIASES:
        if left.issuperset(family):
            names.append(alias)
            left.difference_update(family)
            if not left:
                return names
    names.extend(f"DM_{m.upper()}" for m in _DERIV_SET_MEMBERS if m in left)
    return names


def _parse_deriv_sets(calc: str):
    """Reconstruct per-variable derivative sets from the per-call deriv list.

    Keys off the CALL GRAPH, never off buffer names: a mixed second is
    `grad_{y,z}(dst, S)` where `S` is the output of an earlier first-derivative
    call, and its member follows from (operator, that call's axis). Buffer names
    can contain underscores, so splitting them would be unsound.

    Returns (obj, [(src, {member: buffer}), ...], steps, passthrough), or None
    if a line is not a recognized call whose chain resolves -- staged /
    intermediate solvers, where the caller must fall back.
    """
    obj = None
    sets, order = {}, []
    axis_of = {}        # first-derivative dst buffer -> (src, axis)
    steps = {}          # axis -> step symbol (hx/hy/hz)
    passthrough = []    # advective calls, kept verbatim

    for ln in calc.splitlines():
        if not ln.strip():
            continue
        m = _DERIV_CALL_PAT.match(ln)
        if not m:
            # advective derivatives read `in.X` and write their own agrad
            # buffers -- they neither feed nor consume a set, so keep them.
            if _ADV_CALL_PAT.search(ln):
                passthrough.append(ln)
                continue
            return None
        this_obj, op, dst, src, h = m.groups()
        obj = obj or this_obj

        if src.startswith("in."):
            member, axis = op, op[0]
            if op in ("x", "y", "z"):
                axis_of[dst] = (src, op)
        else:
            base = axis_of.get(src)
            if base is None:
                return None                 # staged buffer or unseen chain
            src, src_axis = base
            member = _MIXED_FROM_CHAIN.get((op, src_axis))
            if member is None:
                return None                 # chain shape grad_set cannot plan
            axis = op[0]

        if src not in sets:
            sets[src] = {}
            order.append(src)
        if member in sets[src]:
            return None                     # two writers for one output
        sets[src][member] = dst
        steps.setdefault(axis, h)

    if not sets:
        return None
    return obj, [(s, sets[s]) for s in order], steps, passthrough


def emit_deriv_calc_grad_set(calc: str):
    """Emit the deriv-calc as planned `grad_set` / `grad_set_batch` calls.

    One call per variable states WHICH derivatives the RHS uses (a DerivMask)
    and where each goes; the engine plans every call's shape from that --
    terminal outputs run the active-region `_last` kernels, a first derivative
    feeding a mixed one stays a full-extent intermediate and is reused. That is
    the win: a hand-emitted call list cannot use `_last`, because it does not
    know which outputs are terminal.

    Variables sharing a mask go through one `grad_set_batch`. Phase ordering
    disappears: a mixed second only ever chains off its OWN variable's first
    derivative, which `grad_set` handles internally.

    CONTRACT: `_last` leaves the block padding undefined, so nothing may read a
    derivative buffer outside [pw, n-pw)^3. The RHS/BC/cascade loops comply; the
    staged-intermediate loop does not, which is why that path returns None.

    Returns None when the chain cannot be resolved -- caller falls back to
    `group_deriv_calc`.
    """
    parsed = _parse_deriv_sets(calc)
    if parsed is None:
        return None
    obj, sets, steps, passthrough = parsed

    # group by mask; first-seen order (the deriv lists are fully sorted, so this
    # is deterministic across PYTHONHASHSEED)
    groups, gorder = {}, []
    for src, members in sets:
        mask = tuple(m for m in _DERIV_SET_MEMBERS if m in members)
        if mask not in groups:
            groups[mask] = []
            gorder.append(mask)
        groups[mask].append((src, members))

    hx, hy, hz = (steps.get(a, "h" + a) for a in ("x", "y", "z"))
    out = ["// planned per-variable derivative sets -- the engine picks each",
           "// call's shape (terminal outputs use the active-region kernels)",
           f"using __DD = {_DERIVS_TYPE};"]
    for k, mask in enumerate(gorder):
        rows = groups[mask]
        mask_expr = " | ".join(f"__DD::{n}" for n in _mask_names(mask))
        # trailing all-null slots are left to the struct's defaults
        keep = max(_DERIV_SET_MEMBERS.index(m) for m in mask) + 1
        cols = _DERIV_SET_MEMBERS[:keep]
        out.append("{")
        out.append(f"    const __DD::DerivSet __ds{k}[] = {{"
                   f"   // {', '.join(cols)}")
        for _src, members in rows:
            row = ", ".join(members.get(m, "nullptr") for m in cols)
            out.append(f"        {{ {row} }},")
        out.append("    };")
        out.append("    const double *const __du{}[] = {{ {} }};".format(
            k, ", ".join(src for src, _m in rows)))
        if len(rows) == 1:
            out.append(f"    {obj}.grad_set(__ds{k}[0], __du{k}[0], {mask_expr},")
            out.append(f"    {' ' * len(obj)}              {hx}, {hy}, {hz}, sz, bflag);")
        else:
            out.append(f"    {obj}.grad_set_batch(__ds{k}, __du{k}, {len(rows)},"
                       f" {mask_expr},")
            out.append(f"    {' ' * len(obj)}                    {hx}, {hy}, {hz}, sz, bflag);")
        out.append("}")
    out.extend(passthrough)
    tail = "\n" if calc.endswith("\n") else ""
    return "\n".join(out) + tail


def count_deriv_buffers(memalloc_code: str) -> int:
    """Count the `T *NAME = deriv_base + k*BLK_SZ;` buffers in a memalloc carve.

    Same line shape `gen_deriv_struct` parses into members, so this equals the
    struct's `count()`. Used to size NUM_DERIVATIVES from the (deduped) workspace
    instead of the hand-set magic number.
    """
    n = 0
    for line in memalloc_code.splitlines():
        if regex.match(
            r"\s*\w[\w ]*\*\s*(\w+)\s*=\s*(deriv_base\s*\+\s*\d+\s*\*\s*BLK_SZ);",
            line,
        ):
            n += 1
    return n


def fold_agrad_to_grad(code: str) -> str:
    """Rewrite advective deriv buffers to the regular ones: agrad_i_X -> grad_i_X.

    Used when advection is disabled (use_advective=False): the call rewrite
    already computes agrad buffers with the centered stencil, making them
    bit-identical to grad_i_X. Folding the names lets the duplicate buffers be
    deduped away. Leaves grad2_/grad_/kograd_ tokens untouched.
    """
    return regex.sub(r"\bagrad_(\d+)_", r"grad_\1_", code)


def dedup_deriv_alloc(code: str) -> str:
    """Drop duplicate `T *NAME = deriv_base + k*BLK_SZ;` lines and renumber.

    After agrad->grad folding the workspace carve has repeated buffer names;
    keep the first of each and recompact offsets 0..N-1 so the workspace shrinks
    by exactly the number of folded duplicates.
    """
    seen = set()
    out, k = [], 0
    for line in code.splitlines():
        m = regex.match(
            r"\s*(\w[\w ]*\*)\s*(\w+)\s*=\s*deriv_base\s*\+\s*\d+\s*\*\s*BLK_SZ;",
            line,
        )
        if m:
            name = m.group(2)
            if name in seen:
                continue
            seen.add(name)
            out.append(f"{m.group(1).strip()} {name} = deriv_base + {k} * BLK_SZ;")
            k += 1
        else:
            out.append(line)
    tail = "\n" if code.endswith("\n") else ""
    return "\n".join(out) + tail


def dedup_lines(code: str) -> str:
    """Remove exact-duplicate non-empty lines (keep first occurrence).

    For the deriv-calc after agrad->grad folding: the folded advective calls
    become byte-identical to the regular ones, so the second is redundant.
    """
    seen, out = set(), []
    for line in code.splitlines():
        s = line.strip()
        if s and s in seen:
            continue
        if s:
            seen.add(s)
        out.append(line)
    tail = "\n" if code.endswith("\n") else ""
    return "\n".join(out) + tail


def generate_fpcore(ex, vnames, idx):
    """Gennerate FPCore code

    FPCore is the formate used in FPBench benchmarks. It is a basic
    programming language with conditionals and simple loops. FPBench
    is a good benchmark for floating-point computation to understand
    how effective the code is.
    """

    # TODO: this may need to be updated to match the power of CPU gen

    mi = [0, 1, 2, 4, 5, 8]
    midx = ["00", "01", "02", "11", "12", "22"]

    # total number of expressions
    # print("--------------------------------------------------------")
    num_e = 0
    lexp = []
    lname = []
    for i, e in enumerate(ex):
        if type(e) == list:
            num_e = num_e + len(e)
            for j, ev in enumerate(e):
                lexp.append(ev)
                lname.append(vnames[i] + repr(j) + idx)
        elif type(e) == sym.Matrix:
            num_e = num_e + len(e)
            for j, k in enumerate(mi):
                lexp.append(e[k])
                lname.append(vnames[i] + midx[j] + idx)
        else:
            num_e = num_e + 1
            lexp.append(e)
            lname.append(vnames[i] + idx)

    cse = construct_cse(ex, vnames, idx)
    _v = cse[0]

    # print("// Dendro: {{{ ")
    # print("// Dendro: original ops: %d " %(cse[1]))

    ee_name = "DENDRO_"
    ee_syms = sym.utilities.numbered_symbols(prefix=ee_name)

    custom_functions = {
        "grad": "grad",
        "grad2": "grad2",
        "agrad": "agrad",
        "kograd": "kograd",
    }
    rops = 0

    # re_symbol=regex.compile(r"Symbol\('[a-z,A-Z,_]+[0-9,\[pp\],\[0-9\]]*'\)")
    re_symbol = regex.compile(r"Symbol\('([a-z,A-Z,0-9,_,\[\]]*)'\)")
    re_integer = regex.compile(r"Integer\(([\-,0-9]+)\)")
    re_float = regex.compile(r"Float\('([\-,0-9]*\.[0-9]*)'\s prec=([0-9]+)\)")
    re_grad = regex.compile(
        r"Function\('([a-z]+[0-9]*)'\)\(Integer\(([0-9]+)\)"
        r",\s*Symbol\('([a-z,A-Z]+[0-9]*\[pp\])'\)\)"
    )

    subs_functions = {
        "Add(": "(+ ",
        "Integer(-1)": "-1 ",
        "Mul(": "(* ",
        "Div(": "(/ ",
        "Pow(": "(pow ",
        "Rational(": "(/ ",
    }

    # print('// Dendro: printing temp variables')
    tmp_vars = list()
    for v1, v2 in _v[0]:
        tmp_vars.append(str(v1))
        sym_sub = dict()
        srep = sym.srepr(v2)
        # print(srep)

        res = re_grad.findall(srep)
        for g in res:
            s = "Function('%s')(Integer(%s), Symbol('%s'))" % (g[0], g[1], g[2])
            # print(s)
            ss = "Symbol('%s')" % (g[0] + "_" + g[1] + "_" + g[2])
            srep = srep.replace(s, ss)

        srep = srep.replace(",", " ")
        # print(srep)

        res = re_symbol.findall(srep)
        inp_params = list()
        # print(res)
        for s in res:
            ss = s.replace("[pp]", "")
            for index in range(0, 6):
                ss = ss.replace("[" + str(index) + "]", str(index))
            inp_params.append(ss)
            tmp_vars.append(ss)
            sym_sub["Symbol('%s')" % (s)] = ss

        int_sub = dict()
        res = re_integer.findall(srep)
        for s in res:
            int_sub["Integer(%s)" % (s)] = s

        float_sub = dict()
        res = re_float.findall(srep)

        for s in res:
            float_sub["Float('%s'  prec=%s)" % (s[0], s[1])] = s[0]

        for key, val in sym_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in int_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in float_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in subs_functions.items():
            srep = srep.replace(key, val)

        print("(FPCore (%s)" % (" ".join(inp_params)))
        print("\t%s" % (srep))
        print(")\n")

    # print(tmp_vars)
    tmp_vars.clear()
    tmp_vars = list()
    for i, e in enumerate(_v[1]):
        srep = sym.srepr(e)

        res = re_grad.findall(srep)
        for g in res:
            s = "Function('%s')(Integer(%s), Symbol('%s'))" % (g[0], g[1], g[2])
            # print(s)
            ss = "Symbol('%s')" % (g[0] + "_" + g[1] + "_" + g[2])
            srep = srep.replace(s, ss)

        srep = srep.replace(",", " ")

        res = re_symbol.findall(srep)
        inp_params = list()
        # print(res)
        for s in res:
            ss = s.replace("[pp]", "")
            for index in range(0, 6):
                ss = ss.replace("[" + str(index) + "]", str(index))
            inp_params.append(ss)
            tmp_vars.append(ss)
            sym_sub["Symbol('%s')" % (s)] = ss

        int_sub = dict()
        res = re_integer.findall(srep)
        for s in res:
            int_sub["Integer(%s)" % (s)] = s

        float_sub = dict()
        res = re_float.findall(srep)

        for s in res:
            float_sub["Float('%s'  prec=%s)" % (s[0], s[1])] = s[0]

        for key, val in sym_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in int_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in float_sub.items():
            # print("{%s: %s} "%(key,val))
            srep = srep.replace(key, val)

        for key, val in subs_functions.items():
            srep = srep.replace(key, val)

        tmp_vars = list(set(tmp_vars))
        print("(FPCore (%s)" % (" ".join(tmp_vars)))
        print("\t%s" % (srep))
        print(")")
        # print(")")
        # print(")")
        # print(change_deriv_names(ccode(e, assign_to=lname[i],
        #       user_functions=custom_functions)))


def generate_avx(ex, vnames, idx):
    """Generate AVX C++ code

    AVX is the 'advanced vector extensions' library that you can
    use in C++.

    Notes
    -----
    I'm unsure if this is still useful to the project, but has been
    kept for future reference and backwards compatibility.
    """

    mi = [0, 1, 2, 4, 5, 8]
    midx = ["00", "01", "02", "11", "12", "22"]

    # total number of expressions
    # print("--------------------------------------------------------")
    num_e = 0
    lexp = []
    lname = []
    for i, e in enumerate(ex):
        if type(e) == list:
            num_e = num_e + len(e)
            for j, ev in enumerate(e):
                lexp.append(ev)
                lname.append(vnames[i] + repr(j) + idx)
        elif type(e) == sym.Matrix:
            num_e = num_e + len(e)
            for j, k in enumerate(mi):
                lexp.append(e[k])
                lname.append(vnames[i] + midx[j] + idx)
        else:
            num_e = num_e + 1
            lexp.append(e)
            lname.append(vnames[i] + idx)

    cse = construct_cse(ex, vnames, idx)
    _v = cse[0]

    print("// Dendro: {{{ ")
    print("// Dendro: original ops: %d " % (cse[1]))

    ee_name = "DENDRO_"
    ee_syms = sym.utilities.numbered_symbols(prefix=ee_name)

    print("// Dendro vectorized code: {{{")
    oper = {"mul": "dmul", "add": "dadd", "load": "*"}
    prevdefvars = set()
    for v1, v2 in _v[0]:
        vv = sym.utilities.numbered_symbols("v")
        vlist = []
        gen_vector_code(v2, vv, vlist, oper, prevdefvars, idx)
        print("  double " + repr(v1) + " = " + repr(vlist[0]) + ";")
    for i, e in enumerate(_v[1]):
        print("//--")
        vv = sym.utilities.numbered_symbols("v")
        vlist = []
        gen_vector_code(e, vv, vlist, oper, prevdefvars, idx)
        # st = '  ' + repr(lname[i]) + '[idx] = ' + repr(vlist[0]) + ';'
        st = "  " + repr(lname[i]) + " = " + repr(vlist[0]) + ";"
        print(st.replace("'", ""))

    print("// Dendro vectorized code: }}} ")


def generate_separate_cpu(
    ex, vnames, idx, orig_n_exp, proj_name="bssn", dtype="double", use_const=False
):
    """Generates the code for separate variable calculation on CPU"""

    total_reduced_ops = 0
    orig_ops = sym.count_ops(ex)

    output_str = (
        "// Dendro: C++ Equation Code Generation for Separate Calculation {{{{ \n"
    )
    output_str += "// =================\n"

    # now we iterate through each one of our variables
    for ii, single_ex in enumerate(ex):
        single_vname = vnames[ii]
        print("== Now generating for " + single_vname, file=sys.stderr)

        output_str += f"// Dendro: Generated code for {single_vname}\n"
        output_str += f"{proj_name}::timer::{single_vname}.start();\n\n"

        # start loop opening for K (z)
        output_str += "for (unsigned int k = PW; k < nz - PW; k++) {\n"
        # definition for the z position
        output_str += "    z = pmin[2] + k * hz;\n"

        # start loop opening for y
        output_str += "for (unsigned int j = PW; j < ny - PW; j++) {\n"
        # definition for the y position
        output_str += "    y = pmin[1] + j * hy;\n"

        # start loop opening for x
        output_str += "for (unsigned int i = PW; i < nx - PW; i++) {\n"
        # definition for the x position
        output_str += "    x = pmin[0] + i * hx;\n"

        # then we add the calculation for pp, r_coord, eta, and more
        output_str += f"    {idx} = i + nx * (j + ny * k);\n"
        output_str += "    r_coord = sqrt(x*x + y*y + z*z);\n"

        # TODO: ETA CONSTANT NEEDS TO BE CONSIDERED HERE!!!! MAY NOT BE NECESSARY
        output_str += "    eta = ETA_CONST;\n"
        output_str += "    if (r_coord >= ETA_R0)\n"
        output_str += "        eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);\n"

        # then we can get the generated code
        exp_ops = sym.count_ops(single_ex)
        # now extract the CSE
        cse_out = construct_cse_from_list([single_ex])
        tmp_str, reduced_ops = generate_cpu_preextracted(
            cse_out, [single_vname], idx, 0, dtype, use_const, True
        )
        total_reduced_ops += reduced_ops
        output_str += tmp_str

        output_str += f"// Dendro: Original operations for this variable: {exp_ops}\n"
        output_str += (
            f"// Dendro: Reduced operations for this variable: {reduced_ops}\n"
        )
        output_str += "    }\n  }\n}\n"
        output_str += f"{proj_name}::timer::{single_vname}.stop();\n"
        output_str += f"// Dendro: End generated code for {single_vname}\n\n"

    # now at the end, we can add some other cool stuff

    output_str += "// =================\n"
    output_str += "// Dendro: INFORMATION\n"
    output_str += "// Dendro: number of original operations: %d \n" % (orig_ops)
    output_str += "// Dendro: number of reduced operations: %d \n" % (total_reduced_ops)
    output_str += "// Dendro: preprocessing reduced the "
    output_str += f"number of operations by {orig_ops - reduced_ops}\n"
    percent_reduction = (orig_ops - total_reduced_ops) / orig_ops
    output_str += f"// Dendro: a {percent_reduction:0.5%}% reduction\n"
    output_str += "// Dendro: }}}} End Code Generation \n"

    return output_str


def generate_separate(ex, vnames, idx, prefix=""):
    """Generate 'separate' C++ code after simplification

    Note
    ----
    I'm not sure what this 'separate' means, but it also includes
    hard coded references to the `bssn` namespace, so this may
    need to be modified in the future.
    """
    # print(ex)
    if len(ex) != 1:
        print("pass each variable separately ", end="\n")
        return

    mi = [0, 1, 2, 4, 5, 8]
    midx = ["00", "01", "02", "11", "12", "22"]

    # total number of expressions
    # print("--------------------------------------------------------")
    num_e = 0
    lexp = []
    lname = []
    for i, e in enumerate(ex):
        if type(e) == list:
            num_e = num_e + len(e)
            for j, ev in enumerate(e):
                lexp.append(ev)
                lname.append(vnames[i] + repr(j) + idx)
        elif type(e) == sym.Matrix:
            num_e = num_e + len(e)
            for j, k in enumerate(mi):
                lexp.append(e[k])
                lname.append(vnames[i] + midx[j] + idx)
        else:
            num_e = num_e + 1
            lexp.append(e)
            lname.append(vnames[i] + idx)

    # print(num_e)
    # print(len(lname))
    c_file = open(prefix + vnames[0] + ".cpp", "w")
    print("generating code for " + vnames[0])
    print("    bssn::timer::t_rhs.start();", file=c_file)
    print("for (unsigned int k = 3; k < nz-3; k++) { ", file=c_file)
    print("    z = pmin[2] + k*hz;", file=c_file)

    print("for (unsigned int j = 3; j < ny-3; j++) { ", file=c_file)
    print("    y = pmin[1] + j*hy; ", file=c_file)

    print("for (unsigned int i = 3; i < nx-3; i++) {", file=c_file)
    print("    x = pmin[0] + i*hx;", file=c_file)
    print("    pp = i + nx*(j + ny*k);", file=c_file)
    print("    r_coord = sqrt(x*x + y*y + z*z);", file=c_file)
    print("    eta=ETA_CONST;", file=c_file)
    print("    if (r_coord >= ETA_R0) {", file=c_file)
    print("    eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);", file=c_file)
    print("    }", file=c_file)

    print("// Dendro: {{{ ", file=c_file)
    print("// Dendro: original ops: ", sym.count_ops(lexp), file=c_file)

    # print("--------------------------------------------------------")
    # print("Now trying Common Subexpression Detection and Collection")
    # print("--------------------------------------------------------")

    # Common Subexpression Detection and Collection
    # for i in range(len(ex)):
    #     # print("--------------------------------------------------------")
    #     # print(ex[i])
    #     # print("--------------------------------------------------------")
    #     ee_name = ''.join(
    #         random.choice(string.ascii_uppercase) for _ in range(5))
    #     ee_syms = sym.utilities.numbered_symbols(prefix=ee_name)
    #     _v = cse(ex[i],symbols=ee_syms)
    #     # print(type(_v))
    #     for (v1,v2) in _v[0]:
    #         print("double %s = %s;" % (v1, v2))
    #     print("%s = %s" % (vnames[i], _v[1][0]))

    # mex = Matrix(ex)
    ee_name = "DENDRO_"
    # (ABOVE) ''.join(random.choice(string.ascii_uppercase) for _ in range(5))
    ee_syms = sym.utilities.numbered_symbols(prefix=ee_name)
    _v = construct_cse(lexp, symbols=ee_syms, optimizations="basic")

    custom_functions = {
        "grad": "grad",
        "grad2": "grad2",
        "agrad": "agrad",
        "kograd": "kograd",
    }

    cprinter = dendrosym.code_printer.DendroCPrinter(
        additional_user_funcs=custom_functions
    )

    rops = 0
    print("// Dendro: printing temp variables", file=c_file)
    for v1, v2 in _v[0]:
        # print("double %s = %s;" % (v1, v2)) # replace_pow(v2)))
        print("double ", end="", file=c_file)
        print(
            change_deriv_names(
                cprinter.doprint(v2, assign_to=v1)
                # sym.ccode(v2, assign_to=v1, user_functions=custom_functions)
            ),
            file=c_file,
        )
        rops = rops + sym.count_ops(v2)

    print("// Dendro: printing variables", file=c_file)
    for i, e in enumerate(_v[1]):
        print("//--", file=c_file)
        # print("%s = %s;" % (lname[i], e)) # replace_pow(e)))
        f = open(str(vnames[0]) + ".gv", "w")
        print(sym.printing.dot.dotprint(e), file=f)
        f.close()
        print(
            change_deriv_names(
                cprinter.doprint(e, assign_to=lname[i])
                # sym.ccode(e, assign_to=lname[i], user_functions=custom_functions)
            ),
            file=c_file,
        )
        # c_file.write('\n')
        rops = rops + sym.count_ops(e)

    print("// Dendro: reduced ops: ", rops, file=c_file)
    print("// Dendro: }}} ", file=c_file)

    print("  }", file=c_file)
    print(" }", file=c_file)
    print("}", file=c_file)
    print("     bssn::timer::t_rhs.stop();", file=c_file)
    c_file.close()
    print("generating code for " + vnames[0] + " completed")


def replace_pow(exp_in):
    """Convert integer powers to multiplications

    This function finds all instances of integer power in expressions
    and converts them within the expression to multiplication.
    This is done because the C++ implementation of `pow` is much
    slower than pure multiplication of expressions.

    Parameters
    ----------
    exp_in : sympy.Expression
        The full expression that should be analyzed

    Returns
    -------
    sympy.Expression
        The new output expression that has replaced the "pow" with
        multiplication.
    """

    # recursive call for expressions that are lists/tuples
    if isinstance(exp_in, (list, tuple)):
        return type(exp_in)(replace_pow(e) for e in expr)

    # make sure we only work on sympy expressions
    if not isinstance(exp_in, sym.Expr):
        return exp_in

    pows = exp_in.atoms(sym.Pow)
    if not pows:
        return exp_in

    replacement_map = {}
    for p in pows:
        b, e = p.as_base_exp()

        if e.is_integer:
            e_int = int(e)

            if e_int == 0:
                replacement_map[p] = sym.Integer(1)
            elif e_int == 1:
                replacement_map[p] = b
            elif e_int > 1:
                # replace with multiplication directly to avoid pow, could help with simplification
                replacement_map[p] = sym.Mul(*[b] * e_int, evaluate=False)
            else:
                denominator = sym.Mul(*[b] * abs(e_int), evaluate=False)
                replacement_map[p] = sym.Integer(1) / denominator
    # NOTE: all non-integer powers are left alone!

    return exp_in.xreplace(replacement_map)


def generate_debug(ex, vnames):
    """Debug version of generating code

    I believe this is depreciated and never used, since generate_cpu
    and other functions have been declared and fleshed out. Kept
    for potential use.
    """
    # print(ex)

    mi = [0, 1, 2, 4, 5, 8]
    midx = ["00", "01", "02", "11", "12", "22"]

    # total number of expressions
    # print("--------------------------------------------------------")
    num_e = 0
    lexp = []
    lname = []
    print("// Dendro: {{{ ")
    for i, e in enumerate(ex):
        if type(e) == list:
            num_e = num_e + len(e)
            for j, ev in enumerate(e):
                # lexp.append(ev)
                print(vnames[i] + repr(j), end="")
                print(" = ", end="")
                print(replace_pow(ev), ";")
        elif type(e) == sym.Matrix:
            num_e = num_e + len(e)
            for j, k in enumerate(mi):
                # lexp.append(e[k])
                print(vnames[i] + midx[j], end="")
                print(" = ", end="")
                print(replace_pow(e[k]), ";")
        else:
            num_e = num_e + 1
            # lexp.append(e)
            print(vnames[i], end="")
            print(" = ", end="")
            print(replace_pow(e), ";")

    print("// Dendro: }}} ")


def vec_print_str(tv, pdvars):
    """Generate vector string

    This returns a string that will be used to print a line of code. If the
    variable 'tv' has not yet been used before, then the declaration of this
    variable must be included in the string.

    Parameters
    ----------
    tv : str
        The new temporary variable to print
    pdvars : list
        List of previously declared variables
    """
    st = "  "
    if tv not in pdvars:
        st += "double "
        pdvars.add(tv)
    return st


def gen_vector_code(ex, vsym, vlist, oper, prevdefvars, idx):
    """Generate vectorized code from an expression

    This function takes the expressions and generates vector
    code to be used in the C++ implementation.

    Parameters
    ----------
    ex : sympy.Function, sympy.Pow, sympy.Expression
        The expression to create the code for. Note that "expression"
        here just means SymPy generated values.
    vsym : list
        Numbered symbols for use
    vlist : list
        An empty list that will be used to process the tree on return
    oper : dict
        Dictionary used to map the '+' and '*' operators
    prevdefvars : dict
        An empty set used to identify previously defined temporary
        variables.
    idx : str
        The name of the index used for accessing arrays. '[pp]'
    """

    one = sym.symbols("one")
    negone = sym.symbols("negone")
    # print (vlist)
    if isinstance(ex, sym.Function):
        # check to see if we are processing a derivative
        if (
            isinstance(ex, nr.ad)
            or isinstance(ex, nr.d)
            or isinstance(ex, nr.kod)
            or isinstance(ex, nr.d2s)
        ):
            # print('...ex and args: ',ex,ex.func,ex.args)
            tv = next(vsym)
            vlist.append(tv)
            st = vec_print_str(tv, prevdefvars)
            str_args = [repr(a) for a in ex.args]
            o1 = oper["load"]
            o1s = repr(o1).replace("'", "")
            idxn = idx.replace("[", "")
            idxn = idxn.replace("]", "")
            st += (
                repr(tv)
                + " = "
                + o1s
                + "("
                + repr(ex.func)
                + "_"
                + "_".join(str_args)
                + "+"
                + idxn
                + " );"
            )
            # st += repr(tv) + ' = ' + repr(ex) + ';'
            print(st.replace(idx, ""))
            return

    if isinstance(ex, sym.Pow):
        # check to see if we are processing a simple pow
        a1, a2 = ex.args
        # print('processing pow...',ex,a1,a2)
        if isinstance(a1, sym.Symbol) and isinstance(a2, sym.Number):
            # This is a simple Pow function. Process it here and return
            tv = next(vsym)
            vlist.append(tv)
            st = vec_print_str(tv, prevdefvars)
            if a2 == -1:
                st += repr(tv) + " = 1.0 / " + repr(a1) + ";"
            elif a2 == 2:
                st += repr(tv) + " = " + repr(a1) + " * " + repr(a1) + ";"
            else:
                st += repr(tv) + " = pow( " + repr(a1) + ", " + repr(a2) + ");"
            print(st)
            return

    # recursively process the arguments of the function or operator
    for arg in ex.args:
        gen_vector_code(arg, vsym, vlist, oper, prevdefvars, idx)

    if isinstance(ex, sym.Number):
        if isinstance(ex, sym.Integer) and ex == 1:
            vlist.append(one)
        elif isinstance(ex, sym.Number) and ex == -1:
            vlist.append(negone)
        else:
            tv = next(vsym)
            vlist.append(tv)
            st = vec_print_str(tv, prevdefvars)
            if isinstance(ex, sym.Rational):
                st += repr(tv) + " = " + repr(float(ex)) + ";"
            else:
                st += repr(tv) + " = " + repr(ex) + ";"
            print(st)

    elif isinstance(ex, sym.Symbol):
        tv = next(vsym)
        vlist.append(tv)
        st = vec_print_str(tv, prevdefvars)
        st += repr(tv) + " = " + repr(ex) + ";"
        print(st)

    elif isinstance(ex, sym.Mul):
        nargs = len(ex.args)
        # print('mul..',len(vlist))
        for i in range(nargs - 1):
            tv = next(vsym)
            st = vec_print_str(tv, prevdefvars)
            st += repr(tv) + " = "
            v1 = vlist.pop()
            v2 = vlist.pop()
            # st += repr(v1) + ' * ' + repr(v2) + ';'
            o1 = oper["mul"]
            st += repr(o1) + "(" + repr(v1) + ", " + repr(v2) + ");"
            print(st.replace("'", ""))
            vlist.append(tv)

    elif isinstance(ex, sym.Add):
        nargs = len(ex.args)
        # print('add..',len(vlist))
        for i in range(nargs - 1):
            tv = next(vsym)
            st = vec_print_str(tv, prevdefvars)
            st += repr(tv) + " = "
            v1 = vlist.pop()
            v2 = vlist.pop()
            o1 = oper["add"]
            st += repr(o1) + "(" + repr(v1) + ", " + repr(v2) + ");"
            print(st.replace("'", ""))
            vlist.append(tv)

    elif isinstance(ex, sym.Pow):
        tv = next(vsym)
        qexp = vlist.pop()
        qman = vlist.pop()
        a1, a2 = ex.args
        o1 = oper["mul"]
        if isinstance(a2, sym.Integer):
            if a2 == -1:
                st = vec_print_str(tv, prevdefvars)
                st += repr(tv) + " =  1.0 / " + repr(qman) + ";"

            elif a2 == 2:
                st = vec_print_str(tv, prevdefvars)
                st += (
                    repr(tv)
                    + " = "
                    + repr(o1)
                    + "("
                    + repr(qman)
                    + ", "
                    + repr(qman)
                    + ");"
                )

            elif a2 == -2:
                v1 = next(vsym)
                st = vec_print_str(v1, prevdefvars)
                st += (
                    repr(v1)
                    + " = "
                    + repr(o1)
                    + "("
                    + repr(qman)
                    + ", "
                    + repr(qman)
                    + ");"
                )
                print(st.replace("'", ""))
                st = vec_print_str(tv, prevdefvars)
                st += repr(tv) + " = 1.0 / " + repr(v1) + ";"

            elif a2 > 2 and a2 < 8:
                v1 = next(vsym)
                st = vec_print_str(v1, prevdefvars)
                st += (
                    repr(v1)
                    + " = "
                    + repr(o1)
                    + "("
                    + repr(qman)
                    + ", "
                    + repr(qman)
                    + ");"
                )
                print(st.replace("'", ""))

                for i in range(a2 - 3):
                    v2 = next(vsym)
                    st = vec_print_str(v2, prevdefvars)
                    st += (
                        repr(v2)
                        + " = "
                        + repr(o1)
                        + "("
                        + repr(v1)
                        + ", "
                        + repr(qman)
                        + ");"
                    )
                    print(st.replace("'", ""))
                    v1 = v2

                st = vec_print_str(tv, prevdefvars)
                st += (
                    repr(tv)
                    + " = "
                    + repr(o1)
                    + "("
                    + repr(v1)
                    + ", "
                    + repr(qman)
                    + ");"
                )

            else:
                st = vec_print_str(tv, prevdefvars)
                st += repr(tv) + " = pow(" + repr(qman) + "," + repr(qexp) + ");"

        else:
            st = vec_print_str(tv, prevdefvars)
            st = repr(tv) + " = pow(" + repr(qman) + "," + repr(qexp) + ");"

        print(st.replace("'", ""))
        vlist.append(tv)


# TODO: rename, this is the wrong function, this just yanks out the values
def gen_enum_info(
    var_names: list,
    enum_name: str = "VAR",
    enum_prefix: str = "U",
    enum_start_idx: int = 0,
):
    """Generates the strings for the enums present in grDef.h"""

    enum_text = f"enum {enum_name}\n{{\n"

    for ii, var_name in enumerate(var_names):
        enum_line = f"    {enum_prefix}_{var_name.upper()}"
        enum_line += f" = {enum_start_idx}" if ii == 0 else ""
        enum_line += ",\n" if ii != len(var_names) - 1 else "\n"
        enum_text += enum_line

    enum_text += "};"

    return enum_text


def gen_var_info(
    var_names: list,
    zip_var_name: str = "uZipVars",
    offset_name: str = "offset",
    enum_name: str = "VAR",
    enum_prefix: str = "U",
    use_const: bool = True,
    enum_start_idx: int = 0,
    enum_var_names: list = [],
    dtype: str = "double",
    num_spaces: int = 0,
):
    """Generates the allocation variables in physcon.cpp

    If enum_var_names is empty, then it'll just use "upper"
    on the var names input. Other wise, the list should match
    one-to-one with the incoming variable names.
    """

    if enum_var_names:
        assert len(var_names) == len(enum_var_names), (
            "The input list sizes do not match"
        )

    physcon_text = ""

    for ii, var_name in enumerate(var_names):
        if enum_var_names:
            enum_entry = enum_var_names[ii]
            enum_prefix_use = ""
        else:
            enum_entry = var_name.upper()
            enum_prefix_use = f"{enum_prefix}_"

        phys_con_line = "".join(" " for i in range(num_spaces))
        # NOTE: before I had it without the ` const `
        phys_con_line += f"{'const ' if use_const else ''}{dtype} *const "
        phys_con_line += f"{var_name} = &"
        phys_con_line += f"{zip_var_name}["
        phys_con_line += f"{enum_name}::{enum_prefix_use}{enum_entry}"
        if offset_name == "":
            phys_con_line += f"];\n"
        else:
            phys_con_line += f"][{offset_name}];\n"

        physcon_text += phys_con_line

    return physcon_text


def gen_var_struct(
    member_names: list,
    enum_var_names: list,
    struct_name: str = "out",
    zip_var_name: str = "unzipVarsRHS",
    offset_name: str = "offset",
    enum_name: str = "VAR",
    dtype: str = "double",
    use_const: bool = False,
):
    """Emit a grouped per-block I/O pointer struct.

    Produces `struct { T *a; T *b; } NAME;` followed by
    `NAME.a = &zip[ENUM::E][offset];` assignments, so generated equation bodies
    read `NAME.var` -- the per-block pointers are grouped under one name
    (`in`/`out`) that states each variable's role. Compiles to the same
    pointers as the flat extraction (zero overhead), just clearer to read/edit.
    """
    assert len(member_names) == len(enum_var_names), (
        "member/enum list sizes do not match"
    )
    cq = "const " if use_const else ""
    out = "    struct {\n"
    for m in member_names:
        out += f"        {cq}{dtype} *{m};\n"
    out += f"    }} {struct_name};\n"
    for m, e in zip(member_names, enum_var_names):
        out += f"    {struct_name}.{m} = &{zip_var_name}[{enum_name}::{e}][{offset_name}];\n"
    return out


def gen_var_name_array(
    enum_names: list, project_name: str = "ccz4", list_name_inner: str = "VAR"
):
    name_array_text = "static const char *"
    name_array_text += f"{project_name.upper()}_{list_name_inner}_NAMES"

    name_array_text += "[] = {"

    for ii, enum_name in enumerate(enum_names):
        name_array_text += f'"{enum_name}"'
        name_array_text += ", " if ii != len(enum_names) - 1 else ""

    return name_array_text + "};\n"


def gen_var_iterable_list(
    enum_names: list, project_name: str = "ccz4", list_name_inner: str = "VAR"
):
    name_array_text = f"static const {list_name_inner.upper()} "
    name_array_text += f"{project_name.upper()}_{list_name_inner}_ITERABLE_LIST"
    name_array_text += "[] = {"

    for ii, enum_name in enumerate(enum_names):
        name_array_text += f"{enum_name}"
        name_array_text += ", " if ii != len(enum_names) - 1 else ""

    return name_array_text + "};\n"


def generate_memory_alloc(
    var_names: list,
    var_type: str = "double",
    use_old_method=False,
    include_byte_declaration=False,
    start_id: int = 0,
):
    if use_old_method:
        if include_byte_declaration:
            return_text = f"const unsigned int bytes = n * sizeof({var_type});\n"
        else:
            return_text = ""

        for va in var_names:
            return_text += f"{var_type} *{va} = ({var_type} *)malloc(bytes);\n"

        last_id = 0
    else:
        return_text = ""
        for ii, va in enumerate(var_names):
            return_text += f"{var_type} *{va} = deriv_base + {start_id} * BLK_SZ;\n"
            start_id += 1

    return return_text, start_id


def generate_memory_dealloc(var_names: list):
    return_text = ""

    for va in var_names:
        return_text += f"free({va});\n"

    return return_text


def generate_deriv_comp(var_names: list, adv_der_var: str = "beta",
                        deriv_obj: str = ""):
    # map dir
    dir_map = {"0": "x", "1": "y", "2": "z"}

    # When deriv_obj is set (e.g. "SOLVER_DERIVS"), emit method calls like:
    #   SOLVER_DERIVS->grad_x(out, in, hx, sz, bflag);
    # When empty, emit legacy function calls like:
    #   deriv_x(out, in, hx, sz, bflag);
    use_obj = deriv_obj != ""
    prefix = f"{deriv_obj}->" if use_obj else ""

    return_text = ""

    for va in var_names:
        # split by underscore
        split_va = va.split("_")

        if split_va[0] == "grad":
            # get the original var name from the string, just in case
            # there are more underscores
            va_name = "".join(short_text + "_" for short_text in split_va[2:])[:-1]
            # get the direction
            the_dir = dir_map[split_va[1]]
            # first-order derivative: grad_x, grad_y, or grad_z
            if use_obj:
                return_text += (
                    f"{prefix}grad_{the_dir}({va}, {va_name}, "
                    + f"h{the_dir}, sz, bflag);\n"
                )
            else:
                return_text += (
                    f"deriv_{the_dir}({va}, {va_name}, "
                    + f"h{the_dir}, sz, bflag);\n"
                )

        elif split_va[0] == "grad2":
            # get the original var name from the string, just in case
            # there are more underscores
            va_name = "".join(short_text + "_" for short_text in split_va[3:])[:-1]

            dir_int = split_va[1]
            dir_int2 = split_va[2]

            # get the direction
            the_dir = dir_map[dir_int]
            the_dir2 = dir_map[dir_int2]

            if the_dir == the_dir2:
                # second-order same-direction: grad_xx, grad_yy, or grad_zz
                if use_obj:
                    return_text += (
                        f"{prefix}grad_{the_dir}{the_dir}({va}, {va_name}, "
                        + f"h{the_dir}, sz, bflag);\n"
                    )
                else:
                    return_text += (
                        f"deriv_{the_dir}{the_dir}({va}, {va_name}, "
                        + f"h{the_dir}, sz, bflag);\n"
                    )

            else:
                # mixed second-order: compute from the first-direction gradient
                if use_obj:
                    grad2_str = f"{prefix}grad_{the_dir2}({va}, "
                else:
                    grad2_str = f"deriv_{the_dir2}({va}, "

                grad2_str += (
                    f"grad_{dir_int}_" + va_name + f", h{the_dir2}, sz, bflag);\n"
                )

                return_text += grad2_str

        elif split_va[0] == "agrad":
            # advective derivative
            va_name = "".join(short_text + "_" for short_text in split_va[2:])[:-1]
            the_dir = dir_map[split_va[1]]
            dir_idx = split_va[1]
            # advective derivatives keep the legacy pattern for now
            # since DendroDerivatives doesn't have an advective method
            return_text += (
                f"adv_deriv_{the_dir}({va}, {va_name}, "
                + f"h{the_dir}, sz, {adv_der_var}{dir_idx}, bflag);\n"
            )

        else:
            raise NotImplementedError("Sorry, but this hasn't been implemented yet")

    return return_text


def generate_bcs_function_call(
    var_rhs_name,
    var_name,
    f_falloff,
    f_asymptotic,
    deriv_names=[],
    prefix="ccz4",
    pmin="pmin",
    pmax="pmax",
    sz="sc",
    bflag="bflag",
):
    if deriv_names:
        assert len(deriv_names) == 3, "Not enough entries in the deriv names"

    temp_str = f"{prefix}_bcs("
    temp_str += f"{var_rhs_name}, "
    temp_str += f"{var_name}, "

    if deriv_names:
        temp_str += ", ".join(x for x in deriv_names)
    else:
        temp_str += ", ".join(f"grad_{ii}_{var_name}" for ii in range(3))

    temp_str += f", {pmin}, {pmax}, "

    temp_str += f"{float(f_falloff)}, {float(f_asymptotic)},"
    temp_str += f" {sz}, {bflag});\n"

    return temp_str


def generate_bcs_table(
    rows,
    prefix="ccz4",
    pmin="pmin",
    pmax="pmax",
    sz="sz",
    bflag="bflag",
):
    """Generate one data-table + loop for the Sommerfeld BCs.

    `rows` is a list of (var_rhs_name, var_name, deriv_names, falloff,
    asymptotic). Emits a local `{rhs, field, dx, dy, dz, falloff, asymptotic}`
    table and a single `{prefix}_bcs(...)` loop over it, instead of N unrolled
    calls. Bit-identical (same args, same order) -- just far less generated
    clutter, and the per-variable falloff/asymptotic read as a data block.
    """
    if not rows:
        return ""

    out = (
        "    // per-variable sommerfeld bc data: {rhs, field, dx, dy, dz,"
        " falloff, asymptotic}\n"
        "    struct __bc_row {\n"
        "        double *rhs;\n"
        "        const double *f, *gx, *gy, *gz;\n"
        "        double falloff, asymptotic;\n"
        "    };\n"
        "    const __bc_row __bc_rows[] = {\n"
    )
    for var_rhs_name, var_name, deriv_names, falloff, asymptotic in rows:
        if deriv_names:
            assert len(deriv_names) == 3, "Not enough entries in the deriv names"
            gx, gy, gz = deriv_names
        else:
            gx, gy, gz = (f"grad_{ii}_{var_name}" for ii in range(3))
        out += (
            f"        {{{var_rhs_name}, {var_name}, {gx}, {gy}, {gz},"
            f" {float(falloff)}, {float(asymptotic)}}},\n"
        )
    out += "    };\n"
    out += "    for (const auto &__r : __bc_rows)\n"
    out += (
        f"        {prefix}_bcs(__r.rhs, __r.f, __r.gx, __r.gy, __r.gz,"
        f" {pmin}, {pmax}, __r.falloff, __r.asymptotic, {sz}, {bflag});\n"
    )
    return out


def generate_force_sym_matrix_det_to_one(
    vname, unzip_access, uzip="uiVar", node="node", dtype="double"
):
    # NOTE: the function already should have `one_third` defined

    return_str = ""

    # then calculate the determinant
    return_str += f"{dtype} det_{vname} = "
    # first term
    det_str = f"{vname}[0][0] * "
    det_str += f"({vname}[1][1] * {vname}[2][2] "
    det_str += f"- {vname}[1][2] * {vname}[1][2])"

    # second term
    det_str += f" - {vname}[0][1] * {vname}[0][1] * {vname}[2][2]"

    # third term
    det_str += f" + 2.0 * {vname}[0][1] * {vname}[0][2] * {vname}[1][2]"

    # fourth term
    det_str += f" - {vname}[0][2] * {vname}[0][2] * {vname}[1][1]"

    # add the determinant calculation to the return str
    return_str += det_str + ";\n\n"

    # TODO: what can we do to fix the metric determinant being negative?
    # for now, we exit
    return_str += f"if (det_{vname} < 0.0){{\n"
    return_str += f'    std::cout << "Determinant of {vname} is negative: " '
    return_str += f"<< det_{vname} << std::endl;\n"
    return_str += "    exit(0);\n}\n"

    # now add the calculation for negative third
    return_str += (
        f"{dtype} det_{vname}_to_neg_third = " + f"1.0 / pow(det_{vname}, one_third);\n"
    )

    # now we go and update all of the values inside the matrix
    return_str += "for (unsigned int j = 0; j < 3; j++)\n{\n"
    return_str += "    for (unsigned int i = 0; i < 3; i++)\n"
    return_str += "{\n"
    return_str += f"        {vname}[i][j] *= det_{vname}_to_neg_third;\n"
    return_str += "    }\n}\n\n"

    # recalculate the determinant
    return_str += f"det_{vname} = " + det_str + ";\n\n"

    # now if it's greater than one, we've got an issue
    return_str += f"if (fabs(det_{vname} - 1.0) > 1.0e-6) {{\n"
    return_str += "    std::cout.precision(14);\n"
    return_str += f'    std::cout << "det({vname}) != 1.0 det="'
    return_str += f" << std::fixed << det_{vname} << std::endl;\n"
    # then print out variable info
    return_str += f'    std::cout << "    {vname}(1,1)" << '
    return_str += f"{vname}[0][0] << std::endl;\n"
    return_str += f'    std::cout << "    {vname}(1,2)" << '
    return_str += f"{vname}[0][1] << std::endl;\n"
    return_str += f'    std::cout << "    {vname}(1,3)" << '
    return_str += f"{vname}[0][1] << std::endl;\n"
    return_str += f'    std::cout << "    {vname}(2,2)" << '
    return_str += f"{vname}[1][1] << std::endl;\n"
    return_str += f'    std::cout << "    {vname}(2,3)" << '
    return_str += f"{vname}[1][2] << std::endl;\n"
    return_str += f'    std::cout << "    {vname}(3,3)" << '
    return_str += f"{vname}[2][2] << std::endl;\n"
    # exit, and close the block
    return_str += "exit(0);\n}\n\n"

    # then we can make the calculations to update everything
    return_str += f"double {vname}_up[3][3];\n"
    return_str += f"double idet_{vname} = 1.0 / det_{vname};\n"
    # fill in this new matrix

    return_str += (
        f"{vname}_up[0][0] = idet_{vname} * "
        + f"({vname}[1][1] * {vname}[2][2] - {vname}[1][2] * {vname}[1][2]);\n"
    )
    return_str += (
        f"{vname}_up[0][1] = idet_{vname} * "
        + f"(-{vname}[0][1] * {vname}[2][2] + {vname}[0][2] * {vname}[1][2]);\n"
    )
    return_str += (
        f"{vname}_up[0][2] = idet_{vname} * "
        + f"({vname}[0][1] * {vname}[1][2] - {vname}[0][2] * {vname}[1][1]);\n"
    )
    return_str += (
        f"{vname}_up[1][1] = idet_{vname} * "
        + f"({vname}[0][0] * {vname}[2][2] - {vname}[0][2] * {vname}[0][2]);\n"
    )
    return_str += (
        f"{vname}_up[1][2] = idet_{vname} * "
        + f"(-{vname}[0][0] * {vname}[1][2] + {vname}[0][1] * {vname}[0][2]);\n"
    )
    return_str += (
        f"{vname}_up[2][2] = idet_{vname} * "
        + f"({vname}[0][0] * {vname}[1][1] - {vname}[0][1] * {vname}[0][1]);\n"
    )
    return_str += f"{vname}_up[1][0] = {vname}_up[0][1];\n"
    return_str += f"{vname}_up[2][0] = {vname}_up[0][2];\n"
    return_str += f"{vname}_up[2][1] = {vname}_up[1][2];\n"

    return return_str + "\n"


def generate_force_symmat_traceless(vname, metric_vname, dtype="double"):
    return_str = ""

    # calculate one third of the trace
    return_str += "//// one third of the trace:\n"
    return_str += f"{dtype} ot_trace_{vname} = "
    return_str += "one_third * ("
    return_str += f"{vname}[0][0] * {metric_vname}_up[0][0]"
    return_str += f" + {vname}[1][1] * {metric_vname}_up[1][1]"
    return_str += f" + {vname}[2][2] * {metric_vname}_up[2][2]"
    return_str += " + 2.0 * ("
    return_str += f"{vname}[0][1] * {metric_vname}_up[0][1]"
    return_str += f" + {vname}[0][2] * {metric_vname}_up[0][2]"
    return_str += f" + {vname}[1][2] * {metric_vname}_up[1][2]"
    return_str += "));\n\n"

    # then update the matrix by subtracing off a third of the trace
    # and the original metric
    return_str += f"{vname}[0][0] -= ot_trace_{vname} * {metric_vname}[0][0];\n"
    return_str += f"{vname}[0][1] -= ot_trace_{vname} * {metric_vname}[0][1];\n"
    return_str += f"{vname}[0][2] -= ot_trace_{vname} * {metric_vname}[0][2];\n"
    return_str += f"{vname}[1][1] -= ot_trace_{vname} * {metric_vname}[1][1];\n"
    return_str += f"{vname}[1][2] -= ot_trace_{vname} * {metric_vname}[1][2];\n"
    return_str += f"{vname}[2][2] -= ot_trace_{vname} * {metric_vname}[2][2];\n"

    # then calculate the trace again but without the one third
    return_str += "//// now the actual trace:\n"
    return_str += f"{dtype} trace_{vname} = "
    return_str += f"{vname}[0][0] * {metric_vname}_up[0][0]"
    return_str += f" + {vname}[1][1] * {metric_vname}_up[1][1]"
    return_str += f" + {vname}[2][2] * {metric_vname}_up[2][2]"
    return_str += " + 2.0 * ("
    return_str += f"{vname}[0][1] * {metric_vname}_up[0][1]"
    return_str += f" + {vname}[0][2] * {metric_vname}_up[0][2]"
    return_str += f" + {vname}[1][2] * {metric_vname}_up[1][2]"
    return_str += ");\n\n"

    # then the actual check that it is zero
    return_str += f"if (fabs(trace_{vname}) > 1.0e-6)\n"
    return_str += "{\n    "
    return_str += (
        f'std::cout << "ERROR: tr({vname}) != 0, ="'
        + f"  << trace_{vname} << std::endl;\n"
    )
    cout_str = '    std::cout << "       '
    cout_end_str = " << std::endl;\n"
    return_str += cout_str + f'{vname}(1,1)=" << {vname}[0][0]' + cout_end_str
    return_str += cout_str + f'{vname}(1,2)=" << {vname}[0][1]' + cout_end_str
    return_str += cout_str + f'{vname}(1,3)=" << {vname}[0][2]' + cout_end_str
    return_str += cout_str + f'{vname}(2,2)=" << {vname}[1][1]' + cout_end_str
    return_str += cout_str + f'{vname}(2,3)=" << {vname}[1][2]' + cout_end_str
    return_str += cout_str + f'{vname}(3,3)=" << {vname}[2][2]' + cout_end_str
    return_str += "    exit(0);\n}\n\n"

    return return_str


def generate_update_sym_mat_extract(
    vname, unzip_access, uzip="uiVar", dtype="double", node="node"
):
    # NOTE: all of the symmetric matrices grab the indexing based on the upper
    # triangle, so get those first
    midx = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
    # that leaves the remaining three of
    midx_exc = [[1, 0], [2, 0], [2, 1]]

    # so we need to generate a quick matrix for them and extract it out
    return_str = f"{dtype} {vname}[3][3];\n\n"

    # then generate the extraction to fill the matrix
    for ii, idxs in enumerate(midx):
        return_str += f"{vname}[{idxs[0]}][{idxs[1]}]"
        return_str += f" = {uzip}[{unzip_access[ii]}][{node}];\n"

    # then copy over the other side
    for idxs in midx_exc:
        return_str += f"{vname}[{idxs[0]}][{idxs[1]}]"
        return_str += f" = {vname}[{idxs[1]}][{idxs[0]}];\n"

    return return_str + "\n"


def generate_update_sym_mat_code(
    vname, unzip_access, uzip="uiVar", node="node", include_up=True
):
    return_str = ""

    # NOTE: all of the symmetric matrices grab the indexing based on the upper
    # triangle, so get those first
    midx = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    # then generate the extraction to fill the matrix
    for ii, idxs in enumerate(midx):
        return_str += f"{uzip}[{unzip_access[ii]}][{node}]"
        return_str += (
            f" = {vname}"
            + f"{'_up' if include_up else ''}"
            + f"[{idxs[0]}][{idxs[1]}];\n"
        )

    return return_str + "\n"


def generate_variable_always_positive(
    uzip_access, floor_var=None, uzip="uiVar", node="node"
):
    if floor_var is None:
        floor_var = "1.0e-6"

    return (
        f"{uzip}[{uzip_access}][{node}] = std::max("
        + f"{uzip}[{uzip_access}][{node}], {floor_var});\n"
    )
