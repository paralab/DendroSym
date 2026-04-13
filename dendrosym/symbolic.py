import sympy as sym
import multiprocessing as mp

import dendrosym


def expand_derivatives(expr, found_derivatives: list, id_str: str = "[pp]"):
    expr = dendrosym.helpers._DerTransformer(var_names, idx_str)(expr)

    return


def replace_sympy_derivatives(expr: list, var_names, idx_str):
    print("  Now attempting to replace derivatives...")
    # symbols we need and will replace
    x, y, z = sym.symbols("xtem ytem ztem")
    d_map = {x: 0, y: 1, z: 2}

    # find derivatives once
    all_derivatives = expr.atoms(sym.Derivative)

    if not all_derivatives:
        return expr

    replacement_map = {}
    for deriv in all_derivatives:
        term_to_diff = deriv.expr

        # variable_count gives a list of ttuples
        counts = deriv.variable_count

        if len(counts) == 1:
            # first or second derivative in single direction
            variable, order = counts[0]
            idx = d_map[variable]

            if order == 1:
                # first derivative
                new_expr = dendrosym.nr.d(idx, term_to_diff)
            elif order == 2:
                # second derivative
                new_expr = dendrosym.nr.d2s(idx, idx, term_to_diff)
            else:
                raise NotImplementedError(f"Derivative order {order} is not supported!")
        elif len(counts) == 2:
            # mixed partial derivatives
            (var1, order1), (var2, order2) = counts
            if order1 != 1 or order2 != 1:
                raise NotImplementedError(
                    "Derivatives of order > 1 in mixed partials are not supported!"
                )

            idx1, idx2 = d_map[var1], d_map[var2]

            # ensure the indices are sorted
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1

            new_expr = dendrosym.nr.d2s(idx1, idx2, term_to_diff)

        else:
            raise NotImplementedError(
                "Derivatives with respect to three or more variables are not supported!"
            )

        replacement_map[deriv] = new_expr

    expr = expr.xreplace(replacement_map)

    return expr
