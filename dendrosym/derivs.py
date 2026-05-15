import sympy as sym


import dendrosym

i = "i"
j = "j"
k = "k"

hx = "hx"
hy = "hy"
hz = "hz"

variable_strs = []
idx_str = "[pp]"

xx, yy, zz = sym.symbols("xx_temp yy_temp zz_temp")

symbols_find = None
symbols_replace = None


def first_der(dir, expr):
    global variable_strs, xx, yy, zz, idx_str, symbols_find, symbols_replace

    if expr == 0:
        return 0

    d_order = (xx, yy, zz)

    if symbols_find is None or symbols_replace is None:
        symbols_find = []
        symbols_replace = []

        for xr in variable_strs:
            # shouldn't end with pp
            if xr.endswith(idx_str):
                xr = xr[: -len(idx_str)]

            # add the symbol version with both normal and idx str
            symbols_find.append(sym.Symbol(xr))
            symbols_find.append(sym.Symbol(xr + idx_str))

            # build up the functional version so we can take the derivative
            to_replace = sym.Function(sym.Symbol(xr))(xx, yy, zz)

            symbols_replace.append(to_replace)
            symbols_replace.append(to_replace)

    # create a dictionary via zipping
    all_repl = dict(zip(symbols_find, symbols_replace))

    # do the full replacement
    expr = expr.xreplace(all_repl)

    # return the derivative version
    return sym.diff(expr, d_order[dir])


def second_der(dir1, dir2, expr):
    global variable_strs, xx, yy, zz, idx_str, symbols_find, symbols_replace

    if expr == 0:
        return 0

    d_order = (xx, yy, zz)

    if symbols_find is None or symbols_replace is None:
        symbols_find = []
        symbols_replace = []

        for xr in variable_strs:
            # shouldn't end with pp
            if xr.endswith(idx_str):
                xr = xr[: -len(idx_str)]

            # add the symbol version with both normal and idx str
            symbols_find.append(sym.Symbol(xr))
            symbols_find.append(sym.Symbol(xr + idx_str))

            # build up the functional version so we can take the derivative
            to_replace = sym.Function(sym.Symbol(xr))(xx, yy, zz)

            symbols_replace.append(to_replace)
            symbols_replace.append(to_replace)

    # create a dictionary via zipping
    all_repl = dict(zip(symbols_find, symbols_replace))

    # do the full replacement
    expr = expr.xreplace(all_repl)

    # then we can build up everything
    return sym.diff(expr, d_order[dir1], d_order[dir2])


def get_derivs():
    dendrosym.nr.set_first_derivative(first_der)
    dendrosym.nr.set_second_derivative(second_der)

    # set advective to first_der just in case
    dendrosym.nr.set_advective_derivative(first_der)

    return first_der, second_der


def restore_original_derivatives(expr):
    """Restores an expression's derivatives to "grad" format

    This function in particular is designed to help us take derivatives
    that we replaced in first_der() and second_der() back into the grad


    Args:
        expr (sympy.core.expr.Expr): The expression to fix

    Returns:
        sympy.core.expr.Expr: The fixed up expression
    """
    global variable_strs, xx, yy, zz, idx_str

    d_order = (xx, yy, zz)

    # now we want it all with no repeats!!!!!
    symbols_find_norep = [sym.Symbol(xr + idx_str) for xr in variable_strs]
    symbols_replace_norep = [
        sym.Function(sym.Symbol(xr))(xx, yy, zz) for xr in variable_strs
    ]
    reverse_repl = dict(zip(symbols_replace_norep, symbols_find_norep))

    all_derivatives = expr.atoms(sym.Derivative)

    if len(all_derivatives) == 0:
        return expr

    # print(all_derivatives)

    while len(all_derivatives) > 0:
        curr_deriv = all_derivatives.pop()

        # print(curr_deriv)

        if len(curr_deriv.args) == 2:
            # single direction derivatives
            diff_term, diff_dir = curr_deriv.args

            if diff_dir[1] > 2:
                raise NotImplemented(
                    "A 3rd or higher order derivative was found! This isn't implemented"
                )
            idx_use = -1

            for ii, val in enumerate(d_order):
                if diff_dir[0] == val:
                    idx_use = ii

            if diff_dir[1] == 1:
                new_expr = dendrosym.nr.d_old(idx_use, diff_term)
            else:
                new_expr = dendrosym.nr.d2s_old(idx_use, idx_use, diff_term)

        elif len(curr_deriv.args) == 3:
            diff_term, diff_dir1, diff_dir2 = curr_deriv.args

            if diff_dir1[1] > 1 or diff_dir2[1] > 1:
                raise NotImplemented(
                    "A 3rd or higher order derivative was found after expanding!"
                )

            idx_1 = -1
            for ii, val in enumerate(d_order):
                if diff_dir1[0] == val:
                    idx_1 = ii

            idx_2 = -1
            for ii, val in enumerate(d_order):
                if diff_dir2[0] == val:
                    idx_2 = ii

            idxtmp = 0

            if idx_1 > idx_2:
                # swapping indices
                idxtmp = idx_1
                idx_1 = idx_2
                idx_2 = idxtmp

            new_expr = dendrosym.nr.d2s_old(idx_1, idx_2, diff_term)

        else:
            raise NotImplemented(
                "A different derivative type was found, this isn't implemented"
            )

        new_expr = new_expr.subs(reverse_repl)

        expr = expr.xreplace({curr_deriv: new_expr})

        all_derivatives = expr.atoms(sym.Derivative)

    return expr


_RESTORE_CACHE = {"key": None, "dict": None}


def restore_only_symbols(expr):
    # build the reverse-replacement dict once per (variable_strs, idx_str)
    # tuple and memoize. previously rebuilt per call (~50-100x in a loop).
    key = (tuple(variable_strs), idx_str)
    if _RESTORE_CACHE["key"] != key:
        symbols_find_norep = [sym.Symbol(xr + idx_str) for xr in variable_strs]
        symbols_replace_norep = [
            sym.Function(sym.Symbol(xr))(xx, yy, zz) for xr in variable_strs
        ]
        _RESTORE_CACHE["key"] = key
        _RESTORE_CACHE["dict"] = dict(zip(symbols_replace_norep, symbols_find_norep))

    return expr.xreplace(_RESTORE_CACHE["dict"])


def set_i_j_k(i_in, j_in, k_in):
    global i, j, k
    i = i_in
    j = j_in
    k = k_in


def set_hx_hy_hz(hx_in, hy_in, hz_in):
    global hx, hy, hz
    hx = hx_in
    hy = hy_in
    hz = hz_in


def build_first_deriv_stencil_by_dim_idx(symbol, order, dim, idx_str="[pp]"):
    if dim == 0:
        return build_stencil_dx(symbol, order, idx_str=idx_str)
    elif dim == 1:
        return build_stencil_dy(symbol, order, idx_str=idx_str)
    elif dim == 2:
        return build_stencil_dz(symbol, order, idx_str=idx_str)
    else:
        raise Exception(
            "Cannot build a derivative stencil for larger than 3 dimensions!"
        )


def build_second_deriv_stencil_by_dim_idx(symbol, order, dim1, dim2, idx_str="[pp]"):
    # swap dim1 and dim2 so they're in order since we can swap things
    if dim1 == dim2:
        if dim1 == 0:
            return build_stencil_dxdx(symbol, order, idx_str=idx_str)
        elif dim1 == 1:
            return build_stencil_dydy(symbol, order, idx_str=idx_str)
        elif dim1 == 2:
            return build_stencil_dzdz(symbol, order, idx_str=idx_str)
        else:
            raise Exception(
                "Cannot build a second order derivative stencil for larger than 3 dimensions!"
            )

    if dim1 > dim2:
        temp = dim1
        dim1 = dim2
        dim2 = temp

    if dim1 == 0:
        if dim2 == 1:
            return build_stencil_dxdy(symbol, order, idx_str=idx_str)
        elif dim2 == 2:
            return build_stencil_dxdz(symbol, order, idx_str=idx_str)
        else:
            raise Exception("Invalid second derivative stencil dimension options")
    elif dim1 == 1:
        if dim2 == 2:
            return build_stencil_dydz(symbol, order, idx_str=idx_str)
        else:
            raise Exception("Invalid second derivative stencil dimension options")

    # if we're at this point, we're dones
    raise Exception("Invalid second derivative stencil dimension options")


def build_stencil_dx(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_x_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_x_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dy(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_y_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_y_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dz(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_z_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_z_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dxdx(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_xx_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_xx_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dydy(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_yy_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_yy_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dzdz(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_zz_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_zz_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dxdy(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_xy_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_xy_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dxdz(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_xz_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_xz_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_stencil_dydz(symbol, order, idx_str="[pp]"):
    if idx_str.startswith("["):
        idx_str = idx_str[1:-1]

    if order == 4:
        return build_4th_stencil_yz_3d(symbol, idx_str=idx_str)
    elif order == 6:
        return build_6th_stencil_yz_3d(symbol, idx_str=idx_str)
    else:
        raise NotImplementedError("That stencil order has not yet been implemented")


def build_4th_stencil_x_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
    ) / (12 * sym.Symbol(hx))

    return expr


def build_4th_stencil_y_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
    ) / (12 * sym.Symbol(hy))

    return expr


def build_4th_stencil_z_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
    ) / (12 * sym.Symbol(hz))

    return expr


def build_4th_stencil_xx_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
        + 16 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
        - 30 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 16 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
    ) / (12 * sym.Symbol(hx) ** 2)

    return expr


def build_4th_stencil_yy_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
        + 16 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
        - 30 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 16 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
    ) / (12 * sym.Symbol(hy) ** 2)

    return expr


def build_4th_stencil_zz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
        + 16 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
        - 30 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 16 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
    ) / (12 * sym.Symbol(hy) ** 2)

    return expr


def build_4th_stencil_xy_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 2
    expr = (
        sym.Symbol(symbol + f"[IDX({i} - 2, {j} - 2, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} - 1, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} - 2, {j} + 2, {k})]")
    )

    # i - 1
    expr += (-8) * (
        sym.Symbol(symbol + f"[IDX({i} - 1, {j} - 2, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} - 1, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} - 1, {j} + 2, {k})]")
    )

    # i + 1
    expr += (8) * (
        sym.Symbol(symbol + f"[IDX({i} + 1, {j} - 2, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} - 1, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} + 1, {j} + 2, {k})]")
    )

    # i + 2
    expr += (-1) * (
        sym.Symbol(symbol + f"[IDX({i} + 2, {j} - 2, {k})]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} - 1, {k})]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} + 1, {k})]")
        - sym.Symbol(symbol + f"[IDX({i} + 2, {j} + 2, {k})]")
    )

    return expr / (12**2 * sym.Symbol(hx) * sym.Symbol(hy))


def build_4th_stencil_xz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 2
    expr = (
        sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} + 2)]")
    )

    # i - 1
    expr += (-8) * (
        sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} + 2)]")
    )

    # i + 1
    expr += (8) * (
        sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} + 2)]")
    )

    # i + 2
    expr += (-1) * (
        sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} + 2)]")
    )

    return expr / (12**2 * sym.Symbol(hx) * sym.Symbol(hz))


def build_4th_stencil_yz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 2
    expr = (
        sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} + 2)]")
    )

    # i - 1
    expr += (-8) * (
        sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} + 2)]")
    )

    # i + 1
    expr += (8) * (
        sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} + 2)]")
    )

    # i + 2
    expr += (-1) * (
        sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} - 2)]")
        - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} - 1)]")
        + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} + 1)]")
        - sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} + 2)]")
    )

    return expr / (12**2 * sym.Symbol(hy) * sym.Symbol(hz))


# =====================
# BEGIN 6TH ORDER STENCIL FUNCTIONS
# =====================


def build_6th_stencil_x_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
    ) / (60 * sym.Symbol(hx))

    return expr


def build_6th_stencil_x_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_x_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            -147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 360 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            - 450 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            + 400 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            - 225 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
            + 72 * sym.Symbol(symbol + f"[IDX({i} + 5, {j}, {k})]")
            - 10 * sym.Symbol(symbol + f"[IDX({i} + 6, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -10 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 150 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            - 100 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            + 50 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            - 15 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
            + 2 * sym.Symbol(symbol + f"[IDX({i} + 5, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            2 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 24 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 80 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            - 30 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            + 8 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            - 1 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))

    return expr


def build_6th_stencil_x_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_x_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +10 * sym.Symbol(symbol + f"[IDX({i} - 6, {j}, {k})]")
            - 72 * sym.Symbol(symbol + f"[IDX({i} - 5, {j}, {k})]")
            + 225 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 400 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 450 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 360 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            + 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -2 * sym.Symbol(symbol + f"[IDX({i} - 5, {j}, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 50 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 100 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 150 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            + 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 10 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +1 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 8 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 30 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 80 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            + 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 24 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            - 2 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
        ) / (60 * sym.Symbol(hx))

    return expr


def build_6th_stencil_y_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
    ) / (60 * sym.Symbol(hy))

    return expr


def build_6th_stencil_y_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_y_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            -147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 360 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            - 450 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            + 400 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            - 225 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
            + 72 * sym.Symbol(symbol + f"[IDX({i}, {j} + 5, {k})]")
            - 10 * sym.Symbol(symbol + f"[IDX({i}, {j} + 6, {k})]")
        ) / (60 * sym.Symbol(hy))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -10 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 150 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            - 100 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            + 50 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            - 15 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
            + 2 * sym.Symbol(symbol + f"[IDX({i}, {j} + 5, {k})]")
        ) / (60 * sym.Symbol(hy))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            2 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 24 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 80 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            - 30 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            + 8 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            - 1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
        ) / (60 * sym.Symbol(hy))

    return expr


def build_6th_stencil_y_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_y_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +10 * sym.Symbol(symbol + f"[IDX({i}, {j} - 6, {k})]")
            - 72 * sym.Symbol(symbol + f"[IDX({i}, {j} - 5, {k})]")
            + 225 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 400 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 450 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 360 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            + 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (60 * sym.Symbol(hy))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -2 * sym.Symbol(symbol + f"[IDX({i}, {j} - 5, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 50 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 100 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 150 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            + 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 10 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        ) / (60 * sym.Symbol(hy))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 8 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 30 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 80 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            + 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 24 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            - 2 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
        ) / (60 * sym.Symbol(hy))

    return expr


def build_6th_stencil_z_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
    ) / (60 * sym.Symbol(hz))

    return expr


def build_6th_stencil_z_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_z_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            -147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 360 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            - 450 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            + 400 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            - 225 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
            + 72 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 5)]")
            - 10 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 6)]")
        ) / (60 * sym.Symbol(hz))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -10 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 150 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            - 100 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            + 50 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            - 15 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
            + 2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 5)]")
        ) / (60 * sym.Symbol(hz))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 24 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 80 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            - 30 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            + 8 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            - 1 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
        ) / (60 * sym.Symbol(hz))

    return expr


def build_6th_stencil_z_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_z_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +10 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 6)]")
            - 72 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 5)]")
            + 225 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 400 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 450 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 360 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            + 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (60 * sym.Symbol(hz))
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 5)]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 50 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 100 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 150 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            + 77 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 10 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        ) / (60 * sym.Symbol(hz))
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +1 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 8 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 30 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 80 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            + 35 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 24 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            - 2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
        ) / (60 * sym.Symbol(hz))

    return expr


def build_6th_stencil_xx_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        2 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
        - 27 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
        + 270 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
        - 490 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 270 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        - 27 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
        + 2 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
    ) / (180 * sym.Symbol(hx) ** 2)

    return expr


def build_6th_stencil_xx_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_xx_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
            - 972 * sym.Symbol(symbol + f"[IDX({i} + 5, {j}, {k})]")
            + 137 * sym.Symbol(symbol + f"[IDX({i} + 6, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 255 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            + 470 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            - 285 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            + 93 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
            - 13 * sym.Symbol(symbol + f"[IDX({i} + 5, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            + 228 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 200 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
            - 12 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k})]")
            + 2 * sym.Symbol(symbol + f"[IDX({i} + 4, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)

    return expr


def build_6th_stencil_xx_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_xx_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i} - 6, {j}, {k})]")
            - 972 * sym.Symbol(symbol + f"[IDX({i} - 5, {j}, {k})]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            + 812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i} - 5, {j}, {k})]")
            + 93 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 285 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 470 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            - 255 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 137 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +2 * sym.Symbol(symbol + f"[IDX({i} - 4, {j}, {k})]")
            - 12 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k})]")
            + 200 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k})]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 228 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k})]")
            - 13 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k})]")
        ) / (180 * sym.Symbol(hx) ** 2)

    return expr


def build_6th_stencil_yy_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        2 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
        - 27 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
        + 270 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
        - 490 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 270 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        - 27 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
        + 2 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
    ) / (180 * sym.Symbol(hy) ** 2)

    return expr


def build_6th_stencil_yy_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_yy_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
            - 972 * sym.Symbol(symbol + f"[IDX({i}, {j} + 5, {k})]")
            + 137 * sym.Symbol(symbol + f"[IDX({i}, {j} + 6, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 255 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            + 470 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            - 285 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            + 93 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
            - 13 * sym.Symbol(symbol + f"[IDX({i}, {j} + 5, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            + 228 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 200 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
            - 12 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k})]")
            + 2 * sym.Symbol(symbol + f"[IDX({i}, {j} + 4, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)

    return expr


def build_6th_stencil_yy_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_yy_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i}, {j} - 6, {k})]")
            - 972 * sym.Symbol(symbol + f"[IDX({i}, {j} - 5, {k})]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            + 812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i}, {j} - 5, {k})]")
            + 93 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 285 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 470 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            - 255 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 137 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +2 * sym.Symbol(symbol + f"[IDX({i}, {j} - 4, {k})]")
            - 12 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k})]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k})]")
            + 200 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k})]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 228 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k})]")
            - 13 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k})]")
        ) / (180 * sym.Symbol(hy) ** 2)

    return expr


def build_6th_stencil_zz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    expr = (
        2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
        - 27 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
        + 270 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
        - 490 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        + 270 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        - 27 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
        + 2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
    ) / (180 * sym.Symbol(hz) ** 2)

    return expr


def build_6th_stencil_zz_3d_negative_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    x x x o o o o
    - - - 0 1 2 3

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_zz_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
            - 972 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 5)]")
            + 137 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 6)]")
        ) / (180 * sym.Symbol(hz) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            - 255 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            + 470 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            - 285 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            + 93 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
            - 13 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 5)]")
        ) / (180 * sym.Symbol(hz) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            + 228 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 200 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
            - 12 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 3)]")
            + 2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 4)]")
        ) / (180 * sym.Symbol(hz) ** 2)

    return expr


def build_6th_stencil_zz_3d_positive_boundary(symbol, boundary_away=0, idx_str="pp"):
    """

    Boundary Away parameter is how far away the boundary is:

    o o o o x x x
    3 2 1 0 - - -

    x's are the "ghost" points beyond the boundary.

    NOTE: idx_str must *not* have brackets!
    """

    if boundary_away < 0:
        raise Exception("Invalid boundary_away parameter, cannot be negative")
    if boundary_away > 2:
        return build_6th_stencil_zz_3d(symbol, idx_str=idx_str)

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    if boundary_away == 0:
        # boundary away means that our current x is right on the boundary
        expr = (
            +137 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 6)]")
            - 972 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 5)]")
            + 2970 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 5080 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 5265 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 3132 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            + 812 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
        ) / (180 * sym.Symbol(hz) ** 2)
    elif boundary_away == 1:
        # one away means we can only use one point on our "left"
        expr = (
            -13 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 5)]")
            + 93 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 285 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 470 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            - 255 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 147 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 137 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
        ) / (180 * sym.Symbol(hz) ** 2)
    elif boundary_away == 2:
        # two away means we can still use two points on our "left"
        expr = (
            +2 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 4)]")
            - 12 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 3)]")
            + 15 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 2)]")
            + 200 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} - 1)]")
            - 420 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k})]")
            + 228 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 1)]")
            - 13 * sym.Symbol(symbol + f"[IDX({i}, {j}, {k} + 2)]")
        ) / (180 * sym.Symbol(hz) ** 2)

    return expr


def build_6th_stencil_xy_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 3
    expr = (-1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 3, {j} + 3, {k})]")
    )

    # i - 2
    expr += (9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 2, {j} + 3, {k})]")
    )

    # i - 1
    expr += (-45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 1, {j} + 3, {k})]")
    )

    # i + 1
    expr += (45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 1, {j} + 3, {k})]")
    )

    # i + 2
    expr += (-9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 2, {j} + 3, {k})]")
    )

    # i + 3
    expr += (1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} - 3, {k})]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} - 2, {k})]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} - 1, {k})]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} + 1, {k})]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} + 2, {k})]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 3, {j} + 3, {k})]")
    )

    return expr / (60**2 * sym.Symbol(hx) * sym.Symbol(hy))


def build_6th_stencil_xz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 3
    expr = (-1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 3, {j}, {k} + 3)]")
    )

    # i - 2
    expr += (9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 2, {j}, {k} + 3)]")
    )

    # i - 1
    expr += (-45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} - 1, {j}, {k} + 3)]")
    )

    # i + 1
    expr += (45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 1, {j}, {k} + 3)]")
    )

    # i + 2
    expr += (-9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 2, {j}, {k} + 3)]")
    )

    # i + 3
    expr += (1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i} + 3, {j}, {k} + 3)]")
    )

    return expr / (60**2 * sym.Symbol(hx) * sym.Symbol(hz))


def build_6th_stencil_yz_3d(symbol, idx_str="pp"):
    """

    NOTE: idx_str must *not* have brackets!
    """

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # i - 3
    expr = (-1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 3, {k} + 3)]")
    )

    # i - 2
    expr += (9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 2, {k} + 3)]")
    )

    # i - 1
    expr += (-45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} - 1, {k} + 3)]")
    )

    # i + 1
    expr += (45) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 1, {k} + 3)]")
    )

    # i + 2
    expr += (-9) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 2, {k} + 3)]")
    )

    # i + 3
    expr += (1) * (
        -1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} - 3)]")
        + 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} - 2)]")
        - 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} - 1)]")
        + 45 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} + 1)]")
        - 9 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} + 2)]")
        + 1 * sym.Symbol(symbol + f"[IDX({i}, {j} + 3, {k} + 3)]")
    )

    return expr / (60**2 * sym.Symbol(hy) * sym.Symbol(hz))


def convert_num_to_str_add(num):
    if num == 0:
        return ""
    elif num < 0:
        return f" - {-1 * num}"
    else:
        return f" + {num}"


def generate_boundary_stencil_expr_mixed_deriv_6th(
    symbol,
    dim1,
    dim2,
    dim1_edge="+",
    dim2_edge="+",
    dim1_boundary_away=0,
    dim2_boundary_away=0,
    idx_str="pp",
):
    # in order, boundary away 0, 1, 2, and then doesn't worry about boundary
    stencil_points_away_pos = [
        [10, -72, 225, -400, 450, -360, 147],
        [-2, 15, -50, 100, -150, 77, 10],
        [1, -8, 30, -80, 35, 24, -2],
        [-1, 9, -45, 0, 45, -9, 1],
    ]

    stencil_points_away_neg = [
        [-147, 360, -450, 400, -225, 72, -10],
        [-10, -77, 150, -100, 50, -15, 2],
        [2, -24, -35, 80, -30, 8, -1],
        [-1, 9, -45, 0, 45, -9, 1],
    ]

    hxhyhz = [sym.Symbol(hx), sym.Symbol(hy), sym.Symbol(hz)]

    if dim1_boundary_away < 0 or dim2_boundary_away < 0:
        raise Exception("Invalid dim1 or dim2 boundary away input, cannot be negative")

    if dim1 == dim2:
        raise Exception(
            "This function should never be called if the dimensions are equal!"
        )

    if isinstance(symbol, sym.Symbol):
        symbol = symbol.name

    if symbol.endswith("[" + idx_str + "]"):
        symbol = symbol[: -(2 + len(idx_str))]

    # cap the boundary values, because once we're away from the boundary we're fine
    if dim1_boundary_away > 3:
        dim1_boundary_away = 3
    if dim2_boundary_away > 3:
        dim2_boundary_away = 3

    # swap order so it's easier
    if dim1 > dim2:
        temp = dim1
        dim1 = dim2
        dim2 = temp

    # calculate the starting position for our arrays based on the direction
    start_pos_dim1 = -3
    stencil_dim1 = stencil_points_away_pos[3]
    if dim1_edge == "+":
        # positive edge, so we offset positive by
        start_pos_dim1 = -6 + dim1_boundary_away
        stencil_dim1 = stencil_points_away_pos[dim1_boundary_away]
    elif dim1_edge == "-":
        start_pos_dim1 = -1 * dim1_boundary_away
        stencil_dim1 = stencil_points_away_neg[dim1_boundary_away]

    start_pos_dim2 = -3
    stencil_dim2 = stencil_points_away_pos[3]
    if dim2_edge == "+":
        # positive edge, so we offset positive by
        start_pos_dim2 = -6 + dim1_boundary_away
        stencil_dim2 = stencil_points_away_pos[dim1_boundary_away]
    elif dim2_edge == "-":
        start_pos_dim2 = -1 * dim1_boundary_away
        stencil_dim2 = stencil_points_away_neg[dim1_boundary_away]

    # we need to now iterate 7 in the first dimension, and

    expr = 0 * sym.Symbol("x")

    curr_dim1 = start_pos_dim1
    for ii in range(7):
        # ok so now we have the stencil, so we start building expr
        temp_expr = 0 * sym.Symbol("x")
        curr_dim2 = start_pos_dim2

        for jj in range(7):
            indexing = "[IDX("

            if dim1 == 0:
                indexing += f"{i}{convert_num_to_str_add(curr_dim1)}, "
            else:
                indexing += f"{i}, "

            if dim1 == 1:
                indexing += f"{j}{convert_num_to_str_add(curr_dim1)}, "
            elif dim2 == 1:
                indexing += f"{j}{convert_num_to_str_add(curr_dim2)}, "
            else:
                indexing += f"{j}, "

            if dim2 == 2:
                indexing += f"{k}{convert_num_to_str_add(curr_dim2)}"
            else:
                indexing += f"{k}"

            indexing += ")]"
            temp_expr += stencil_dim2[jj] * sym.Symbol(symbol + indexing)

            curr_dim2 += 1

        expr += stencil_dim1[ii] * temp_expr

        curr_dim1 += 1

    return expr / (60 * 60 * hxhyhz[dim1] * hxhyhz[dim2])


# for ii in range(3):
#     for jj in range(3):
#         if ii == jj:
#             continue

#         temp = generate_boundary_stencil_expr_mixed_deriv_6th(sym.Symbol("alpha"), ii, jj, "+", "-", 0, 0)

#         print(temp)
