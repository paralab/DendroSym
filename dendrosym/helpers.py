from functools import lru_cache
import sympy as sym


# these are to help with tree traversal to make finding and optimizing derivatives much faster
@lru_cache(maxsize=1)
def _repl_dicts(var_names, idx_str):
    """Return forward and reverse replacement dictionaries."""
    x, y, z = sym.symbols("xtem ytem ztem")
    d_order = (x, y, z)

    symbols_find, symbols_replace = [], []

    for v in var_names:
        base = v[: -len(idx_str)] if v.endswith(idx_str) else v
        f = sym.Function(sym.Symbol(base))
        for s in (base, base + idx_str):
            symbols_find.append(sym.Symbol(s))
            symbols_replace.append(f(*d_order))
    return dict(zip(symbols_find, symbols_replace)), dict(
        zip(symbols_replace, symbols_find)
    )


class _DerTransformer:
    """Bottom-up transformer class that replaces grad / grad2 / agrad nodes."""

    def __init__(self, var_names, idx_str):
        self.fwd, self.rev = _repl_dicts(tuple(var_names), idx_str)
        self.x, self.y, self.z = sym.symbols("xtem ytem ztem")
        self.d_order = (self.x, self.y, self.z)

    def __call__(self, expr):
        def bottom_up(e):
            # get all the way to the bottom of the tree
            if e.args:
                new_args = tuple(bottom_up(a) for a in e.args)
                if new_args != e.args:
                    e = e.func(*new_args)

            # process current node once it's children have been processed
            # Only check user-defined functions (grad, grad2, agrad).
            # Built-in functions like exp, sin, cos are also sym.Function
            # but don't have a .name attribute in the same way.
            if isinstance(e, sym.Function) and hasattr(e, 'name'):
                if e.name == "grad":
                    # simple symbol substitution
                    dim, arg = e.args
                    arg_f = arg.subs(self.fwd)

                    # then differentiate and do full evaluation of the chain rule
                    diff_expr = sym.diff(arg_f, self.d_order[dim]).doit()

                    # substitute original symbols back in right away
                    return diff_expr.subs(self.rev)

                # do the same for grad2
                if e.name == "grad2":
                    d1, d2, arg = e.args
                    arg_f = arg.subs(self.fwd)
                    diff_expr = sym.diff(
                        arg_f, self.d_order[d1], self.d_order[d2]
                    ).doit()
                    return diff_expr.subs(self.rev)

                if e.name == "agrad":
                    print("WARNING: agrad in expression – skipping simplification")

            return e

        return bottom_up(expr)
