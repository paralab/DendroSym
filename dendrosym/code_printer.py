import sympy as sym
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import precedence

import dendrosym
from dendrosym.derivs import variable_strs, xx, yy, zz, idx_str


class DendroCPrinter(C99CodePrinter):
    """
    The Dendro Custom C-Code printer for SymPy

    This inherits the C99CodePrinter object from SymPy so it
    maintains all of the base functionality from SymPy. The additions
    added are custom printers for the "function" variables and
    for printing Derivatives in the way our program is ready
    to accept them.

    """

    def __init__(self, additional_user_funcs={}):
        global variable_strs
        super().__init__()

        # add additional user functions
        self.known_functions = dict(self.known_functions, **additional_user_funcs)

        # add our variables to our list so that we don't have to do any
        # replacement stuff
        # NOTE: arg1, arg2, and arg3 are the x y and z dimensions
        self.known_functions = dict(
            self.known_functions,
            **{
                x: [
                    (
                        lambda arg1, arg2, arg3: True,
                        lambda arg1, arg2, arg3: x + idx_str,
                    ),
                ]
                for x in variable_strs
            },
        )

    def _print_im(self, expr):
        # all GR quantities are real; sympy emits im() only when it can't prove
        # this (e.g. psi**p_expo with unconstrained p_expo)
        return "0.0"

    def _print_re(self, expr):
        # dual to _print_im
        return self._print(expr.args[0])

    def _print_Pow(self, expr):
        """Special printer for remaining Pow expressions to force multiplication

        It is probably wiser to take any Pow expressions and expand them
        first for potential mathematical optimizations, but this will
        ensure that we're not calling the Pow function and the compiler
        can have the option to adjust them.

        Parameters
        ----------
        expr
            The expression in SymPy format ready to be printed

        Returns
        -------
        expr
            The expression as C Code ready for additional manipulation
        """
        PREC = precedence(expr)
        if expr.exp in range(2, 7):
            # wrap in parens so it's safe as a Mul denominator
            inner = "*".join([self.parenthesize(expr.base, PREC)] * int(expr.exp))
            return "(" + inner + ")"
        elif expr.exp in range(-6, 0):
            return (
                "(1.0/("
                + ("*".join([self.parenthesize(expr.base, PREC)] * int(-expr.exp)))
                + "))"
            )
        else:
            return super()._print_Pow(expr)

    def _print_Derivative(self, expr):
        """Special printer for the Derivative class

        Do note that this will only be called if we have pure SymPy
        derivatives in the function. Otherwise, we should be using
        the rest of the ccode stuff (which basically just calls
        this class anyway).

        The formatting is based specifically on `grad(dir, expr)` because
        the `expr` variable may not always be just a variable name. Regex
        in the `codegen.py` file will correct these to our proper derivative
        type.

        Parameters
        ----------
        expr
            The expression in SymPy format ready to be printed

        Returns
        -------
        expr
            The expression as C Code ready for additional manipulation

        Raises
        ------
        NotImplementedError
            If the derivative is 3rd order or has incorrect dimensionality
            the printer will not yet work. We can add additional derivative
            support, but Dendro itself does not yet have any code to compute
            derivatives beyond the second order.
        """

        differand, *(wrt_counts) = expr.args

        d_order = (xx, yy, zz)

        if len(wrt_counts) > 2 or wrt_counts[0][1] > 2:
            # this is third order or higher
            raise NotImplementedError(
                "Currently only first and second order derivs are supported!"
            )
        elif wrt_counts[0][1] == 2:
            # second order same direction
            ((wrt, count),) = wrt_counts

            # check the order of the wrt and that's our idx
            for ii, val in enumerate(d_order):
                if wrt == val:
                    idx_use = ii
                    break

            return "grad2(%d, %d, %s)" % (idx_use, idx_use, self._print(differand))

        elif len(wrt_counts) == 2:
            (
                (wrt1, count1),
                (wrt2, count2),
            ) = wrt_counts

            # check the order of the wrt and that's our idx
            for ii, val in enumerate(d_order):
                if wrt1 == val:
                    idx1 = ii

                if wrt2 == val:
                    idx2 = ii

            # swap so the order is "increasing"
            if idx1 > idx2:
                idxtmp = idx1
                idx1 = idx2
                idx2 = idx1

            return "grad2(%d, %d, %s)" % (idx1, idx2, self._print(differand))

        else:
            # single order derivative
            ((wrt, count),) = wrt_counts

            # check the order of the wrt and that's our idx
            for ii, val in enumerate(d_order):
                if wrt == val:
                    idx_use = ii

            return "grad(%d, %s)" % (idx_use, self._print(differand))
