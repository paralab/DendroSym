// planned per-variable derivative sets -- the engine picks each
// call's shape (terminal outputs use the active-region kernels)
using __DD = dendroderivs::DendroDerivatives;
{
    const __DD::DerivSet __ds0[] = {   // x, y, z, xx, yy, zz
        { d.chi_x, d.chi_y, d.chi_z, d.chi_xx, d.chi_yy, d.chi_zz },
    };
    const double *const __du0[] = { in.chi };
    SOLVER_DERIVS.grad_set(__ds0[0], __du0[0], __DD::DM_FIRST | __DD::DM_SECOND,
                               hx, hy, hz, sz, bflag);
}
{
    const __DD::DerivSet __ds1[] = {   // x, y, z, xx, yy, zz
        { nullptr, nullptr, nullptr, d.alpha_xx, d.alpha_yy, d.alpha_zz },
        { nullptr, nullptr, nullptr, d.trK_xx, d.trK_yy, d.trK_zz },
    };
    const double *const __du1[] = { in.alpha, in.trK };
    SOLVER_DERIVS.grad_set_batch(__ds1, __du1, 2, __DD::DM_SECOND,
                                     hx, hy, hz, sz, bflag);
}
