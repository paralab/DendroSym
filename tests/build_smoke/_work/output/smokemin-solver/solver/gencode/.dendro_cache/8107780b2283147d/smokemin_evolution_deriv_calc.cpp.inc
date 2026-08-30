// planned per-variable derivative sets -- the engine picks each
// call's shape (terminal outputs use the active-region kernels)
using __DD = dendroderivs::DendroDerivatives;
{
    const __DD::DerivSet __ds0[] = {   // x, y, z, xx, yy, zz
        { d.u_x, d.u_y, d.u_z, d.u_xx, d.u_yy, d.u_zz },
    };
    const double *const __du0[] = { in.u };
    SOLVER_DERIVS.grad_set(__ds0[0], __du0[0], __DD::DM_FIRST | __DD::DM_SECOND,
                               hx, hy, hz, sz, bflag);
}
{
    const __DD::DerivSet __ds1[] = {   // x, y, z
        { d.v_x, d.v_y, d.v_z },
    };
    const double *const __du1[] = { in.v };
    SOLVER_DERIVS.grad_set(__ds1[0], __du1[0], __DD::DM_FIRST,
                               hx, hy, hz, sz, bflag);
}
