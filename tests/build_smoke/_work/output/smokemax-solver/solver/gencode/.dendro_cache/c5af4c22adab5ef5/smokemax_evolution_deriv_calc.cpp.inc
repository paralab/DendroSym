// planned per-variable derivative sets -- the engine picks each
// call's shape (terminal outputs use the active-region kernels)
using __DD = dendroderivs::DendroDerivatives;
{
    const __DD::DerivSet __ds0[] = {   // x, y, z, xx, yy, zz
        { d.At00_x, d.At00_y, d.At00_z, d.At00_xx, d.At00_yy, d.At00_zz },
        { d.At01_x, d.At01_y, d.At01_z, d.At01_xx, d.At01_yy, d.At01_zz },
        { d.At02_x, d.At02_y, d.At02_z, d.At02_xx, d.At02_yy, d.At02_zz },
        { d.At11_x, d.At11_y, d.At11_z, d.At11_xx, d.At11_yy, d.At11_zz },
        { d.At12_x, d.At12_y, d.At12_z, d.At12_xx, d.At12_yy, d.At12_zz },
        { d.At22_x, d.At22_y, d.At22_z, d.At22_xx, d.At22_yy, d.At22_zz },
        { d.B0_x, d.B0_y, d.B0_z, d.B0_xx, d.B0_yy, d.B0_zz },
        { d.B1_x, d.B1_y, d.B1_z, d.B1_xx, d.B1_yy, d.B1_zz },
        { d.B2_x, d.B2_y, d.B2_z, d.B2_xx, d.B2_yy, d.B2_zz },
        { d.Gt0_x, d.Gt0_y, d.Gt0_z, d.Gt0_xx, d.Gt0_yy, d.Gt0_zz },
        { d.Gt1_x, d.Gt1_y, d.Gt1_z, d.Gt1_xx, d.Gt1_yy, d.Gt1_zz },
        { d.Gt2_x, d.Gt2_y, d.Gt2_z, d.Gt2_xx, d.Gt2_yy, d.Gt2_zz },
        { d.alpha_x, d.alpha_y, d.alpha_z, d.alpha_xx, d.alpha_yy, d.alpha_zz },
        { d.beta0_x, d.beta0_y, d.beta0_z, d.beta0_xx, d.beta0_yy, d.beta0_zz },
        { d.beta1_x, d.beta1_y, d.beta1_z, d.beta1_xx, d.beta1_yy, d.beta1_zz },
        { d.beta2_x, d.beta2_y, d.beta2_z, d.beta2_xx, d.beta2_yy, d.beta2_zz },
        { d.chi_x, d.chi_y, d.chi_z, d.chi_xx, d.chi_yy, d.chi_zz },
        { d.gt00_x, d.gt00_y, d.gt00_z, d.gt00_xx, d.gt00_yy, d.gt00_zz },
        { d.gt01_x, d.gt01_y, d.gt01_z, d.gt01_xx, d.gt01_yy, d.gt01_zz },
        { d.gt02_x, d.gt02_y, d.gt02_z, d.gt02_xx, d.gt02_yy, d.gt02_zz },
        { d.gt11_x, d.gt11_y, d.gt11_z, d.gt11_xx, d.gt11_yy, d.gt11_zz },
        { d.gt12_x, d.gt12_y, d.gt12_z, d.gt12_xx, d.gt12_yy, d.gt12_zz },
        { d.gt22_x, d.gt22_y, d.gt22_z, d.gt22_xx, d.gt22_yy, d.gt22_zz },
        { d.trK_x, d.trK_y, d.trK_z, d.trK_xx, d.trK_yy, d.trK_zz },
    };
    const double *const __du0[] = { in.At00, in.At01, in.At02, in.At11, in.At12, in.At22, in.B0, in.B1, in.B2, in.Gt0, in.Gt1, in.Gt2, in.alpha, in.beta0, in.beta1, in.beta2, in.chi, in.gt00, in.gt01, in.gt02, in.gt11, in.gt12, in.gt22, in.trK };
    SOLVER_DERIVS.grad_set_batch(__ds0, __du0, 24, __DD::DM_FIRST | __DD::DM_SECOND,
                                     hx, hy, hz, sz, bflag);
}
