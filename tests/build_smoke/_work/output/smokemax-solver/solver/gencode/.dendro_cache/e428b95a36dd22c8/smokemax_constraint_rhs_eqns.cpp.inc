// Dendro: C++ Equation Code Generation {{{{ 
// Dendro: TEMPORARY VARIABLES
// Dendro: END TEMPORARY VARIABLES

// Dendro: MAIN VARIABLES
//--
out.ham[pp] = d.chi_xx[pp] + d.chi_yy[pp] + d.chi_zz[pp];

//--
out.mom0[pp] = d.chi_x[pp];

//--
out.mom1[pp] = d.chi_y[pp];

//--
out.mom2[pp] = d.chi_z[pp];

//--
out.psi4_real[pp] = d.alpha_xx[pp] + d.alpha_yy[pp] + d.alpha_zz[pp];

//--
out.psi4_imag[pp] = d.trK_xx[pp] + d.trK_yy[pp] + d.trK_zz[pp];
// Dendro: END MAIN VARIABLES

// Dendro: }}}} End Code Generation 
