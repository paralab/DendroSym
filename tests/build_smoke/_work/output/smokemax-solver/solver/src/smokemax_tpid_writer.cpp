// smokemax per-point ADM -> evolution-variable writer for the
// dendrolib TPID hook.
//
// Generated once, then yours: written only when absent, so edits survive a
// regenerate.
//
// ---------------------------------------------------------------------------
// CHECK THIS BEFORE TRUSTING A TPID RUN.
//
// TwoPunctures returns gtd, Atd, chi, trK and the lapse. The assignments below
// are filled in by exact name match against the evolution variables; anything
// that did not match stays zero. That is correct for a shift, a Gamma-driver
// and a Z4 constraint variable at t=0, and wrong wherever the formulation
// renames a quantity.
//
// Example: a Z4 formulation evolves Khat = K - 2 s Theta, not a bare trK, so
// the trace matches nothing and is initialised to zero. Assign it here.
// ---------------------------------------------------------------------------

#include "smokemax_tpid_writer.h"

#include "grDef.h"

namespace smokemax {

void writePunctureVars(double *vars, const double gtd[3][3],
                   const double Atd[3][3], double chi, double trK,
                   double alpha) {
    // everything not named below stays at its zero initial value
    vars[VAR::U_ALPHA] = 0.0;
    vars[VAR::U_CHI] = 0.0;
    vars[VAR::U_TRK] = 0.0;
    vars[VAR::U_BETA0] = 0.0;
    vars[VAR::U_BETA1] = 0.0;
    vars[VAR::U_BETA2] = 0.0;
    vars[VAR::U_GT0] = 0.0;
    vars[VAR::U_GT1] = 0.0;
    vars[VAR::U_GT2] = 0.0;
    vars[VAR::U_B0] = 0.0;
    vars[VAR::U_B1] = 0.0;
    vars[VAR::U_B2] = 0.0;
    vars[VAR::U_AT00] = 0.0;
    vars[VAR::U_AT01] = 0.0;
    vars[VAR::U_AT02] = 0.0;
    vars[VAR::U_AT11] = 0.0;
    vars[VAR::U_AT12] = 0.0;
    vars[VAR::U_AT22] = 0.0;
    vars[VAR::U_GT00] = 0.0;
    vars[VAR::U_GT01] = 0.0;
    vars[VAR::U_GT02] = 0.0;
    vars[VAR::U_GT11] = 0.0;
    vars[VAR::U_GT12] = 0.0;
    vars[VAR::U_GT22] = 0.0;

    vars[VAR::U_ALPHA] = alpha;
    vars[VAR::U_CHI] = chi;
    vars[VAR::U_TRK] = trK;
    vars[VAR::U_AT00] = Atd[0][0];
    vars[VAR::U_AT01] = Atd[0][1];
    vars[VAR::U_AT02] = Atd[0][2];
    vars[VAR::U_AT11] = Atd[1][1];
    vars[VAR::U_AT12] = Atd[1][2];
    vars[VAR::U_AT22] = Atd[2][2];
    vars[VAR::U_GT00] = gtd[0][0];
    vars[VAR::U_GT01] = gtd[0][1];
    vars[VAR::U_GT02] = gtd[0][2];
    vars[VAR::U_GT11] = gtd[1][1];
    vars[VAR::U_GT12] = gtd[1][2];
    vars[VAR::U_GT22] = gtd[2][2];
}

}  // namespace smokemax
