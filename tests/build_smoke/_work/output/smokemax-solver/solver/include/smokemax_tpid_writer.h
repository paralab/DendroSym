// smokemax per-point ADM -> evolution-variable writer for the
// dendrolib TPID hook. Registered via TPID::registerPunctureVarsWriter in
// smokemax_main.cpp.
//
// Generated once, then yours: written only when absent, so edits survive a
// regenerate.
#pragma once

namespace smokemax {
void writePunctureVars(double *vars,
                   const double gtd[3][3],
                   const double Atd[3][3],
                   double chi, double trK, double alpha);
}
