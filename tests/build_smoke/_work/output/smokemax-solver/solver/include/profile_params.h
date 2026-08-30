/**
 * @file profile_params.h
 * @brief Profiling timer declarations for the smokemax solver.
 */

#ifndef PROFILE_PARAMS_H
#define PROFILE_PARAMS_H

#include "profiler.h"
#include "mpi.h"

namespace smokemax {
namespace timer {

extern profiler_t total_runtime;

extern profiler_t t_f2o;
extern profiler_t t_cons;
extern profiler_t t_bal;
extern profiler_t t_mesh;

extern profiler_t t_rkSolve;
extern profiler_t t_ghostEx_sync;

extern profiler_t t_unzip_sync;
extern profiler_t t_unzip_async;

extern profiler_t t_deriv;
extern profiler_t t_rhs;
extern profiler_t t_ko;

extern profiler_t t_bdyc;

extern profiler_t t_zip;
extern profiler_t t_rkStep;

extern profiler_t t_isReMesh;
extern profiler_t t_remesh;
extern profiler_t t_gridTransfer;
extern profiler_t t_ioVtu;
extern profiler_t t_ioCheckPoint;

// end-of-run report (min / mean / max over ranks); enabled by the generator's
// profiling option (config.enable_profiling / --profile).
void print_profile(MPI_Comm comm);

}  // namespace timer
}  // namespace smokemax

#endif  // PROFILE_PARAMS_H
