/**
 * @file profile_params.h
 * @brief Profiling timer declarations for the smokemin solver.
 */

#ifndef PROFILE_PARAMS_H
#define PROFILE_PARAMS_H

#include "profiler.h"

namespace smokemin {
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


}  // namespace timer
}  // namespace smokemin

#endif  // PROFILE_PARAMS_H
