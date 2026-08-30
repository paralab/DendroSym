/**
 * @file profile_params.cpp
 * @brief Profiling timer definitions for the smokemax solver.
 */

#include "profile_params.h"
#include <cstdio>

namespace smokemax {
namespace timer {

profiler_t total_runtime;

profiler_t t_f2o;
profiler_t t_cons;
profiler_t t_bal;
profiler_t t_mesh;

profiler_t t_rkSolve;

profiler_t t_ghostEx_sync;

profiler_t t_unzip_sync;

profiler_t t_unzip_async;

profiler_t t_deriv;
profiler_t t_rhs;
profiler_t t_ko;

profiler_t t_bdyc;

profiler_t t_zip;
profiler_t t_rkStep;

profiler_t t_isReMesh;
profiler_t t_remesh;
profiler_t t_gridTransfer;
profiler_t t_ioVtu;
profiler_t t_ioCheckPoint;

void print_profile(MPI_Comm comm) {
    int rank = 0, npes = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    struct row_t { const char *name; const profiler_t *t; };
    // only timers the generated code actually starts/stops (dendrolib's profiler_t
    // never increments num_calls, so no call counts)
    const row_t rows[] = {
        {"rhs (algebraic body)", &t_rhs},
        {"deriv (stencils + workspace)", &t_deriv},
        {"ko filter", &t_ko},
        {"bdyc (sommerfeld)", &t_bdyc},
        {"unzip_async", &t_unzip_async},
        {"zip", &t_zip},
    };
    if (rank == 0) {
        std::printf("\n[profile] %s timers, seconds (min / mean / max over %d rank%s)\n",
                    "SMOKEMAX", npes, npes == 1 ? "" : "s");
        std::printf("[profile] %-30s %12s %12s %12s\n", "timer", "min", "mean", "max");
    }
    for (const auto &r : rows) {
        const double loc = (double)r.t->seconds;
        double mn = 0, mx = 0, sum = 0;
        MPI_Reduce(&loc, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&loc, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&loc, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        if (rank == 0)
            std::printf("[profile] %-30s %12.4f %12.4f %12.4f\n", r.name, mn, sum / npes, mx);
    }
    if (rank == 0) std::fflush(stdout);
}

}  // namespace timer
}  // namespace smokemax
