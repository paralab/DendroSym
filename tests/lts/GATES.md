# LTS contract gates on the generated wave solver

Wave equation (`wave_eqn_config.py`, the Ch. 4 example), generated with `enable_lts` on, built against
`dendrolib_dfvk_ws_tidy` (read-only, `DENDRO_dendrolib_DIR`). Single node, 16 cores, no cluster.
The scripts are in `tests/lts/wave_gates/` and carry this machine's paths.

## Setup
- Generate: `python gen.py <out> on|off` (`cfg.enable_lts = True` for `on`). Control = `off`.
- Build: Ninja, Release (`-O3 -g -DNDEBUG`), `DENDRO_HYBRID_OMP=ON`, `DENDRO_RK_FUSE`, `DENDRO_UNZIP_SPEEDUP/PLAN`,
  `DENDRO_ZIP_PLAN`, `DENDRO_UNZIP_BATCH`, `DENDRO_GHOST_PACK_FLAT`, fastpart from `fastpart/build_balance/install`,
  spdlog/toml11/libxsmm from `dendrolib_dfvk_copy/build/_deps` (offline). Target `waveSolver`, 60 s with 10 jobs.
- Problem: domain [-16,16]^3, `mindepth 4`, `maxdepth 8`, `ele_order 6`, RK4, `cfl_factor 0.25`, `id_type 2`
  (Gaussian), wavelet AMR (876 elements at t=0, 2500+ later), fine step dt = 0.0208333. Pars come from
  `mkpar.py` (generated sample + overrides). The sample TOML itself does not load: `rk_time_begin = 0` is an
  integer where the reader wants a float (same with the flag off); `mkpar.py` writes `0.0`.
- Comparison: `DENDRO_EVAR_DUMP` + `DENDRO_EVAR_DUMP_POS` (new in the driver template), by node position
  (`cmpev.py`), duplicate nodes dropped.
- Every run holds `repart_gate/work/mpi.lock`.

## Results
| Gate | Command | Result | Tolerance | |
|---|---|---|---|---|
| 1. ratio_cap 1 = global | `g1.sh` (25 steps, frozen mesh) | max abs diff 8.6e-16, rel 4.4e-15 (max abs u 0.195, 179479 nodes) | 1e-14 | pass |
| 2. rank count, iface 15 + g_exchange, frozen mesh | `g2.sh frozen remesh_test_freq=1000000` (1, 2, 4 ranks, t=3) | bitwise equal (diff 0 on 179479 nodes) | bitwise | pass |
| 2. same, remeshing every 8 steps | `g2.sh remesh remesh_test_freq=8` | meshes identical; diff 4.5e-16 (np2), 4.6e-16 (np4), rel 2.7e-14 | bitwise | **fail, not LTS** |
| 3. same meshes, ratio_cap 1 | `g3.sh cap1 16 ratio_cap=1` | element count equal at all 192 common times, 11 remeshes; final nodes identical; field diff 3.4e-16 | equal | pass |
| 3. same meshes, ratio_cap 64 | `g3.sh cap64b 16 rk_time_end=3.99` | equal at all 16 common times (every LTS step), 11 remeshes; final nodes identical (518419); field diff 3.4e-7 (rel 2.7e-5) | equal | pass |
| 4. restart, iface 3, remesh every 8 | `g4.sh i3 iface=3` (checkpoint every 24 steps, restart at step 7, end t=4) | bitwise equal; every CSV row after the restart is in the uninterrupted run's CSV | bitwise | pass |
| 4. restart, iface 15 + g_exchange, remesh | `g4.sh i15 iface=15 g_exchange=true` | bitwise equal (565183 nodes), CSVs as above | bitwise | pass |
| 4. restart, iface 15 + g_exchange, frozen mesh | `g4.sh i15frozen iface=15 g_exchange=true remesh_test_freq=1000000` | bitwise equal; same restart with the `_enuts_g` files deleted: diff 7.2e-7 (rel 5.7e-5) | bitwise | pass |
| 5. threads 1, 2, 4 | `g5.sh` (iface 15 + g_exchange, remesh) | bitwise equal | bitwise | **vacuous** |

## What the table does not say
- Gate 2 with remeshing: the global stepper on the same binary shows the same size of rank-count difference
  (7.2e-16 at np2, 6.6e-16 at np4, `remesh_test_freq=8`, frozen mesh: bitwise). So the remesh path (grid transfer or the
  refinement test's partition dependence) is not rank-count independent at round-off on this solver; LTS does
  not add to it. Not investigated further.
- Gate 5: `lts_thread_safe` is not declared, so the stepper's block loops run on one thread and equality at
  1/2/4 threads says nothing about threading. Declaring it needs per-thread scratch for what `rhs_blk` writes:
  the derivative workspace (`WAVE_DERIV_WORKSPACE`, read inside the generated `rhs.cpp`), the shared derivative
  object `WAVE_DERIVS`, and the profiling timers in `rhs.cpp`. That changes the emitted kernel, so it is its own piece.
- Gate 3 compares element counts per time plus the final mesh by position, not every remesh's node set.
- Gate 4's "every output file" is the evolution vector plus the two CSVs; VTU was off.
- Flag-off control: the flag-off build and the flag-on build on the global stepper write identical
  `ANALYTICAL_DIFF` and `Constraints` CSVs (`remesh_test_freq=8`, t=3). The flag-off wave output differs from the
  pre-change output only in the stamp's template hash.
- LTS with ratio_cap 64 took 19 s against 47 s global on the t=4 run (coarse step 16 or 8 fine steps).

## Found by running it (fixed on the branch)
- `get_async_batch_sz` missing from the generated ctx: the CRTP base forwards to the leaf, so `evolve()` recursed
  into itself (a hang at -O2, a stack overflow at -O0). It compiles, so the syntax-only compile never saw it. CCZ4 had the same gap.
- `main` deleted its own mesh pointer after the LTS driver had remeshed (double free at exit).
- Both have a test that fails when the line is removed.
