#!/bin/bash
# run.sh <dir> <np> <threads> [BIN=path] KEY=VAL...   one gate run: par from mkpar.py, EV dump with positions, shared mpi.lock
D=$1; NP=$2; T=$3; shift 3
BIN=$HOME/research/gen_lts_trial/wave_lts_build/solver/waveSolver
KV=()
for a in "$@"; do case $a in BIN=*) BIN=${a#BIN=};; PRE=*) PRE=${a#PRE=};; *) KV+=("$a");; esac; done
rm -rf $D; mkdir -p $D/vtu $D/cp; cd $D
python3 $HOME/research/gen_lts_trial/wave_runs/mkpar.py par.toml "${KV[@]}" || exit 3
[ -n "${PRE:-}" ] && cp $PRE/cp/* cp/
export OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores DENDRO_EVAR_DUMP=$D/ev DENDRO_EVAR_DUMP_POS=1
t0=$(date +%s)
flock $HOME/research/repart_gate/work/mpi.lock timeout ${TMO:-900} mpirun -np $NP --oversubscribe --map-by ppr:$NP:node:pe=$T --bind-to core \
    -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x DENDRO_EVAR_DUMP -x DENDRO_EVAR_DUMP_POS $BIN par.toml > run.log 2>&1
echo "rc=$? secs=$(( $(date +%s) - t0 ))" > summary.txt
echo "ev=$(cat $(ls ev_rank*.bin 2>/dev/null | sort -V) 2>/dev/null | md5sum | cut -c1-12)" >> summary.txt
