#!/bin/bash
# g2.sh <label> [KEY=VAL..]: iface 15 + g_exchange, ranks 1/2/4, compared by node position
cd $(dirname $0); L=$1; shift; G=gates/g2_$L; B=(rk_time_end=3.0 'time_stepper="lts"' iface=15 g_exchange=true "$@")
for n in 1 2 4; do ./run.sh $PWD/$G/np$n $n 1 "${B[@]}"; echo np=$n $(tr '\n' ' ' < $G/np$n/summary.txt); done
for n in 2 4; do echo -n "np1 vs np$n: "; ./cmpev.py $G/np1 $G/np$n; done
