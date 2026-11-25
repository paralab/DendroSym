#!/bin/bash
cd $(dirname $0); G=gates/g5; B=(remesh_test_freq=8 rk_time_end=3.0 'time_stepper="lts"' iface=15 g_exchange=true)
for t in 1 2 4; do ./run.sh $PWD/$G/t$t 1 $t "${B[@]}"; echo t=$t $(tr '\n' ' ' < $G/t$t/summary.txt); done
for t in 2 4; do echo -n "1 vs $t: "; ./cmpev.py $G/t1 $G/t$t; done
