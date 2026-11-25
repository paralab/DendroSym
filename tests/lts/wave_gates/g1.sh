#!/bin/bash
cd $(dirname $0); G=gates; N=(remesh_test_freq=1000000 rk_time_end=0.52)
./run.sh $PWD/$G/g1_global 1 1 "${N[@]}" 'time_stepper="global"'; cat $G/g1_global/summary.txt
./run.sh $PWD/$G/g1_lts1 1 1 "${N[@]}" 'time_stepper="lts"' ratio_cap=1; cat $G/g1_lts1/summary.txt
./cmpev.py $G/g1_global $G/g1_lts1; grep -c "WAVE\] step" $G/g1_global/run.log $G/g1_lts1/run.log
