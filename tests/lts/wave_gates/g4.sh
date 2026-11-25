#!/bin/bash
# g4.sh <label> [KEY=VAL ...]: uninterrupted run C vs checkpoint run A + restart B (LTS, remesh on)
cd $(dirname $0); L=$1; shift; G=gates/g4_$L
B=(remesh_test_freq=8 checkpt_freq=24 'time_stepper="lts"' "$@")
./run.sh $PWD/$G/C 1 1 "${B[@]}" rk_time_end=${TC:-4.0}; cat $G/C/summary.txt
./run.sh $PWD/$G/A 1 1 "${B[@]}" rk_time_end=${TA:-2.0}; cat $G/A/summary.txt
ls $G/A/cp
./run.sh $PWD/$G/B 1 1 "${B[@]}" rk_time_end=${TC:-4.0} restore_solver=true PRE=$PWD/$G/A; cat $G/B/summary.txt
./cmpev.py $G/C $G/B
grep -E "Final step|Final time|restor|Restor|carried|cold" $G/C/run.log $G/B/run.log | cut -c1-200
