#!/bin/bash
# g3.sh <label> <remesh_freq> [KEY=VAL..]: LTS vs global, same par; element count at every common time + final mesh by position
cd $(dirname $0); L=$1; RF=$2; shift 2; G=gates/g3_$L; B=(rk_time_end=4.0 remesh_test_freq=$RF "$@")
./run.sh $PWD/$G/glob 1 1 "${B[@]}" 'time_stepper="global"'; echo glob $(tr '\n' ' ' < $G/glob/summary.txt)
./run.sh $PWD/$G/lts 1 1 "${B[@]}" 'time_stepper="lts"'; echo lts $(tr '\n' ' ' < $G/lts/summary.txt)
python3 - $G <<'PY'
import re, sys
G = sys.argv[1]
def rd(f):
    out = {}
    for m in re.finditer(r"\[WAVE\] step=\d+ t=([0-9.e+-]+) dt=\S+ elements=(\d+)", open(f, errors="replace").read()):
        out[round(float(m.group(1)), 9)] = int(m.group(2))
    return out
g, l = rd(G + "/glob/run.log"), rd(G + "/lts/run.log")
common = sorted(set(g) & set(l))
bad = [(t, g[t], l[t]) for t in common if g[t] != l[t]]
chg = sum(1 for a, b in zip(common, common[1:]) if g[b] != g[a])
print(f"times: global {len(g)}, lts {len(l)}, common {len(common)}; mesh changes seen in common times {chg}; element-count mismatches {len(bad)} {bad[:5]}")
print("global element counts at common times:", [g[t] for t in common])
print("lts    element counts at common times:", [l[t] for t in common])
PY
echo -n "final mesh+field: "; ./cmpev.py $G/glob $G/lts
