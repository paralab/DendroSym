#!/bin/bash
# Byte-identical regeneration oracle for the cascade fold-in.
#
# Port of vikr/scripts/regen_all_generated.sh: regenerates every IR-generated kernel
# checked in under vikr/harness + vikr/harness_emda from the MOVED package
# (dendrosym.cascade) into a scratch dir, then diffs against the checked-in files.
# Silence after "=== DIFF" means byte-identical -- the only proof nothing drifted.
#
# Flag pins are copied from the vikr script and are load-bearing:
#   --inline-threshold 0   scalar kernels (CLI default is 2)
#   --no-fma-tree          AVX kernels frozen before fma-tree became the default
#   PYTHONHASHSEED=0       + sympy 1.13.3 (requirements-cascade.txt)
#
# Usage: scripts/regen_vikr_kernels.sh [OUT_DIR]     (env: VIKR, PYTHON)
set -e
VIKR=${VIKR:-$HOME/research/vikr}
PYTHON=${PYTHON:-python3}
OUT=$(realpath -m "${1:-${TMPDIR:-/tmp}/cascade_regen}")
export PYTHONHASHSEED=0
BSSN="$PYTHON -m dendrosym.cascade.systems.bssn.emit"
LOOPED="$PYTHON -m dendrosym.cascade.systems.bssn.looped"
EMDA="$PYTHON -m dendrosym.cascade.systems.emda.cascade"
G=$OUT/bssn; E=$OUT/emda
rm -rf "$G" "$E"; mkdir -p "$G" "$E"
# run from OUT: importing emda-gr's emdabssn_eqns_configs writes
# EMDA_GENCODE_constraints.cpp into the cwd (external module-level side effect).
cd "$OUT"
$PYTHON -c "import sympy; assert sympy.__version__=='1.13.3', sympy.__version__"

# SCALAR kernels: --inline-threshold 0 EXPLICITLY (see vikr script for the measurement).
$BSSN --inline-threshold 0 -o $G/bssneqs_cascade_ir.cpp
for L in 1 5 6 7 8 9 11 13; do
  $BSSN --inline-threshold 0 --L $L -o $G/bssneqs_cascade_ir_L$L.cpp
done
# AVX base variants: --no-fma-tree pins the committed (non-FMA) form.
for L in 1 5 6 7 8 9; do
  $BSSN --simd avx2 --L $L --no-fma-tree -o $G/bssneqs_cascade_ir_avx2_L$L.cpp
done
$BSSN --simd avx2 --L 9 --split-mode dumb  --no-fma-tree -o $G/bssneqs_cascade_ir_avx2_L9dumb.cpp
$BSSN --simd avx2 --L 9 --split-mode smart --no-fma-tree -o $G/bssneqs_cascade_ir_avx2_L9smart.cpp
# Structured looped emitter (scalar control, AVX2 tool, AVX-512 A/B point).
$LOOPED --simd scalar -o $G/bssneqs_cascade_ir_scalar_looped.cpp
$LOOPED --simd avx2 -o $G/bssneqs_cascade_ir_avx2_looped.cpp
$LOOPED --simd avx2 --ssl --cahd -o $G/bssneqs_cascade_ir_avx2_looped_ssl_cahd.cpp
$LOOPED --simd avx512 -o $G/bssneqs_cascade_ir_avx512_looped.cpp
# Explicit-FMA companions.
for L in 7 9; do
  $BSSN --simd avx2 --L $L --fma-tree -o $G/bssneqs_cascade_ir_avx2_L${L}_fma.cpp
done
$BSSN --simd avx2 --L 7 --global-cse --fma-tree -o $G/bssneqs_cascade_ir_avx2_gcse.cpp
$BSSN --inline-threshold 0 --ssl --cahd -o $G/bssneqs_cascade_ir_ssl_cahd.cpp
$BSSN --simd avx2 --ssl --cahd --no-fma-tree -o $G/bssneqs_cascade_ir_avx_ssl_cahd.cpp
$BSSN --simd avx2 --L 7 --fused --no-fma-tree -o $G/bssneqs_cascade_ir_avx_fused.cpp
# DEPLOYED body (cascade-ir-avx512-ssl-cahd-fused reuses this width-agnostic file).
$BSSN --simd avx2 --L 7 --ssl --cahd --fused --no-fma-tree -o $G/bssneqs_cascade_ir_avx_ssl_cahd_fused.cpp
$BSSN --simd avx2 --inline-threshold 999999999 --no-fma-tree -o $G/bssneqs_cascade_ir_avx_nocse.cpp
$BSSN --inline-threshold 0 --ssl --cahd --L 7 -o $G/bssneqs_cascade_ir_ssl_cahd_L7.cpp
$BSSN --inline-threshold 999999999 -o $G/bssneqs_cascade_ir_nocse.cpp

$EMDA --output $E/bssneqs_cascade_emda_unified_v2.cpp
$EMDA --simd avx --output $E/bssneqs_cascade_emda_unified_v2_avx.cpp
$EMDA --simd avx --fused --output $E/bssneqs_cascade_emda_unified_v2_avx_fused.cpp
for L in 1 6 7 9 10; do
  $EMDA --simd avx --L $L --output $E/bssneqs_cascade_emda_unified_v2_avx_L$L.cpp
done
for L in 1 6 7 9 10; do
  $EMDA --L $L --output $E/bssneqs_cascade_emda_unified_v2_L$L.cpp
done
# --hoist-exp variants (vikr 664b1e9; not yet in vikr's own regen script).
$EMDA --simd avx --fused --hoist-exp --output $E/bssneqs_cascade_emda_unified_v2_avx_fused_hoist.cpp
$EMDA --simd avx --L 7 --hoist-exp --output $E/bssneqs_cascade_emda_unified_v2_avx_L7_hoist.cpp
$EMDA --simd avx --L 8 --hoist-exp --output $E/bssneqs_cascade_emda_unified_v2_avx_L8_hoist.cpp

# Known-stale checked-in files: vikr's CURRENT emitter (664b1e9) does not reproduce
# them either (verified 2026-08-26: vikr's cascade_emit.py == ours byte-for-byte).
#   bssneqs_cascade_ir_nocse.cpp  -- header comment from the retired python
#                                    "ABLATION" workaround; body identical
KNOWN_STALE="bssneqs_cascade_ir_nocse.cpp"

echo "=== DIFF vs vikr checked-in (silence = byte-identical) ==="
fail=0
for f in $G/*.cpp; do
  b=$(basename $f)
  if cmp -s "$f" "$VIKR/harness/src/rhsfuncs/generated/$b"; then :;
  elif [[ " $KNOWN_STALE " == *" $b "* ]]; then echo "known-stale (vikr HEAD differs too): $b";
  else echo "CHANGED: $b"; fail=1; fi
done
for f in $E/*.cpp; do
  cmp -s "$f" "$VIKR/harness_emda/src/gencode/$(basename $f)" || { echo "CHANGED: $(basename $f)"; fail=1; }
done
echo "=== REGEN COMPLETE ($(ls $G/*.cpp $E/*.cpp | wc -l) kernels) fail=$fail ==="
exit $fail
