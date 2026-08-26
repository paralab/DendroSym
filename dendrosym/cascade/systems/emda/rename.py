"""rename_cascade_to_emda.py -- mechanically translate vikr's BSSN cascade
output to EMDA-BSSN naming conventions.

Reads /tmp/vikr_cascade_out/bssneqs_cascade.cpp (from
`python cascade_codegen.py --target cpu --output /tmp/vikr_cascade_out/`)
and emits a sibling file with all state-variable, RHS-output, and
derivative names rewritten to match EMDA gencode conventions:

  BSSN naming        EMDA naming
  -----------        -----------
  gt0..gt5           gt00, gt01, gt02, gt11, gt12, gt22
  At0..At5           At00, At01, At02, At11, At12, At22
  K                  trK
  B0..B2             gaugeB0..gaugeB2
  Gt0..Gt2           CAP_Gt0..CAP_Gt2
  a_rhs              alpha_rhs
  K_rhs              trK_rhs
  gt_rhs00..gt_rhs22 gt00_rhs..gt22_rhs
  At_rhs00..At_rhs22 At00_rhs..At22_rhs
  b_rhs0..b_rhs2     beta0_rhs..beta2_rhs
  B_rhs0..B_rhs2     gaugeB0_rhs..gaugeB2_rhs
  Gt_rhs0..Gt_rhs2   CAP_Gt0_rhs..CAP_Gt2_rhs
  lambda_f           lf

Derivative names get translated through the same field renames:
  grad_d_gt0     -> grad_d_gt00
  grad_d_K       -> grad_d_trK
  grad2_i_j_At0  -> grad2_i_j_At00
  agrad_d_B0     -> agrad_d_gaugeB0

NOTE: this is a one-shot translation. The output is missing EMDA-specific
matter source terms (perpT in At_rhs, rho/perpTTrace in trK_rhs,
stressCurrent in CAP_Gt_rhs). The harness should fall back to the
production gencode for the 12 matter-only RHS components.
"""

from __future__ import annotations

import argparse
import re

# Symmetric-tensor index map: 6-component flat -> 2-digit
_SYM6 = {0: "00", 1: "01", 2: "02", 3: "11", 4: "12", 5: "22"}

# (regex pattern, replacement). Patterns use lookbehind/lookahead to avoid
# matching inside a longer identifier (e.g. don't rewrite `K` inside
# `BSSN_KO_DISS`). Order matters: process longer patterns first.
RENAMES: list[tuple[str, str]] = []


def _add_field_renames():
    # gt0..5, At0..5  -> gt00..gt22, At00..At22, in derivative + plain forms
    for i in range(6):
        ij = _SYM6[i]
        # plain references and as suffix in grad_/grad2_/agrad_ identifiers.
        # (?<![A-Za-z0-9_])gt0(?![0-9])  matches "gt0" at identifier boundary
        # but not "gt00" or "ZZZZgt0"
        for kind in ("gt", "At"):
            old = f"{kind}{i}"
            new = f"{kind}{ij}"
            RENAMES.append((rf"(?<![A-Za-z0-9_]){old}(?![0-9])", new))

    # gt_rhs00..gt_rhs22 -> gt00_rhs..gt22_rhs (suffix swap, BSSN harness
    # uses gt_rhs00 but EMDA gencode uses gt00_rhs)
    for ij in ("00", "01", "02", "11", "12", "22"):
        RENAMES.append((rf"\bgt_rhs{ij}\b", f"gt{ij}_rhs"))
        RENAMES.append((rf"\bAt_rhs{ij}\b", f"At{ij}_rhs"))

    # K, K_rhs (alone), grad_d_K, grad2_i_j_K  -> trK
    RENAMES.append((r"(?<![A-Za-z0-9_])K_rhs(?![A-Za-z0-9_])", "trK_rhs"))
    RENAMES.append((r"(?<![A-Za-z0-9_])K(?![A-Za-z0-9_])", "trK"))

    # B0..B2 -> gaugeB0..gaugeB2 (and B_rhs0 -> gaugeB0_rhs)
    for c in range(3):
        RENAMES.append((rf"\bB_rhs{c}\b", f"gaugeB{c}_rhs"))
        RENAMES.append((rf"(?<![A-Za-z0-9_])B{c}(?![A-Za-z0-9_])",
                        f"gaugeB{c}"))

    # Gt0..Gt2 -> CAP_Gt0..CAP_Gt2 (and Gt_rhs0 -> CAP_Gt0_rhs)
    for c in range(3):
        RENAMES.append((rf"\bGt_rhs{c}\b", f"CAP_Gt{c}_rhs"))
        RENAMES.append((rf"(?<![A-Za-z0-9_])Gt{c}(?![A-Za-z0-9_])",
                        f"CAP_Gt{c}"))

    # b_rhs0..b_rhs2 -> beta0_rhs..beta2_rhs
    for c in range(3):
        RENAMES.append((rf"\bb_rhs{c}\b", f"beta{c}_rhs"))

    # a_rhs -> alpha_rhs
    RENAMES.append((r"\ba_rhs\b", "alpha_rhs"))

    # lambda_f -> lf  (gauge f(alpha) coefficient)
    RENAMES.append((r"\blambda_f\b", "lf"))

    # The vikr cascade body references a scalar `eta` for gauge-B damping.
    # In EMDA that role is played by `etadamp` (and the symbol `eta` is
    # already taken as a 2-element array for matter-sector damping).
    # Rename, but don't touch eta[...] subscripts (those are EMDA's matter
    # eta) or longer identifiers like `etadamp` itself or `eta_*`.
    RENAMES.append((r"(?<![A-Za-z0-9_])eta(?![A-Za-z0-9_\[])", "etadamp"))


_add_field_renames()


def translate(text: str) -> str:
    out = text
    for pat, repl in RENAMES:
        out = re.sub(pat, repl, out)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="vikr cascade .cpp output")
    p.add_argument("--output", required=True, help="EMDA-renamed output path")
    args = p.parse_args()

    with open(args.input) as f:
        src = f.read()

    dst = translate(src)
    with open(args.output, "w") as f:
        f.write("// Auto-translated from vikr's BSSN cascade output by\n")
        f.write("// codegen/rename_cascade_to_emda.py. Do not edit by hand.\n")
        f.write("// Source: " + args.input + "\n//\n")
        f.write(dst)
    print(f"Wrote {args.output} ({len(dst)} bytes)")


if __name__ == "__main__":
    main()
