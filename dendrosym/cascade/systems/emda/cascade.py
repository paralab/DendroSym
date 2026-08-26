"""emda_cascade.py -- EMDA cascade driver via CascadeBuilder.

Route 1 of the cascade-generalisation plan. Replaces the text-splicing path
(`rename_cascade_to_emda.py` + `emda_cascade_unified.py`) with a direct
CascadeBuilder pipeline:

    1. Pull the BSSN cascade specs from bssn_cascade.build_specs() -- those
       are correct vacuum-BSSN symbolic expressions, computed against vikr's
       dendro/bssn modules.
    2. Apply EMDA naming dialect everywhere: rewrite leaf symbols
       (K -> trK, B0 -> gaugeB0, a -> alpha, ...) inside the SymPy
       expressions, AND rewrite chunk output keys (Gt_rhs0 -> CAP_Gt_rhs0,
       a_rhs -> alpha_rhs, ...).
    3. Feed the renamed chunks straight into CascadeBuilder.
    4. emit_cpp_unrolled() -> EMDA-named cascade body, ready to drop into
       harness_emda/src/gencode/.

Scope of v1: vacuum BSSN cascade only (the 7 natural chunks). Matter sector
(matter-source overlay + 12 matter RHS) is a v2 follow-up; for now the
matter-source `.cpp.inc` and matter-rhs `.cpp.inc` are still produced by
emda_matter_source.py / emda_matter_rhs.py and spliced separately. The win
of v1 is removing the rename_cascade_to_emda.py text-substitution layer.

Usage:
    python3 emda_cascade.py --output ../harness_emda/src/gencode/bssneqs_cascade_emda.cpp
    python3 emda_cascade.py --emit-stdout                       # print to stdout
    python3 emda_cascade.py --self-test                         # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Iterable, Mapping, Optional

import sympy as sym

from dendrosym.cascade.systems.bssn import cascade as bssn_cascade  # noqa: E402
from dendrosym.cascade.builder import CascadeResult, build_cascade_ir  # noqa: E402
from dendrosym.cascade.dialect import (  # noqa: E402
    EMDA_DIALECT,
    bssn_to_emda_key_renames,
    bssn_to_emda_symbol_renames,
    rename_chunk,
)


# Matter fields whose RHS comes from emda-gr (not the BSSN cascade).
_MATTER_FIELDS = [
    "dilatonPhi", "kappa", "capitalPi", "capitalXi",
    "perpE0", "perpE1", "perpE2",
    "perpB0", "perpB1", "perpB2",
    "dampingPsi", "dampingPhi",
]


def _collect_free_symbols(specs) -> set:
    syms = set()
    for _name, outputs in specs:
        for v in outputs.values():
            syms.update(v.free_symbols)
    return syms


def _import_emda_gr():
    """Lazy-import emdabssn_eqns_configs + the matter-source builders. Slow
    import (~30s due to dendrosym CSE pass at module load); guarded so the
    pure-BSSN driver (--matter-off) doesn't pay it."""
    import os as _os
    emda_gr = _os.environ.get("DENDRO_EMDA_GR", _os.path.expanduser("~/research/emda-gr"))
    if emda_gr not in sys.path:
        sys.path.insert(0, emda_gr)
    import emdabssn_eqns_configs as E  # noqa: E402
    from dendrosym.cascade.systems.emda.matter_source import build_matter_sources, replace_derivs_and_funcs  # noqa: E402
    return E, build_matter_sources, replace_derivs_and_funcs


def build_matter_source_overlay(verbose: bool = False):
    """Return {emda_rhs_key: sym.Expr} for the 13 += matter contributions.
    Keys are stripped of `[pp]` (e.g. 'trK_rhs', 'At00_rhs', 'CAP_Gt0_rhs',
    'gaugeB0_rhs') so they match the rhs_assembly chunk keys directly."""
    _E, build_matter_sources, replace_derivs = _import_emda_gr()
    if verbose:
        print("[emda_cascade] computing matter-source overlay...")
    overlay = {}
    for lhs, rhs in build_matter_sources():
        # 'trK_rhs[pp]' -> 'trK_rhs'
        key = lhs.replace("[pp]", "")
        expr = replace_derivs(sym.S(rhs))
        overlay[key] = expr
    if verbose:
        print(f"  matter-source overlay: {len(overlay)} += contributions")
    return overlay


def build_matter_rhs_outputs(verbose: bool = False):
    """Return OrderedDict[matter_field_rhs -> sym.Expr] for the 12 matter
    field RHS. Uses E.dendroConfigs.get_rhs_functions_all('evolution')."""
    from collections import OrderedDict
    E, _build_ms, replace_derivs = _import_emda_gr()
    if verbose:
        print("[emda_cascade] pulling matter RHS from emda-gr config...")
    exprs, names, *_ = E.dendroConfigs.get_rhs_functions_all("evolution")
    by_name = {str(n): e for n, e in zip(names, exprs)}

    out = OrderedDict()
    missing = []
    for fld in _MATTER_FIELDS:
        key = f"{fld}_rhs"
        # emda-gr stores LHS as 'dilatonPhi_rhs' (no [pp] suffix here).
        if key in by_name:
            out[key] = replace_derivs(sym.S(by_name[key]))
        elif f"{key}[pp]" in by_name:
            out[key] = replace_derivs(sym.S(by_name[f"{key}[pp]"]))
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"matter RHS missing from emda-gr config: {missing}")
    if verbose:
        print(f"  matter RHS: {len(out)} outputs")
    return out


def build_emda_specs(with_matter: bool = True, verbose: bool = False,
                     hoist_exp: bool = False):
    """Build the EMDA cascade chunks.

    With matter (default): 8 chunks -- the 7 BSSN cascade layers (with matter
    source contributions folded into rhs_assembly) plus a final matter_rhs
    chunk.  Without matter: 7 chunks, pure vacuum-BSSN-with-EMDA-naming.

    Returns (specs, leaves) compatible with bssn_cascade.build_ir's collapse/split.
    """
    bssn_specs, _ = bssn_cascade.build_specs()

    # Rename tables built from actual symbols / keys (no stale entries).
    sym_renames = bssn_to_emda_symbol_renames(_collect_free_symbols(bssn_specs))
    key_renames = bssn_to_emda_key_renames(
        k for _n, out in bssn_specs for k in out.keys()
    )

    emda_specs = []
    for name, outputs in bssn_specs:
        new_outputs = rename_chunk(outputs,
                                   key_renames=key_renames,
                                   symbol_renames=sym_renames)
        emda_specs.append((name, new_outputs))

    if with_matter:
        # 1. Fold the 13 matter-source += contributions into rhs_assembly.
        overlay = build_matter_source_overlay(verbose=verbose)
        for i, (name, outputs) in enumerate(emda_specs):
            if name != "rhs_assembly":
                continue
            for key, delta in overlay.items():
                if key not in outputs:
                    raise RuntimeError(
                        f"matter overlay key {key!r} not in rhs_assembly keys: "
                        f"{list(outputs.keys())}")
                outputs[key] = outputs[key] + delta
            emda_specs[i] = (name, outputs)
            break
        # 2. Append the matter_rhs chunk (12 outputs).
        matter_rhs = build_matter_rhs_outputs(verbose=verbose)
        emda_specs.append(("matter_rhs", matter_rhs))

    if with_matter and hoist_exp:
        # Name every exponential of the state once, as its own object, so it
        # crosses one layer boundary instead of being re-evaluated in every
        # layer that uses it (per-layer CSE cannot share across boundaries).
        # Exact: the same exp of the same argument, computed once.
        from collections import OrderedDict as _OD
        seen, first_idx = _OD(), None
        for i, (_name, outputs) in enumerate(emda_specs):
            for e in outputs.values():
                for a in sorted(e.atoms(sym.exp), key=str):
                    if a not in seen:
                        seen[a] = sym.Symbol(f"expPhi{len(seen)}")
                    if first_idx is None:
                        first_idx = i
        if seen:
            chunk = _OD((s.name, a) for a, s in seen.items())
            emda_specs.insert(first_idx, ("dilaton_coupling", chunk))
            if verbose:
                print(f"  hoisted {len(seen)} exponentials into 'dilaton_coupling' "
                      f"before chunk {first_idx}")

    # Leaves = free symbols of the final specs minus chunk-output names.
    # Derived from the specs (not the renamed vacuum set) so the matter
    # overlay's emda-gr symbols are included.
    out_names = {n for _nm, outs in emda_specs for n in outs.keys()}
    new_leaves = set()
    for _nm, outs in emda_specs:
        for e in outs.values():
            new_leaves |= {s for s in e.free_symbols if s.name not in out_names}

    return emda_specs, new_leaves


def build_ir(with_matter: bool = True, target_L=None, smart_split=None,
             verbose: bool = False, auto_layers: bool = False,
             hoist_exp: bool = False) -> CascadeResult:
    """Run the EMDA cascade through the shared pipeline. Mirrors
    bssn_cascade.build_ir()'s contract, including the collapse/split L knob
    (see cascade_builder.build_cascade_ir)."""
    specs, leaves = build_emda_specs(with_matter=with_matter, verbose=verbose,
                                     hoist_exp=hoist_exp)
    return build_cascade_ir(specs, leaves, target_L=target_L,
                            smart_split=smart_split, cse_prefix="EMDA_",
                            verbose=verbose, auto_layers=auto_layers)


def emit_avx(result: CascadeResult, with_matter: bool = True,
             inline_threshold: int = 2, fused: bool = False) -> str:
    """SIMD-agnostic VEC-macro body (4-wide AVX2 / 8-wide AVX-512 depending
    on the including wrapper's macro definitions). Same emitter internals as
    the BSSN IR AVX path; the wrapper must provide `etadamp` as a VEC.

    fused=True inlines 6th-order stencils for 1st/pure-2nd derivative
    leaves (prologue swap; chunk body identical). Default OFF."""
    from dendrosym.cascade.emit import (_inline_low_use_temps, _classify_leaves,
                              _emit_avx_prologue, _emit_avx_chunks)
    if inline_threshold > 0:
        result = _inline_low_use_temps(result, threshold=inline_threshold)
    leaves = _classify_leaves(result, fused=fused)
    lines = [
        "// EMDA RHS via polynomial cascade -- IR-driven, SIMD-batched.",
        "// AUTO-GENERATED by codegen/emda_cascade.py --simd avx -- do not hand-edit.",
        f"// L = {len(result.chunks)} chunks"
        + (" (7 vacuum BSSN + matter_rhs)" if with_matter else " (vacuum)")
        + "; wrapper defines VEC macros and provides etadamp as VEC.",
    ]
    if fused:
        lines.append(
            "// FUSED: 1st/pure-2nd derivs inlined as stencils; wrapper runs "
            "mixed-2nd-only pre-pass; interior-only (bflag==0).")
    lines.append("")
    lines += _emit_avx_prologue(leaves)
    lines += _emit_avx_chunks(result, leaves)
    from dendrosym.cascade.emit import _lazy_reorder
    lines = _lazy_reorder(lines)
    return "\n".join(lines) + "\n"


def emit_cpp(result: CascadeResult, with_matter: bool = True,
             header_comment: bool = True,
             emit_style: str = "flat") -> str:
    if emit_style == "tensor-loop":
        body = result.emit_cpp_tensor_loop(EMDA_DIALECT)
    else:
        body = result.emit_cpp_unrolled(
            dendro_var_style=True,
            inline_threshold=1,
            short_names=True,
        )
    if not header_comment:
        return body
    if with_matter:
        header = (
            "// AUTO-GENERATED by codegen/emda_cascade.py -- do not hand-edit.\n"
            "// EMDA full cascade (8 chunks: 7 vacuum BSSN + matter_rhs).\n"
            "// Matter-source contributions are folded into rhs_assembly,\n"
            "// so all 36 RHS land in this one file; no separate splice.\n"
            "\n"
        )
    else:
        header = (
            "// AUTO-GENERATED by codegen/emda_cascade.py --no-matter.\n"
            "// EMDA vacuum-BSSN cascade (7 chunks, EMDA naming dialect).\n"
            "// Matter-source overlay + matter_rhs not included.\n"
            "\n"
        )
    return header + body


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _self_test() -> None:
    print("=== emda_cascade self-test ===")
    specs, leaves = build_emda_specs()
    # 7 BSSN layers + matter_rhs (with_matter defaults True)
    assert len(specs) == 8, f"expected 8 chunks, got {len(specs)}"
    vac_specs, _ = build_emda_specs(with_matter=False)
    assert len(vac_specs) == 7, f"expected 7 vacuum chunks, got {len(vac_specs)}"

    # Verify the rename hit the chunks: RHS-assembly chunk should use EMDA keys.
    rhs_chunk = dict(specs)["rhs_assembly"]
    keys = list(rhs_chunk.keys())
    # EMDA suffix convention: <name><idx>_rhs (e.g. CAP_Gt0_rhs, gt00_rhs)
    assert "alpha_rhs" in keys, f"alpha_rhs missing; got: {keys[:5]}..."
    assert "trK_rhs" in keys, "trK_rhs missing"
    assert any(k.startswith("CAP_Gt") and k.endswith("_rhs") for k in keys), \
        f"CAP_Gt*_rhs missing; got: {keys}"
    assert any(k.startswith("gaugeB") and k.endswith("_rhs") for k in keys), \
        "gaugeB*_rhs missing"
    assert any(k.startswith("beta") and k.endswith("_rhs") for k in keys), \
        "beta*_rhs missing"
    assert "gt00_rhs" in keys and "At00_rhs" in keys, "gt/At*_rhs not in EMDA form"
    print(f"  rhs_assembly chunk keys: {len(keys)} outputs")
    print(f"    sample: {keys[:6]}...{keys[-3:]}")

    # Verify EMDA symbols appear in the renamed expressions (not BSSN names).
    found_emda_syms = set()
    found_bssn_residue = set()
    for _, outputs in specs:
        for v in outputs.values():
            for s in v.free_symbols:
                sn = str(s)
                if sn in ("trK", "alpha", "etadamp",
                          "gaugeB0", "gaugeB1", "gaugeB2",
                          "CAP_Gt0", "CAP_Gt1", "CAP_Gt2",
                          "beta0", "beta1", "beta2"):
                    found_emda_syms.add(sn)
                if sn in ("K", "a", "eta", "B0", "B1", "B2",
                          "Gt0", "Gt1", "Gt2", "b0", "b1", "b2"):
                    found_bssn_residue.add(sn)
    assert found_emda_syms, "no EMDA symbol names found in expressions"
    assert not found_bssn_residue, (
        f"residual BSSN symbols leaked through: {found_bssn_residue}")
    print(f"  EMDA symbols present: {sorted(found_emda_syms)}")
    print(f"  no BSSN residue: ok")

    # Build + emit small slice to confirm the pipeline doesn't blow up.
    print("  running CascadeBuilder.build()... ", end="", flush=True)
    result = build_ir(verbose=False)
    print(f"done. {len(result.chunks)} chunks, "
          f"{sum(len(c.outputs) for c in result.chunks)} outputs")

    cpp = emit_cpp(result, header_comment=False)
    # Sanity-check that EMDA names ended up in the C++ output.
    # rhs-output keys come out as `<key>_rhs` -> EMDA form has the index between
    # name and `_rhs`, e.g. `CAP_Gt0_rhs`, `gaugeB0_rhs`, `gt00_rhs`, `At00_rhs`.
    for tok in ("alpha_rhs", "trK_rhs", "CAP_Gt0_rhs", "gaugeB0_rhs",
                "etadamp", "beta0_rhs", "gt00_rhs", "At00_rhs"):
        assert tok in cpp, f"{tok!r} missing from emitted C++"
    import re
    # Use word-boundary checks so 'a_rhs' doesn't false-match inside 'alpha_rhs'.
    forbidden_word = [r"\ba_rhs\b", r"\bK_rhs\b",
                      r"\bGt_rhs0\b", r"\bB_rhs0\b", r"\bb_rhs0\b",
                      r"\bgt_rhs00\b", r"\bAt_rhs00\b"]
    for pat in forbidden_word:
        m = re.search(pat, cpp)
        assert m is None, f"BSSN-named token leaked: {pat!r} at offset {m.start()}"
    # Also verify a [pp]-suffixed leaf got renamed:
    assert "trK[pp]" in cpp, "leaf rename K[pp]->trK[pp] failed"
    assert re.search(r"\bK\[pp\]", cpp) is None, "leaf K[pp] still present"
    print(f"  emit_cpp: {len(cpp):,} chars, all EMDA tokens present")

    print("=== all self-tests passed ===")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", help="path to write emitted C++")
    ap.add_argument("--emit-stdout", action="store_true",
                    help="print emitted C++ to stdout instead of writing a file")
    ap.add_argument("--self-test", action="store_true",
                    help="run internal smoke tests and exit")
    ap.add_argument("--no-matter", action="store_true",
                    help="emit vacuum-BSSN-with-EMDA-naming only (no matter chunks)")
    ap.add_argument("--L", type=int, default=None,
                    help="target cascade depth (default: natural 8, or 7 with "
                         "--no-matter)")
    ap.add_argument("--split-mode", choices=("auto", "smart", "dumb"),
                    default="auto",
                    help="split strategy when L > natural (default auto: "
                         "smart for even delta, dumb otherwise)")
    ap.add_argument("--emit-style", choices=["flat", "tensor-loop"], default="flat",
                    help="output shape; default 'flat' matches existing behaviour. "
                         "'tensor-loop' is opt-in experimental.")
    ap.add_argument("--simd", choices=["scalar", "avx"], default="scalar",
                    help="'avx' emits the SIMD-agnostic VEC-macro body "
                         "(compiles 4-wide or 8-wide depending on wrapper).")
    ap.add_argument("--fused", action="store_true",
                    help="inline stencils for 1st/pure-2nd derivs (requires "
                         "--simd avx). Default OFF; non-fused stays primary.")
    ap.add_argument("--auto", action="store_true",
                    help="choose layer boundaries automatically (exact DP)")
    ap.add_argument("--hoist-exp", action="store_true",
                    help="name each exponential of the state as its own object "
                         "so it is evaluated once per point instead of once per "
                         "layer that uses it (exact; default off)")
    ap.add_argument("--verbose", action="store_true",
                    help="verbose CSE / build output")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    if not args.output and not args.emit_stdout:
        ap.error("either --output or --emit-stdout is required (or --self-test)")

    smart = {"auto": None, "smart": True, "dumb": False}[args.split_mode]
    result = build_ir(with_matter=not args.no_matter, target_L=args.L,
                      auto_layers=args.auto, hoist_exp=args.hoist_exp,
                      smart_split=smart, verbose=args.verbose)
    if args.simd == "avx":
        cpp = emit_avx(result, with_matter=not args.no_matter,
                       fused=args.fused)
    elif args.fused:
        ap.error("--fused requires --simd avx")
    else:
        cpp = emit_cpp(result, with_matter=not args.no_matter,
                       emit_style=args.emit_style)
    if args.emit_stdout:
        sys.stdout.write(cpp)
    else:
        with open(args.output, "w") as f:
            f.write(cpp)
        print(f"[emda_cascade] wrote {len(cpp):,} chars to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
