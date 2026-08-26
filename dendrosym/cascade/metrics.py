"""cascade_metrics.py -- one correct tool for FLOPs + register pressure.

Replaces the three ad-hoc tools (count_flops.py / count_machine_flops.py /
analyze_pressure.py). Two levels of measurement, clearly labelled:

  MACHINE (authoritative -- what the CPU runs, post FMA/CSE/fold):
    objdump the .o, scoped to ONE function symbol (the old tool counted the
    whole .o and double-counted multi-function TUs), and count:
      * fp_instr  -- scalar (*sd) AND packed (*pd) FP arithmetic instructions
      * flops     -- per-point FLOPs: add/sub/mul/div/sqrt = 1, FMA = 2
                     (packed counts as 1 *per point*: one lane per grid point)
      * spills    -- "FP stack ops": vector mov touching a [rsp/[rbp slot
                     (the register-pressure proxy; spill+reload traffic)

  ALGEBRAIC (source level, exact via SymPy):
    sympy.count_ops over the BUILT cascade's per-chunk expressions (post-CSE,
    cross-chunk outputs as name atoms). A reproducible scalar-level op count.
    NOTE: this is NOT the paper's "954" -- that is a higher-level tensor-op
    hand count; count_ops is finer-grained (counts every Add/Mul node).

Usage:
  python cascade_metrics.py machine  <obj.o> [--symbol SUBSTR]
  python cascade_metrics.py compare  <build_rhsfuncs_dir> [--names a b c]
  python cascade_metrics.py algebraic [--gauge standard] [--ssl] [--cahd]

Compiler-agnostic: reads the ELF .o via objdump, so it works on gcc / clang /
icpx output identically (ISA-level mnemonics; shared Itanium mangling makes the
--symbol substring match work). Caveat: transcendentals routed through SVML/libm
become `call`s and are NOT counted -- matters only for exp/log kernels (e.g.
Neo-Hookean), not BSSN (whose only non-arith op, sqrt, IS counted).
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict

# --- FP arithmetic mnemonics (scalar *sd/*ss + packed *pd/*ps; AVX v-prefixed) ---
_ARITH = ("add", "sub", "mul", "div", "sqrt", "max", "min")
_SUFFIX = ("sd", "ss", "pd", "ps")
_FP_MOV = {"movsd", "movss", "movapd", "movaps", "movupd", "movups",
           "vmovsd", "vmovss", "vmovapd", "vmovaps", "vmovupd", "vmovups"}


def _is_fma(mnem: str) -> bool:
    return mnem.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")) and \
        mnem.endswith(_SUFFIX)


def _is_arith(mnem: str) -> bool:
    m = mnem[1:] if mnem.startswith("v") else mnem
    return any(m == a + s for a in _ARITH for s in _SUFFIX)


def _disasm_by_symbol(obj_path: str) -> "OrderedDict[str, list]":
    """objdump -d, split into {symbol: [(mnemonic, operands), ...]} per function."""
    out = subprocess.check_output(
        ["objdump", "-d", "-M", "intel", "--no-show-raw-insn", obj_path],
        text=True, stderr=subprocess.DEVNULL)
    funcs: "OrderedDict[str, list]" = OrderedDict()
    cur = None
    hdr = re.compile(r"^[0-9a-f]+ <(.+)>:$")
    for line in out.splitlines():
        h = hdr.match(line)
        if h:
            cur = h.group(1)
            funcs[cur] = []
            continue
        if cur is None or "\t" not in line:
            continue
        # "   0:\taddsd  xmm0,xmm1"  (raw bytes suppressed by --no-show-raw-insn)
        insn = line.split("\t", 1)[1].strip()
        parts = insn.split(None, 1)
        mnem = parts[0]
        ops = parts[1] if len(parts) > 1 else ""
        funcs[cur].append((mnem, ops))
    return funcs


def count_machine(obj_path: str, symbol: str | None = None) -> dict:
    """Per-symbol machine FP metrics. Picks the FP-heaviest symbol matching
    `symbol` (or the single FP-heaviest function if symbol is None)."""
    funcs = _disasm_by_symbol(obj_path)
    cands = {s: ins for s, ins in funcs.items()
             if symbol is None or symbol in s}
    if not cands:
        raise SystemExit(f"no symbol matching {symbol!r} in {obj_path}\n"
                         f"  symbols: {list(funcs)[:8]}")

    def score(ins):
        return sum(1 for m, _ in ins if _is_arith(m) or _is_fma(m))

    sym = max(cands, key=lambda s: score(cands[s]))
    ins = cands[sym]

    c = dict(scalar=0, packed=0, fma=0, div=0, sqrt=0, spills=0, total=len(ins))
    flops = 0
    for mnem, ops in ins:
        if _is_fma(mnem):
            c["fma"] += 1
            flops += 2
            (c.__setitem__("packed", c["packed"] + 1) if mnem.endswith(("pd", "ps"))
             else c.__setitem__("scalar", c["scalar"] + 1))
        elif _is_arith(mnem):
            flops += 1
            base = mnem[1:] if mnem.startswith("v") else mnem
            if base.startswith("div"):
                c["div"] += 1
            elif base.startswith("sqrt"):
                c["sqrt"] += 1
            (c.__setitem__("packed", c["packed"] + 1) if mnem.endswith(("pd", "ps"))
             else c.__setitem__("scalar", c["scalar"] + 1))
        if mnem in _FP_MOV and ("[rsp" in ops or "[rbp" in ops):
            c["spills"] += 1            # spill or reload to a stack slot
    c["fp_instr"] = c["scalar"] + c["packed"]
    c["flops"] = flops
    c["symbol"] = sym
    return c


def count_algebraic(gauge="standard", ssl=False, cahd=False) -> dict:
    """Exact pre-compiler algebraic op count via SymPy over the BUILT cascade
    (post per-chunk CSE; cross-chunk outputs appear as name atoms, not inlined)."""
    import sympy as sym
    from dendrosym.cascade.systems.bssn import clean as bssn_clean
    result = bssn_clean.build_ir(gauge=gauge, ssl=ssl, cahd=cahd)
    per_chunk = OrderedDict()
    total = 0
    for c in result.chunks:
        n = sum(int(sym.count_ops(e)) for _s, e in c.cse_temps)
        n += sum(int(sym.count_ops(e)) for e in c.outputs.values())
        per_chunk[c.name] = n
        total += n
    return {"total": total, "per_chunk": per_chunk}


# ----------------------------------------------------------------------------
def _machine_cmd(args):
    c = count_machine(args.obj, args.symbol)
    print(f"symbol: {c['symbol']}")
    print(f"  fp_instr {c['fp_instr']:5d}  (scalar {c['scalar']}, packed {c['packed']}, "
          f"fma {c['fma']}, div {c['div']}, sqrt {c['sqrt']})")
    print(f"  FLOPs/pt {c['flops']:5d}  (FMA=2)")
    print(f"  spills   {c['spills']:5d}  (FP stack ops -> register pressure)")


def _compare_cmd(args):
    names = args.names or ["production", "naive", "cascade"]
    rows = []
    for nm in names:
        objs = [f for f in os.listdir(args.dir)
                if nm in f and f.endswith(".o")]
        if not objs:
            print(f"  [{nm}] no .o found in {args.dir}", file=sys.stderr)
            continue
        c = count_machine(os.path.join(args.dir, sorted(objs, key=len)[0]), nm)
        rows.append((nm, c))
    print(f"{'variant':<22} {'fp_instr':>8} {'FLOPs':>7} {'spills':>7}")
    print("-" * 48)
    base = rows[0][1] if rows else None
    for nm, c in rows:
        r = f"  ({c['flops']/base['flops']:.2f}x)" if base else ""
        print(f"{nm:<22} {c['fp_instr']:>8} {c['flops']:>7} {c['spills']:>7}{r}")


def _algebraic_cmd(args):
    r = count_algebraic(gauge=args.gauge, ssl=args.ssl, cahd=args.cahd)
    print(f"algebraic ops (SymPy count_ops), gauge={args.gauge} ssl={args.ssl} cahd={args.cahd}")
    for name, n in r["per_chunk"].items():
        print(f"  {name:<22} {n:>5}")
    print(f"  {'TOTAL':<22} {r['total']:>5}")
    print("  (SymPy count_ops, scalar-level -- finer than the paper's tensor-op 954)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("machine", help="per-symbol FP metrics from a .o")
    m.add_argument("obj"); m.add_argument("--symbol", default=None)
    m.set_defaults(fn=_machine_cmd)

    cp = sub.add_parser("compare", help="table over a build rhsfuncs dir")
    cp.add_argument("dir"); cp.add_argument("--names", nargs="+", default=None)
    cp.set_defaults(fn=_compare_cmd)

    al = sub.add_parser("algebraic", help="exact SymPy op count of the cascade")
    al.add_argument("--gauge", default="standard")
    al.add_argument("--ssl", action="store_true"); al.add_argument("--cahd", action="store_true")
    al.set_defaults(fn=_algebraic_cmd)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
