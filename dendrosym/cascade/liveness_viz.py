#!/usr/bin/env python3
"""liveness_viz.py -- data-driven register-pressure figures for the cascade paper.

Two figures, two REAL data sources. No invented numbers: every value traces to a
parsed .cpp or a disassembled .o, and the tool prints its provenance.

  source  parses the generated C++ kernels (production CSE vs cascade) and computes the
          true logical live-temporary count over the AS-EMITTED statement order, plus
          each temporary's lifespan. Shows production's wide, sustained live set vs the
          cascade's layer-bounded sawtooth.        -> liveness_source.{pdf,png}

  object  disassembles the SCALAR object files and counts the true stack spills the
          compiler actually emitted, over the instruction stream. Production spills 1719
          times, cascade 558 -- the hardware consequence of the source picture. Self-tests
          against those anchors before plotting.   -> liveness_object.{pdf,png}

Reuses codegen/cascade_metrics.py (_disasm_by_symbol, _FP_MOV spill rule) and, for an
optional cross-check, cascade/regopt.py + codegen/bssn_clean.py.

Fidelity notes (see plan eager-brewing-glacier.md):
 * The SOURCE liveness is the true *logical* live set (uncapped) -- it can exceed the
   16-register file, which is exactly what predicts spilling.
 * The OBJECT side reports true *measured spills*; physical-register liveness saturates
   at the file size for both variants, so spills (not register count) are the real signal.
"""
import os
import re
import sys
import argparse
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# vikr checkout root (paperfig + harness build trees); override with CASCADE_VIKR_ROOT.
VIKR = os.environ.get("CASCADE_VIKR_ROOT", os.path.expanduser("~/research/vikr"))
if VIKR not in sys.path:
    sys.path.insert(0, VIKR)
try:
    import paperfig  # figure styling; only the figure_* entry points need it
except ImportError:  # pragma: no cover
    paperfig = None
GEN = os.path.join(VIKR, "harness/src/rhsfuncs/generated")
SWEEP = os.path.join(VIKR, "harness/.sweep_builds/g++-15/CMakeFiles/BSSNRHSTests.dir/src/rhsfuncs")

DEF_PROD_CPP = os.path.join(GEN, "bssneq_PRODUCTION.cpp")
DEF_CASC_CPP = os.path.join(GEN, "bssneqs_cascade.cpp")
DEF_CASC_IR_L8 = os.path.join(GEN, "bssneqs_cascade_ir_L8.cpp")  # named per-layer emission
DEF_PROD_O = os.path.join(SWEEP, "bssn_production_rhs.cpp.o")
DEF_CASC_O = os.path.join(SWEEP, "bssn_cascade_rhs.cpp.o")
DEF_IR_L_O = os.path.join(SWEEP, "bssn_cascade_ir_l_rhs.cpp.o")  # one .o, a symbol per depth L


def system_spec(name):
    """Per-system paths so the sweep/depth/object figures are not BSSN-hardcoded.
    cpp_L(L)/obj_L(L)/sym_L(L) map a depth L to its source file / object file / symbol hint."""
    if name == "emda":
        G = os.path.join(VIKR, "harness_emda/src/gencode")
        B = os.path.join(VIKR, "harness_emda/build/CMakeFiles/EMDARHSTests.dir/src/rhsfuncs")
        return SimpleNamespace(
            name="emda", prefix="liveness_emda", casc_label="EMDA cascade", Ls=(1, 6, 7, 9, 10),
            prod_cpp=os.path.join(G, "emda_rhs_eqns_ALL_VARS.cpp.inc"),
            natural_cpp=os.path.join(G, "bssneqs_cascade_emda_unified_v2.cpp"),
            cpp_L=lambda L: os.path.join(G, f"bssneqs_cascade_emda_unified_v2_L{L}.cpp"),
            prod_o=os.path.join(B, "emda_production_rhs.cpp.o"), prod_hint="emda_production_rhs",
            casc_o=os.path.join(B, "emda_cascade_unified_v2_rhs.cpp.o"),
            casc_hint="emda_cascade_unified_v2_rhs",
            obj_L=lambda L: os.path.join(B, f"emda_cascade_unified_v2_L{L}_rhs.cpp.o"),
            sym_L=lambda L: f"emda_cascade_unified_v2_L{L}_rhs")
    return SimpleNamespace(
        name="bssn", prefix="liveness", casc_label="cascade", Ls=(1, 5, 6, 7, 8, 9),
        prod_cpp=DEF_PROD_CPP, natural_cpp=DEF_CASC_IR_L8,
        cpp_L=lambda L: os.path.join(GEN, f"bssneqs_cascade_ir_L{L}.cpp"),
        prod_o=DEF_PROD_O, prod_hint="bssn_production_rhs",
        casc_o=DEF_CASC_O, casc_hint="bssn_cascade_rhs",
        obj_L=lambda L: DEF_IR_L_O, sym_L=lambda L: f"bssn_cascade_ir_L{L}_rhs")

# Verified reproducible from the binaries 2026-06-29 (objdump re-derivation == register_pressure.json).
SPILL_ANCHOR = {"production": 1719, "cascade": 558}
XMM = 16  # scalar x86-64 architectural xmm register file

# Palette + styling come from the shared paperfig package (single source of truth).
C_PROD, C_CASC = (paperfig.C_PROD, paperfig.C_CASC) if paperfig else (None, None)
# short, pretty names for the cascade chunk labels read from the markers
PRETTY = {
    "global": "all (flat)", "inputs": "inputs",
    "inverse_metric": "inv. metric",
    "first_christoffel": "Γ(1st)", "second_christoffel": "Γ(2nd)",
    "complete_christoffel": "Γ(full)",
    "ricci": "Ricci", "ricci_shared": "Ricci(shr)",
    "ricci_p0": "Ricci(a)", "ricci_p1": "Ricci(b)",
    "derived_quantities": "derived",
    "rhs_assembly": "RHS", "rhs_assembly_p0": "RHS(a)", "rhs_assembly_p1": "RHS(b)",
    "matter_rhs": "matter",   # EMDA matter sector (dilaton/Maxwell/damping)
}


def _pretty_layer(label):
    """Turn a marker label into a short legend name; merged chunks (joined by '_x_')
    collapse to 'first … last'."""
    parts = [PRETTY.get(p, p.replace("_", " ")) for p in label.split("_x_")]
    if len(parts) <= 2:
        return " + ".join(parts)
    return f"{parts[0]} … {parts[-1]}"


# ---------------------------------------------------------------------------
# C++ source parsing -> true logical liveness over the as-emitted order
# ---------------------------------------------------------------------------
_DEF_RE = re.compile(r"^\s*(?:const\s+)?double\s+([A-Za-z_]\w*)\s*=(.*)$", re.S)
_OUT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*pp\s*\]\s*\+?=(.*)$", re.S)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# A bare load/copy: a SINGLE token (grid access xxx[pp], param xxx[i], or a name),
# optionally negated, with NO operators or function calls. If the token is a previously
# defined local it is an ALIAS (symmetry copy, e.g. igt10 = igt01) and must resolve to that
# local's compute root so the dependency chain is not severed. If the token is a grid/param
# it is an input LOAD (al = alpha[pp]). Anything with operators (2*alpha[pp], pow(gt4[pp],2))
# is genuine COMPUTE -- exactly as production treats its inline inputs.
_BARE_RE = re.compile(r"^-?\s*([A-Za-z_]\w*)(?:\s*\[\s*(?:pp|\d+)\s*\])?$")


class Stmt:
    __slots__ = ("idx", "line", "name", "kind", "root", "rhs", "uses", "layer")

    def __init__(self, idx, line, name, kind, root, rhs, uses, layer):
        self.idx = idx
        self.line = line
        self.name = name
        self.kind = kind        # 'leaf' | 'alias' | 'compute' | 'output'
        self.root = root        # canonical compute name (compute / alias-of-compute) else None
        self.rhs = rhs
        self.uses = uses        # set of COMPUTE roots this statement reads
        self.layer = layer


def parse_kernel(path):
    """Parse straight-line single-assignment C++ with ALIAS RESOLUTION.

    Each `const double NAME = RHS;` is classified:
      * input LOAD  -- RHS is a bare grid/param load (al = alpha[pp]); a leaf, excluded.
      * ALIAS       -- RHS is a single previously-defined local (igt10 = igt01); resolved to
                       that local's canonical root so symmetry copies never sever the use-chain.
      * COMPUTE     -- anything with operators/functions; a live value.
    Every statement's `uses` are resolved to the set of COMPUTE roots it reads, so which
    symmetry copy is referenced is irrelevant. Returns (stmts, compute_roots).
    """
    with open(path) as f:
        raw = f.readlines()
    layer_of_line = []
    layer_labels = {0: "inputs"}     # each file's OWN layer names (read from the markers)
    cur = 0
    for ln in raw:
        if "// ===" in ln:
            cur += 1
            mk = re.search(r"//\s*===\s*(.*?)\s*===", ln)
            lab = mk.group(1) if mk else ""
            lab = re.sub(r"\s*\([^()]*\)\s*$", "", lab)   # drop "(N temps, M outputs)"
            lab = re.sub(r"^L\d*:\s*", "", lab).strip()    # drop "L1:" / "L:" prefix
            layer_labels[cur] = lab or f"layer {cur}"
        layer_of_line.append(cur)

    nkind = {}   # name -> 'leaf' | 'compute'
    croot = {}   # name -> canonical root name
    stmts = []
    buf = ""
    buf_start = None

    def resolved_uses(rhs):
        return {croot[t] for t in _IDENT_RE.findall(rhs)
                if t in croot and nkind[croot[t]] == "compute"}

    for i, ln in enumerate(raw):
        code = ln.split("//", 1)[0]
        if buf_start is None and code.strip() == "":
            continue
        if buf_start is None:
            buf_start = i
        buf += " " + code
        if ";" not in code:
            continue
        text = buf.strip().rstrip(";").strip()
        start = buf_start
        layer = layer_of_line[buf_start]
        buf = ""
        buf_start = None

        mo = _OUT_RE.match(text)
        if mo:
            stmts.append(Stmt(len(stmts), start + 1, mo.group(1) + "[pp]", "output",
                              None, mo.group(2), resolved_uses(mo.group(2)), layer))
            continue
        m = _DEF_RE.match(text)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        bare = _BARE_RE.match(rhs.strip())
        if bare and bare.group(1) in croot:        # alias of a previously-defined local
            tgt = croot[bare.group(1)]
            croot[name] = tgt
            nkind[name] = nkind[tgt]
            uses = {tgt} if nkind[tgt] == "compute" else set()
            stmts.append(Stmt(len(stmts), start + 1, name, "alias", tgt, rhs, uses, layer))
        elif bare:                                  # bare grid/param load -> leaf input
            croot[name] = name
            nkind[name] = "leaf"
            stmts.append(Stmt(len(stmts), start + 1, name, "leaf", name, rhs, set(), layer))
        else:                                       # genuine compute -> a live value
            croot[name] = name
            nkind[name] = "compute"
            stmts.append(Stmt(len(stmts), start + 1, name, "compute", name, rhs,
                              resolved_uses(rhs), layer))

    compute_roots = {n for n, k in nkind.items() if k == "compute"}
    return stmts, compute_roots, layer_labels


def source_liveness(stmts):
    """True logical liveness of COMPUTE values over the as-emitted order.

    Only compute defs occupy a value slot; aliases reuse their root's slot; leaves/inputs are
    free (like production's inline operands). The trace is recorded over EVENT positions
    (compute defs + output sinks only) so the x-axis isn't padded with input-load dead-space.
    Mirrors regopt.analyze_pressure.
    """
    events = [s for s in stmts if s.kind in ("compute", "output")]
    epos = {id(s): k for k, s in enumerate(events)}

    last_use = {}
    for s in events:
        for r in s.uses:
            last_use[r] = epos[id(s)]

    def_pos = {}
    layer_of = {}
    live = set()
    trace = []
    for s in events:
        p = epos[id(s)]
        if s.kind == "compute":
            live.add(s.root)
            def_pos[s.root] = p
            layer_of[s.root] = s.layer
        dead = {r for r in live if last_use.get(r, -1) <= p}
        live -= dead
        trace.append(len(live))

    lifespans = {r: (def_pos[r], last_use.get(r, def_pos[r])) for r in def_pos}
    return {
        "trace": trace,
        "lifespans": lifespans,
        "layer_of": layer_of,
        "n_stmts": len(events),     # events (compute defs + outputs) -- the x-axis basis
        "n_temps": len(def_pos),
        "peak": max(trace) if trace else 0,
    }


# ---------------------------------------------------------------------------
# Object-file analysis -> true measured spills over the instruction stream
# ---------------------------------------------------------------------------
def _load_cm():
    from dendrosym.cascade import metrics as cascade_metrics
    return cascade_metrics


def hot_symbol(obj, hint):
    """Return (symbol, [(mnem, ops), ...]) for the largest symbol matching the hint."""
    cm = _load_cm()
    funcs = cm._disasm_by_symbol(obj)
    cands = [s for s in funcs if hint in s] or list(funcs)
    if not cands:
        raise RuntimeError(f"no symbols in {obj}")
    sym = max(cands, key=lambda s: len(funcs[s]))
    return sym, funcs[sym]


_MOV_SSE = {"movsd", "movss", "movapd", "movaps", "movupd", "movups"}


def _is_spill(mnem, ops):
    """An FP move to/from a stack slot (spill or reload). Broad `vmov*` rule -- matches
    the verified anchor counts (production 1719, cascade 558); the narrow whitelist in
    cascade_metrics._FP_MOV misses ~6 vmov variants (vmovq/vmovhpd/...) and undercounts."""
    if "[rsp" not in ops and "[rbp" not in ops:
        return False
    return mnem.startswith("vmov") or mnem in _MOV_SSE


def spill_flags(insns):
    """Per-instruction 1/0: is it an FP move to/from a stack slot (spill or reload)?"""
    return [1 if _is_spill(m, ops) else 0 for m, ops in insns]


def _spill_side(ops):
    """intel `dst, src`: stack slot as dst -> store (spill); as src -> reload (load)."""
    head = ops.split(",", 1)[0]
    return "store" if ("[rsp" in head or "[rbp" in head) else "load"


def object_spills(obj, hint, label):
    cm = _load_cm()
    sym, insns = hot_symbol(obj, hint)
    flags = spill_flags(insns)
    total = sum(flags)
    fp = sum(1 for m, _ in insns if cm._is_arith(m) or cm._is_fma(m))
    loads = stores = 0
    samples = []
    for (m, ops), f in zip(insns, flags):
        if not f:
            continue
        if _spill_side(ops) == "store":
            stores += 1
        else:
            loads += 1
        if len(samples) < 4:
            samples.append(f"{m} {ops}".strip())
    print(f"  {label:<11} {os.path.relpath(obj, VIKR)}")
    print(f"              symbol={sym[:60]}{'...' if len(sym) > 60 else ''}")
    print(f"              instructions={len(insns)}  FP-arith={fp}  spills(FP<->stack)={total}")
    print(f"              spill breakdown: {loads} reloads + {stores} stores")
    print(f"              sample spills: {'  |  '.join(samples)}")
    # Soft confidence check: the anchors are the values independently re-derived from the
    # known scalar g++-15 build. We never gate on them -- the tool plots whatever the real
    # binary contains -- but flag agreement so you know the count is the verified one.
    anchor = SPILL_ANCHOR.get(label)
    is_default = os.path.abspath(obj) in (os.path.abspath(DEF_PROD_O), os.path.abspath(DEF_CASC_O))
    if anchor is not None and is_default:
        if total == anchor:
            print(f"              [confidence] matches verified scalar-g++15 anchor {anchor}")
        else:
            print(f"              [note] {total} != verified anchor {anchor}")
    return {"insns": len(insns), "flags": flags, "total": total, "symbol": sym}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style():
    return paperfig.apply_style()


def _save(fig, out_base):
    paperfig.save(fig, out_base, root=VIKR)


def _layer_spans(res):
    """Event-axis (start, end) of each layer, from the definition positions of the
    values it introduces. The trace is indexed by event position, so these line up
    with the curve without any rescaling."""
    spans = {}
    for root, (dpos, _last) in res["lifespans"].items():
        lay = res["layer_of"].get(root)
        if lay is None:
            continue
        s0, s1 = spans.get(lay, (dpos, dpos))
        spans[lay] = (min(s0, dpos), max(s1, dpos))
    return dict(sorted(spans.items(), key=lambda kv: kv[1][0]))


def _peak_layer(res, spans):
    """Which layer the global peak occurs in. This is the 'one layer sets the peak'
    result: for BSSN and EMDA alike it comes out as Ricci."""
    trace = res["trace"]
    if not trace:
        return None
    pk = max(range(len(trace)), key=lambda i: trace[i])
    for lay, (s0, s1) in spans.items():
        if s0 <= pk <= s1:
            return lay
    return None


def figure_source(prod, casc, out_base, casc_label="cascade", casc_layer_labels=None):
    import numpy as np
    plt = _style()
    pl = source_liveness(prod)
    cl = source_liveness(casc)
    tab = plt.get_cmap("tab10")
    def lcolor(l):                       # distinct color per layer (no clamping/repeats)
        return tab((l - 1) % 10)

    fig = plt.figure(figsize=paperfig.FIG["composite"], constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])

    # --- top: overlaid live-value curves (normalized kernel progress) ---
    axA = fig.add_subplot(gs[0, :])
    xp = np.arange(pl["n_stmts"]) / max(1, pl["n_stmts"] - 1)
    xc = np.arange(cl["n_stmts"]) / max(1, cl["n_stmts"] - 1)

    # Layer boundaries on the cascade curve. Without these the sawtooth reads as a
    # property of the plot; with them it is visibly a consequence of the construction,
    # since the live set collapses exactly where a layer ends and its outputs become
    # the only survivors. Also marks WHICH layer sets the peak, which is otherwise a
    # prose-only result.
    spans = _layer_spans(cl)
    peak_layer = _peak_layer(cl, spans)
    denom = max(1, cl["n_stmts"] - 1)
    ytop = max(pl["peak"], cl["peak"]) * 1.12
    for lay, (s0, s1) in spans.items():
        if lay == 0:
            continue
        a, b = s0 / denom, s1 / denom
        if lay == peak_layer:
            axA.axvspan(a, b, color=C_CASC, alpha=0.10, lw=0, zorder=0)
        axA.axvline(a, color="0.75", lw=0.6, ls=":", zorder=0)
        # Only label a layer wide enough to hold the text. The early layers are
        # narrow and their names would collide into an unreadable smear; their
        # boundaries still show as rules, and the gantt legend below names all of
        # them anyway, so nothing is lost.
        if (b - a) >= 0.055:
            name = _pretty_layer((casc_layer_labels or {}).get(lay, f"L{lay}"))
            axA.text((a + b) / 2, ytop * 0.97, name, rotation=90, ha="center", va="top",
                     fontsize=6.5, color=C_CASC if lay == peak_layer else "0.45", zorder=1)

    axA.plot(xp, pl["trace"], color=C_PROD, lw=1.4, label=f"production (peak {pl['peak']})")
    axA.plot(xc, cl["trace"], color=C_CASC, lw=1.4, label=f"{casc_label} (peak {cl['peak']})")
    axA.axhline(XMM, ls="--", color="0.4", lw=1.0)
    axA.text(0.004, XMM + 1.5, f"{XMM} regs", color="0.35", fontsize=8)
    if peak_layer is not None:
        pname = _pretty_layer((casc_layer_labels or {}).get(peak_layer, f"L{peak_layer}"))
        pk_i = max(range(len(cl["trace"])), key=lambda i: cl["trace"][i])
        pk_x = pk_i / denom
        axA.annotate(f"{pname} sets the peak ({cl['peak']})",
                     xy=(pk_x, cl["peak"]), xytext=(pk_x + 0.04, cl["peak"] * 1.45),
                     fontsize=7.5, color=C_CASC, ha="left",
                     arrowprops=dict(arrowstyle="->", color=C_CASC, lw=0.8,
                                     shrinkA=0, shrinkB=2))
    axA.set_xlim(0, 1)
    axA.set_ylim(0, ytop)
    axA.set_xlabel("kernel progress")
    axA.set_ylabel("live values")
    axA.legend(loc="upper left")
    axA.spines[["top", "right"]].set_visible(False)

    # --- bottom: lifespan gantts ---
    def gantt(ax, life, layer_of, n_stmts, title, by_layer):
        items = sorted(life.items(), key=lambda kv: kv[1][0])  # by def order
        n = len(items)
        ys = np.arange(n)
        x0 = np.array([life[k][0] for k, _ in items]) / max(1, n_stmts - 1)
        x1 = np.array([life[k][1] for k, _ in items]) / max(1, n_stmts - 1)
        if by_layer:
            cols = [lcolor(layer_of.get(k, 1)) for k, _ in items]
            ax.hlines(ys, x0, x1, colors=cols, lw=0.7, alpha=0.85)
        else:
            ax.hlines(ys, x0, x1, colors=C_PROD, lw=0.5, alpha=0.45)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, n)
        ax.set_xlabel("kernel progress")
        ax.set_ylabel("temporary")
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    axL = fig.add_subplot(gs[1, 0])
    gantt(axL, pl["lifespans"], pl["layer_of"], pl["n_stmts"],
          f"production ({pl['n_temps']} temps)", False)
    axR = fig.add_subplot(gs[1, 1])
    gantt(axR, cl["lifespans"], cl["layer_of"], cl["n_stmts"],
          f"{casc_label} ({cl['n_temps']} temps)", True)
    # layer legend on cascade gantt
    used = sorted(set(cl["layer_of"].values()))
    handles = [plt.Line2D([0], [0], color=lcolor(l), lw=3) for l in used]
    lbl = casc_layer_labels or {}
    labels = [_pretty_layer(lbl.get(l, f"layer {l}")) for l in used]
    axR.legend(handles, labels, fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    _save(fig, out_base)


def figure_object(prod, casc, out_base, window=64):
    import numpy as np
    plt = _style()

    def density(flags, w):
        f = np.array(flags, dtype=float)
        if len(f) == 0:
            return f
        k = np.ones(w) / w
        return np.convolve(f, k, mode="same") * 100.0  # percent of instrs in window

    pf, cf = prod["flags"], casc["flags"]
    xp = np.arange(len(pf))
    xc = np.arange(len(cf))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=paperfig.FIG["stack"], constrained_layout=True)

    # --- top: cumulative spills ---
    ax1.plot(xp, np.cumsum(pf), color=C_PROD, lw=1.6,
             label=f"production CSE  ({prod['total']} spills / {prod['insns']} instr)")
    ax1.plot(xc, np.cumsum(cf), color=C_CASC, lw=1.6,
             label=f"cascade  ({casc['total']} spills / {casc['insns']} instr)")
    ax1.set_xlabel("instruction index")
    ax1.set_ylabel("cumulative stack spills")
    ax1.legend(loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- bottom: spill density ---
    ax2.plot(xp, density(pf, window), color=C_PROD, lw=1.1, alpha=0.9, label="production CSE")
    ax2.plot(xc, density(cf, window), color=C_CASC, lw=1.1, alpha=0.9, label="cascade")
    ax2.set_xlabel("instruction index")
    ax2.set_ylabel(f"spill density (% / {window}-instr)")
    ax2.legend(loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    _save(fig, out_base)


# ---------------------------------------------------------------------------
# Variant sweep -- liveness across the depth knob and reference baselines
# ---------------------------------------------------------------------------
# Scalar (text-parseable) kernels only; the AVX variants use intrinsics, not const-double.
# Full-CSE kernels only (production + the IR depth knob). naive and the simple cascade are
# heavily inlined (15-24 ops/line) so named-temp liveness undercounts them ~3x -- not
# comparable on this axis, per the audit -- so they are intentionally excluded.
SWEEP = [
    ("production", "bssneq_PRODUCTION.cpp", None, C_PROD),
    ("IR L1", "bssneqs_cascade_ir_L1.cpp", 1, C_CASC),
    ("IR L5", "bssneqs_cascade_ir_L5.cpp", 5, C_CASC),
    ("IR L6", "bssneqs_cascade_ir_L6.cpp", 6, C_CASC),
    ("IR L7", "bssneqs_cascade_ir_L7.cpp", 7, C_CASC),
    ("IR L8", "bssneqs_cascade_ir_L8.cpp", 8, C_CASC),
    ("IR L9", "bssneqs_cascade_ir_L9.cpp", 9, C_CASC),
]


def figure_sweep(out_base, spec):
    """Peak/mean logical liveness: production vs the cascade depth knob L."""
    import numpy as np
    plt = _style()
    rows = []   # (label, L, color, peak, mean, ntemps)

    def add(label, L, color, path):
        if not os.path.exists(path):
            print(f"  (skip) {label}: missing")
            return
        liv = source_liveness(parse_kernel(path)[0])
        mean = sum(liv["trace"]) / len(liv["trace"]) if liv["trace"] else 0.0
        rows.append((label, L, color, liv["peak"], mean, liv["n_temps"]))
        print(f"  {label:12s} peak-live={liv['peak']:4d}  mean-live={mean:5.1f}  temps={liv['n_temps']}")

    add("production", None, C_PROD, spec.prod_cpp)
    for L in spec.Ls:
        add(f"L{L}", L, C_CASC, spec.cpp_L(L))
    if not rows:
        print("  (no variants found)")
        return

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=paperfig.FIG["duo"], constrained_layout=True,
        gridspec_kw={"width_ratios": [1.6, 1.0]})

    # Panel A: peak (bars) + mean (markers) per variant
    x = np.arange(len(rows))
    peaks = [r[3] for r in rows]
    means = [r[4] for r in rows]
    cols = [r[2] for r in rows]
    axA.bar(x, peaks, width=0.66, color=cols, edgecolor="white", linewidth=0.6)
    axA.scatter(x, means, color="#111111", s=20, zorder=5, label="mean live")
    for xi, p in zip(x, peaks):
        axA.annotate(str(p), (xi, p), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=7.5)
    axA.axhline(XMM, ls="--", color="0.4", lw=1.0)
    axA.text(len(rows) - 0.4, XMM + 6, f"{XMM} regs", color="0.35", fontsize=8, ha="right")
    axA.set_xticks(x)
    axA.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=8)
    axA.set_ylabel("live values")
    axA.legend(loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)

    # Panel B: depth knob -- peak & mean live vs number of layers L
    dk = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1])
    if dk:
        Ls = [r[1] for r in dk]
        axB.plot(Ls, [r[3] for r in dk], "-o", color=C_CASC, lw=1.6, ms=5, label="peak live")
        axB.plot(Ls, [r[4] for r in dk], "-s", color="#8aa9c2", lw=1.4, ms=4, label="mean live")
        prod_peak = next((r[3] for r in rows if r[0] == "production"), None)
        if prod_peak:
            axB.axhline(prod_peak, ls=":", color=C_PROD, lw=1.2)
            axB.text(Ls[-1], prod_peak - 8, f"production peak {prod_peak}",
                     color=C_PROD, fontsize=7.5, ha="right", va="top")
    axB.set_xlabel("cascade depth L")
    axB.set_ylabel("live values")
    axB.legend(loc="center right")
    axB.spines[["top", "right"]].set_visible(False)

    _save(fig, out_base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def figure_object_sweep(out_base, spec):
    """Measured stack spills per cascade depth L vs the production reference. Reads each L's
    compiled symbol (BSSN: one multi-symbol .o; EMDA: one .o per L)."""
    import numpy as np
    plt = _style()
    cm = _load_cm()
    cache = {}
    rows = []
    for L in spec.Ls:
        o = spec.obj_L(L)
        if not os.path.exists(o):
            print(f"  (skip) L{L}: {os.path.relpath(o, VIKR)} not found")
            continue
        if o not in cache:
            cache[o] = cm._disasm_by_symbol(o)
        funcs = cache[o]
        hint = spec.sym_L(L)
        cands = [s for s in funcs if hint in s]
        if not cands:
            print(f"  (skip) L{L}: no symbol {hint}")
            continue
        sym = max(cands, key=lambda s: len(funcs[s]))
        sp = sum(spill_flags(funcs[sym]))
        rows.append((L, sp))
        print(f"  L{L}: measured spills={sp}")
    if not rows:
        return
    prod_sp = (sum(spill_flags(hot_symbol(spec.prod_o, spec.prod_hint)[1]))
               if os.path.exists(spec.prod_o) else None)
    fig, ax = plt.subplots(figsize=paperfig.FIG["short"], constrained_layout=True)
    ys = [r[1] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, ys, width=0.6, color=C_CASC, edgecolor="white", linewidth=0.6)
    for xi, sp in zip(x, ys):
        ax.annotate(str(sp), (xi, sp), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    if prod_sp:
        ax.axhline(prod_sp, ls=":", color=C_PROD, lw=1.4)
        ax.text(len(rows) - 1, prod_sp - 35, f"production {prod_sp}",
                color=C_PROD, ha="right", va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{r[0]}" for r in rows])
    ax.set_ylim(0, (prod_sp or max(ys)) * 1.12)
    ax.set_xlabel("cascade depth L")
    ax.set_ylabel("measured stack spills")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_base)


def figure_depth_overlay(out_base, spec, Ls=None):
    """Overlay the cascade liveness curve at each depth L on a CASCADE-scaled axis, so the
    depth-knob effect is visible (production would otherwise dominate the y-scale)."""
    import numpy as np
    plt = _style()
    Ls = Ls or spec.Ls
    shades = ["#c98a3a", "#a7c6de", "#6e9ec4", "#30638e", "#1f3f5b", "#7a4fa3"]
    styles = paperfig.lss_thin   # vendored monotone solid->sparse progression (ordered knob)
    fig, ax = plt.subplots(figsize=paperfig.FIG["short"], constrained_layout=True)
    prod_peak = source_liveness(parse_kernel(spec.prod_cpp)[0])["peak"]
    maxpeak = 0
    for i, L in enumerate(Ls):
        cpp = spec.cpp_L(L)
        if not os.path.exists(cpp):
            continue
        liv = source_liveness(parse_kernel(cpp)[0])
        tr = liv["trace"]
        x = np.arange(len(tr)) / max(1, len(tr) - 1)
        ax.plot(x, tr, color=shades[i % len(shades)], lw=1.5, alpha=0.9,
                ls=styles[i % len(styles)], label=f"L{L}  (peak {liv['peak']})")
        maxpeak = max(maxpeak, liv["peak"])
        print(f"  L{L}: peak={liv['peak']}  temps={liv['n_temps']}")
    ax.axhline(XMM, ls="--", color="0.4", lw=1.0)
    ax.text(0.004, XMM + 2, f"{XMM} regs", color="0.35", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, maxpeak * 1.18 if maxpeak else 1)
    ax.text(0.99, maxpeak * 1.07, f"production peak {prod_peak} (off scale)",
            ha="right", color=C_PROD, fontsize=8)
    ax.set_xlabel("kernel progress")
    ax.set_ylabel("live values")
    ax.legend(title="cascade depth", loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_base)


def _do_source(prod_cpp, casc_cpp, out, name, casc_label):
    print(f"[source] parsing C++ kernels -> {name}:")
    prod = parse_kernel(prod_cpp)[0]
    casc, _, casc_labels = parse_kernel(casc_cpp)
    print(f"  production {os.path.relpath(prod_cpp, VIKR)}: {len(prod)} statements")
    print(f"  {casc_label:<14} {os.path.relpath(casc_cpp, VIKR)}: {len(casc)} statements")
    figure_source(prod, casc, os.path.join(out, name), casc_label, casc_labels)


def _do_object(prod_o, casc_o, out, prod_hint="bssn_production_rhs",
               casc_hint="bssn_cascade_rhs", name="liveness_object"):
    print("[object] disassembling scalar object files:")
    prod = object_spills(prod_o, prod_hint, "production")
    casc = object_spills(casc_o, casc_hint, "cascade")
    figure_object(prod, casc, os.path.join(out, name))


def _do_all(spec, out):
    """Every figure for one system: per-L source + depth overlay + object + object-sweep + sweep."""
    for L in spec.Ls:
        cpp = spec.cpp_L(L)
        if os.path.exists(cpp):
            _do_source(spec.prod_cpp, cpp, out, f"{spec.prefix}_source_L{L}", f"{spec.casc_label} L{L}")
    print("[depth] cascade liveness overlay across L:")
    figure_depth_overlay(os.path.join(out, f"{spec.prefix}_depth"), spec)
    if os.path.exists(spec.prod_o) and os.path.exists(spec.casc_o):
        _do_object(spec.prod_o, spec.casc_o, out, spec.prod_hint, spec.casc_hint, f"{spec.prefix}_object")
        print("[obj-sweep] measured spills per depth L:")
        figure_object_sweep(os.path.join(out, f"{spec.prefix}_object_sweep"), spec)
    else:
        print(f"[object] skipped: {spec.name} .o not found (build the harness)")
    print("[sweep] liveness across variants:")
    figure_sweep(os.path.join(out, f"{spec.prefix}_sweep"), spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("source", help="parse any .cpp kernels -> a source-liveness figure")
    s.add_argument("--prod", default=DEF_PROD_CPP, help="production-style kernel .cpp")
    s.add_argument("--cascade", default=DEF_CASC_CPP, help="cascade-style kernel .cpp (any variant)")
    s.add_argument("--label", default="cascade", help="legend label for the cascade variant")
    s.add_argument("--name", default="liveness_source", help="output basename")
    s.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    o = sub.add_parser("object", help="disassemble any scalar .o -> a spill figure")
    o.add_argument("--prod-o", default=DEF_PROD_O)
    o.add_argument("--cascade-o", default=DEF_CASC_O)
    o.add_argument("--prod-hint", default="bssn_production_rhs", help="production symbol substring")
    o.add_argument("--cascade-hint", default="bssn_cascade_rhs", help="cascade symbol substring")
    o.add_argument("--name", default="liveness_object", help="output basename")
    o.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    sw = sub.add_parser("sweep", help="liveness across the depth knob -> *_sweep")
    sw.add_argument("--system", default="bssn", help="bssn | emda")
    sw.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    dp = sub.add_parser("depth", help="overlay cascade liveness at several L -> *_depth")
    dp.add_argument("--system", default="bssn", help="bssn | emda")
    dp.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    osw = sub.add_parser("objsweep", help="measured spills per depth L -> *_object_sweep")
    osw.add_argument("--system", default="bssn", help="bssn | emda")
    osw.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    a = sub.add_parser("all", help="every figure for a system (bssn or emda)")
    a.add_argument("--system", default="bssn", help="bssn | emda")
    a.add_argument("--out", default=os.path.join(VIKR, "paper/figures"))

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.cmd == "source":
        _do_source(args.prod, args.cascade, args.out, args.name, args.label)
    elif args.cmd == "object":
        _do_object(args.prod_o, args.cascade_o, args.out,
                   args.prod_hint, args.cascade_hint, args.name)
    elif args.cmd == "sweep":
        spec = system_spec(args.system)
        print("[sweep] liveness across variants:")
        figure_sweep(os.path.join(args.out, f"{spec.prefix}_sweep"), spec)
    elif args.cmd == "depth":
        spec = system_spec(args.system)
        print("[depth] cascade liveness overlay across L:")
        figure_depth_overlay(os.path.join(args.out, f"{spec.prefix}_depth"), spec)
    elif args.cmd == "objsweep":
        spec = system_spec(args.system)
        print("[obj-sweep] measured spills per depth L:")
        figure_object_sweep(os.path.join(args.out, f"{spec.prefix}_object_sweep"), spec)
    elif args.cmd == "all":
        _do_all(system_spec(args.system), args.out)


if __name__ == "__main__":
    main()
