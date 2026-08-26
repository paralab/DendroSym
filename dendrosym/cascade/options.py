"""CascadeOptions -- every knob of the cascade pipeline in one dataclass.

The vikr research CLIs spread these over three argparse blocks
(cascade_emit.py / emda_cascade.py / cascade_autolayer.py). This dataclass is
the single source of truth for the *system-agnostic* knobs; it maps 1:1 onto
CLI flags (``add_argparse_args`` / ``from_namespace`` / ``to_cli_args``) so the
config-script API and ``python ccz4.py --cascade-...`` never drift.

System-specific knobs (BSSN gauge/ssl/cahd/eta_mode, EMDA with_matter/hoist_exp)
live with their spec modules, not here.

Defaults are the *clean reproducibility baseline* (scalar, natural depth), the
same defaults the vikr codegen chose on purpose; the paper's deployed config is
opt-in (simd="avx2"|"avx512", L=7, fused=True for BSSN with ssl+cahd).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Optional

SIMD_CHOICES = ("scalar", "avx2", "avx512")
SPLIT_MODES = ("auto", "smart", "dumb")
EMIT_STYLES = ("flat", "tensor-loop")
_LANES = {"scalar": 1, "avx2": 4, "avx512": 8}


@dataclass(frozen=True)
class CascadeOptions:
    # --- build (IR) -----------------------------------------------------------
    L: Optional[int] = None        # layer count; None = as declared (or DP-chosen with auto)
    auto: bool = False             # exact-DP layer boundaries from the declared order
    search_order: bool = False     # bottleneck search over linear extensions (needs auto)
    split_mode: str = "auto"       # auto | smart | dumb  (only for L > natural)
    cse_prefix: str = "CASC_"
    # --- emit -----------------------------------------------------------------
    simd: str = "scalar"           # scalar | avx2 | avx512
    emit_style: str = "flat"       # flat | tensor-loop (tensor-loop needs a NamingDialect)
    inline_threshold: int = 2      # inline CSE temps used <= N times (0 = off; scalar kernels use 0)
    fma_tree: bool = True          # tree-level VFMA/VFNMADD (SIMD only)
    fma_split: int = 1             # split long sums into k FMA chains (SIMD only; implies fma_tree in CLI)
    vfma: int = 0                  # legacy regex VADD(VMUL)->VFMA fold passes (superseded by fma_tree)
    fused: bool = False            # inline 6th-order stencils for 1st/pure-2nd derivs (SIMD only)
    global_cse: bool = False       # one symbol-aware CSE across chunks instead of per-chunk
    lazy_prologue: bool = True     # move leaf VLOADs to first use (SIMD only)
    # --- output / misc --------------------------------------------------------
    fn_name: str = "cascade_kernel"
    out: Optional[str] = None
    verbose: bool = False
    enabled: bool = True           # dendro-bridge: registered but inert when False

    # ------------------------------------------------------------------ helpers
    def __post_init__(self):
        if self.simd not in SIMD_CHOICES:
            raise ValueError(f"simd must be one of {SIMD_CHOICES}, got {self.simd!r}")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {self.split_mode!r}")
        if self.emit_style not in EMIT_STYLES:
            raise ValueError(f"emit_style must be one of {EMIT_STYLES}, got {self.emit_style!r}")
        if self.fused and self.simd == "scalar":
            raise ValueError("fused=True needs a SIMD emitter (simd='avx2'|'avx512'); "
                             "the scalar path reads precomputed derivative arrays only")

    @property
    def width(self) -> int:
        """Lane width of the chosen SIMD dialect (1 for scalar)."""
        return _LANES[self.simd]

    @property
    def smart_split(self):
        """build_cascade_ir's tri-state: None=auto, True=smart, False=dumb."""
        return {"auto": None, "smart": True, "dumb": False}[self.split_mode]

    def replace(self, **kw) -> "CascadeOptions":
        return dataclasses.replace(self, **kw)

    def cache_key(self) -> str:
        """Stable string for gencode cache discrimination (excludes out/verbose)."""
        import sympy
        skip = {"out", "verbose"}
        parts = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.name not in skip]
        return "cascade|" + "|".join(parts) + f"|sympy={sympy.__version__}"

    # ----------------------------------------------------------------- argparse
    @classmethod
    def add_argparse_args(cls, parser, prefix: str = "") -> None:
        """Add one flag per field: ``--{prefix}L``, ``--{prefix}simd`` ...

        Booleans get ``--{prefix}name / --{prefix}no-name`` pairs so a config
        script's default can be overridden either way from the command line.
        Flags default to None (= "not given") so ``from_namespace`` can layer
        them over an existing CascadeOptions instance.
        """
        p = prefix
        parser.add_argument(f"--{p}L", type=int, default=None, dest=f"{p}L", metavar="N",
                            help="layer count (None = as declared, or DP-chosen with --auto)")
        parser.add_argument(f"--{p}auto", action="store_true", default=None, dest=f"{p}auto",
                            help="exact-DP layer boundaries from the declared order")
        parser.add_argument(f"--{p}search-order", action="store_true", default=None,
                            dest=f"{p}search_order", help="bottleneck search over linear extensions")
        parser.add_argument(f"--{p}split-mode", choices=SPLIT_MODES, default=None, dest=f"{p}split_mode")
        parser.add_argument(f"--{p}cse-prefix", default=None, dest=f"{p}cse_prefix")
        parser.add_argument(f"--{p}simd", choices=SIMD_CHOICES, default=None, dest=f"{p}simd")
        parser.add_argument(f"--{p}emit-style", choices=EMIT_STYLES, default=None, dest=f"{p}emit_style")
        parser.add_argument(f"--{p}inline-threshold", type=int, default=None, dest=f"{p}inline_threshold",
                            metavar="N", help="inline CSE temps used <= N times (0 = off)")
        parser.add_argument(f"--{p}fma-tree", dest=f"{p}fma_tree", action="store_true", default=None)
        parser.add_argument(f"--{p}no-fma-tree", dest=f"{p}fma_tree", action="store_false")
        parser.add_argument(f"--{p}fma-split", type=int, default=None, dest=f"{p}fma_split", metavar="K")
        parser.add_argument(f"--{p}vfma", type=int, default=None, dest=f"{p}vfma", metavar="N",
                            help="legacy regex VFMA fold passes")
        parser.add_argument(f"--{p}fused", action="store_true", default=None, dest=f"{p}fused")
        parser.add_argument(f"--{p}global-cse", action="store_true", default=None, dest=f"{p}global_cse")
        parser.add_argument(f"--{p}no-lazy-prologue", dest=f"{p}lazy_prologue", action="store_false",
                            default=None)
        parser.add_argument(f"--{p}fn-name", default=None, dest=f"{p}fn_name")
        parser.add_argument(f"--{p}out", "-o" if not prefix else f"--{p}o", default=None, dest=f"{p}out")
        parser.add_argument(f"--{p}verbose", action="store_true", default=None, dest=f"{p}verbose")

    @classmethod
    def from_namespace(cls, ns, prefix: str = "", base: "CascadeOptions" = None) -> "CascadeOptions":
        """Overlay the flags that were actually given onto ``base`` (or defaults).

        Mirrors the vikr CLI rule ``fma_tree = fma_tree or fma_split > 1``.
        """
        base = base if base is not None else cls()
        kw = {}
        for f in fields(cls):
            v = getattr(ns, f"{prefix}{f.name}", None)
            if v is not None:
                kw[f.name] = v
        if kw.get("fma_split", base.fma_split) > 1:
            kw["fma_tree"] = True
        return base.replace(**kw)

    def to_cli_args(self, prefix: str = "") -> list:
        """The flag list that reproduces this instance (inverse of from_namespace)."""
        d = self.__class__()
        p = prefix
        args = []
        for f in fields(self):
            if f.name in ("enabled",):
                continue
            v, dv = getattr(self, f.name), getattr(d, f.name)
            if v == dv:
                continue
            flag = f"--{p}{f.name.replace('_', '-')}"
            if isinstance(v, bool):
                args.append(flag if v else flag.replace(f"--{p}", f"--{p}no-"))
            else:
                args += [flag, str(v)]
        return args
