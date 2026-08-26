"""``dendro-cascade`` / ``python -m dendrosym.cascade`` -- the cascade CLIs.

Subcommands map 1:1 onto the vikr research scripts so their invocations
(scripts/regen_vikr_kernels.sh) translate mechanically:

  bssn          cascade_emit.py main()      (BSSN kernels: --simd --L --ssl --cahd ...)
  bssn-looped   bssn_looped.py              (structured looped BSSN emitter)
  emda          emda_cascade.py             (EMDA kernels; needs ~/research/emda-gr)
  autolayer     cascade_autolayer.py        (DP layering analysis, no code)
  order-check   cascade_order_check.py
  metrics       cascade_metrics.py          (objdump spill/FP counts, algebraic counts)
  compile       generic: a user module exposing spec() -> (chunks, leaves)
                plus every CascadeOptions flag
"""
import importlib
import importlib.util
import os
import sys


def _delegate(mod_name, fn, prog, rest, argv_style):
    mod = importlib.import_module(mod_name)
    main = getattr(mod, fn)
    if argv_style:
        return main(rest)
    saved = sys.argv
    try:
        sys.argv = [prog] + rest
        return main()
    finally:
        sys.argv = saved


def _compile(rest):
    import argparse
    from dendrosym.cascade.api import compile_system, report, warn_if_unpinned
    from dendrosym.cascade.options import CascadeOptions
    ap = argparse.ArgumentParser(prog="dendro-cascade compile",
                                 description="compile a user spec module to a kernel")
    ap.add_argument("specfile", help="python file (or dotted module) exposing spec() -> (chunks, leaves)")
    ap.add_argument("--spec-fn", default="spec", help="name of the spec function (default: spec)")
    ap.add_argument("--report", action="store_true", help="print the layer table")
    CascadeOptions.add_argparse_args(ap)
    ns = ap.parse_args(rest)
    opts = CascadeOptions.from_namespace(ns)
    if os.path.exists(ns.specfile):
        spec = importlib.util.spec_from_file_location("_cascade_user_spec", ns.specfile)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(ns.specfile)
    chunks, leaves = getattr(mod, ns.spec_fn)()
    warn_if_unpinned()
    code, ir = compile_system(chunks, leaves, options=opts)
    if ns.report:
        report(ir)
    if not opts.out:
        sys.stdout.write(code)
    return 0


SUBCOMMANDS = {
    "bssn": ("dendrosym.cascade.systems.bssn.emit", "main", True),
    "bssn-looped": ("dendrosym.cascade.systems.bssn.looped", "main", False),
    "emda": ("dendrosym.cascade.systems.emda.cascade", "_main", True),
    "autolayer": ("dendrosym.cascade.autolayer", "main", False),
    "order-check": ("dendrosym.cascade.order_check", "main", False),
    "metrics": ("dendrosym.cascade.metrics", "main", True),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "compile":
        return _compile(rest)
    if sub not in SUBCOMMANDS:
        print(f"dendro-cascade: unknown subcommand {sub!r}\n{__doc__}", file=sys.stderr)
        return 2
    mod, fn, argv_style = SUBCOMMANDS[sub]
    rc = _delegate(mod, fn, f"dendro-cascade {sub}", rest, argv_style)
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
