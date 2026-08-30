"""Compare two solver configs numerically: ``python -m dendrosym.numgate_cli``.

    python -m dendrosym.numgate_cli OLD.py NEW.py --var-type evolution

Each config is evaluated in its **own subprocess**, which is not an
implementation detail: a config calls ``dendrosym.nr.set_metric`` at import, so
importing two of them into one interpreter silently gives the second one the
first one's metric. Values cross the process boundary as decimal strings, which
also keeps the comparison free of any sympy-version pickling concern.

Use it to gate a config refactor that is *not* meant to be byte-identical --
index-notation rewrites, symmetric-matrix helpers, staging, regrouped algebra.
For a refactor that IS meant to be a no-op, the byte-identical regen is stricter
and cheaper; use that instead.
"""

import argparse
import json
import os
import subprocess
import sys

_WORKER = r"""
import json, os, sys, importlib.util
sys.path.insert(0, {sympygr!r})
cfg_path = {cfg!r}
spec = importlib.util.spec_from_file_location("_numgate_cfg", cfg_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["_numgate_cfg"] = mod
sys.argv = [cfg_path]
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass

config = None
for name in ("dendroConfigs", "config", "configs"):
    if hasattr(mod, name):
        config = getattr(mod, name)
        break
if config is None:
    raise SystemExit("no config object found (looked for dendroConfigs/config/configs)")

from dendrosym import numgate

names, exprs, staged_names, staged_exprs = [], [], [], []
all_exp, all_names, st_exp, st_names, _ = config._extract_rhs_expressions({vt!r})
names = [str(n) for n in all_names]
exprs = list(all_exp)
staged_names = [str(n) for n in st_names]
staged_exprs = list(st_exp)

vals = numgate.evaluate_block(
    names, exprs, staged_names, staged_exprs, seed={seed}, dps={dps}
)
sys.stdout.write("@@NUMGATE@@" + json.dumps({{k: str(v) for k, v in vals.items()}}))
"""


def _run_side(cfg_path, var_type, seed, dps, sympygr_root):
    src = _WORKER.format(
        sympygr=sympygr_root, cfg=os.path.abspath(cfg_path),
        vt=var_type, seed=seed, dps=dps,
    )
    proc = subprocess.run(
        [sys.executable, "-c", src],
        cwd=os.path.dirname(os.path.abspath(cfg_path)) or ".",
        capture_output=True, text=True,
    )
    if "@@NUMGATE@@" not in proc.stdout:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"worker for {cfg_path} produced no result")
    payload = proc.stdout.split("@@NUMGATE@@", 1)[1]
    return json.loads(payload)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dendrosym.numgate_cli")
    ap.add_argument("config_a")
    ap.add_argument("config_b")
    ap.add_argument("--var-type", default="evolution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--tol", default="1e-40")
    ap.add_argument("--show", type=int, default=10,
                    help="rows to print (worst first); 0 for all")
    ns = ap.parse_args(sys.argv[1:] if argv is None else argv)

    import mpmath

    from dendrosym import numgate

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"evaluating {ns.config_a} ...", file=sys.stderr)
    a = _run_side(ns.config_a, ns.var_type, ns.seed, ns.dps, root)
    print(f"evaluating {ns.config_b} ...", file=sys.stderr)
    b = _run_side(ns.config_b, ns.var_type, ns.seed, ns.dps, root)

    mpmath.mp.dps = ns.dps

    def _num(v):
        return mpmath.mpc(v) if "j" in v else mpmath.mpf(v)

    a = {k: _num(v) for k, v in a.items()}
    b = {k: _num(v) for k, v in b.items()}
    ok, rows = numgate.compare(a, b, tol=mpmath.mpf(ns.tol))

    print(f"\n{len(rows)} outputs compared, tol {ns.tol}, dps {ns.dps}, seed {ns.seed}")
    print(numgate.format_report(rows, limit=None if ns.show == 0 else ns.show))
    worst = rows[0][3] if rows else 0
    if ok:
        print(f"\nAGREE -- worst relative difference {mpmath.nstr(worst, 6)}")
        return 0
    print(f"\nDISAGREE -- worst relative difference {mpmath.nstr(worst, 6)}")
    print("The two configs do not compute the same physics. The rows above are "
          "sorted worst-first; start with the top one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
