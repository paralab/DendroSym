"""Generate and (optionally) compile the two poles of the feature-flag matrix.

The template gate renders; it never compiles. Every `{% if X is defined %}`
block is absent under its stub context, so nothing gencode emits is covered and
a missing header or a renamed dendrolib symbol passes it cleanly. This closes
that gap on two configs:

  smoke_min  every feature flag off   -- a solver with no GR capabilities
  smoke_max  every feature flag on    -- TPID, tracking, GW, AH, analytic, profiling

Two bugs were already sitting at HEAD when this was written: `enable_ah`
defaulted on, so a scalar-wave solver linked BHaHAHA; and solver_ctx included
`gwExtract.h`, which dendrolib had renamed to `gw_extract.h`, so every solver
with GW extraction failed to compile.

Usage:
    python tests/build_smoke/run_all.py                 # generate only (~1 min)
    python tests/build_smoke/run_all.py --build         # also compile
    python tests/build_smoke/run_all.py --build --dendrolib PATH

--build needs a dendrolib with DendroDerivatives (`include/derivatives.h`).
The default CMake pin, upstream Dendro-5.01 master, does not have it, so pass
--dendrolib or set DENDRO_LIB_DIR. Compiling dendrolib itself takes ~20 min the
first time; the build directory is reused after that.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
POLES = ("smoke_min", "smoke_max")


def _run(cmd, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        rc = subprocess.call(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    return rc


def _errors(log, limit=6):
    if not log.exists():
        return ""
    bad = [ln for ln in log.read_text(errors="replace").splitlines()
           if "error:" in ln or "Error " in ln]
    return "\n".join("      " + ln for ln in bad[:limit])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="also cmake-configure and compile each pole")
    ap.add_argument("--dendrolib", default=os.environ.get("DENDRO_LIB_DIR", ""),
                    help="local dendrolib source (needs include/derivatives.h)")
    ap.add_argument("--work", default="",
                    help="scratch directory (default: tests/build_smoke/_work)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch directory instead of reusing clean")
    ns = ap.parse_args(sys.argv[1:] if argv is None else argv)

    work = Path(ns.work) if ns.work else HERE / "_work"
    if work.exists() and not ns.keep and not ns.build:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

    failures = []
    for pole in POLES:
        cfg = HERE / "configs" / f"{pole}.py"
        log = work / f"{pole}.generate.log"
        rc = subprocess.call([sys.executable, str(cfg)], cwd=work, env=env,
                             stdout=log.open("w"), stderr=subprocess.STDOUT)
        name = pole.replace("_", "") + "-solver"
        out = work / "output" / name
        if rc != 0 or not (out / "CMakeLists.txt").exists():
            print(f"FAIL  {pole} did not generate ({log})")
            failures.append(pole)
            continue
        print(f"PASS  {pole} generated")

        if not ns.build:
            continue

        if not ns.dendrolib:
            print(f"SKIP  {pole} build -- pass --dendrolib PATH "
                  "(the default pin has no derivatives.h)")
            continue

        cfg_log = work / f"{pole}.cmake.log"
        rc = _run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                   f"-DDENDRO_dendrolib_DIR={ns.dendrolib}"], out, cfg_log)
        if rc != 0:
            print(f"FAIL  {pole} cmake configure ({cfg_log})")
            print(_errors(cfg_log))
            failures.append(pole)
            continue

        target = name.replace("-solver", "") + "Solver"
        build_log = work / f"{pole}.build.log"
        rc = _run(["cmake", "--build", "build", "-j",
                   str(os.cpu_count() or 4), "--target", target],
                  out, build_log)
        if rc != 0 or not (out / "build" / "solver" / target).exists():
            print(f"FAIL  {pole} compile ({build_log})")
            print(_errors(build_log))
            failures.append(pole)
            continue
        print(f"PASS  {pole} compiled -> {target}")

    if failures:
        print(f"\n{len(failures)} pole(s) failed: {', '.join(failures)}")
        return 1
    print("\nall build-smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
