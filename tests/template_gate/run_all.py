#!/usr/bin/env python
"""run_all.py -- the template guard-mismatch gate.

    python tests/template_gate/run_all.py

Renders every solver template across the feature-flag cross-product and fails on
any symbol used where its definition is guarded away. Plain asserts, no pytest.

Two checks, deliberately. "The tree is clean" alone would also pass if the gate
stopped detecting anything at all -- that is how a check quietly becomes a no-op
(see the skip_gencode staleness check that once passed by never running). The
second check injects the bug shape and asserts it is caught.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jinja2  # noqa: E402

from dendrosym import template_gate  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAILURES.append(name)
        print(f"FAIL  {name}\n      {exc}")
    else:
        print(f"PASS  {name}")


def test_tree_is_clean():
    findings = template_gate.scan()
    assert not findings, (
        "guarded definitions used outside their guard: " + ", ".join(sorted(findings))
    )


def test_gate_still_bites(tmpdir):
    """Prove the gate fires by making the thing it guards against happen."""
    gr = os.path.join(tmpdir, "gr")
    os.makedirs(gr, exist_ok=True)
    with open(os.path.join(gr, "x.h.j2"), "w") as fh:
        fh.write("{% if enable_tpid %}\nextern double {{ project_upper }}_WIDGET;\n"
                 "{% endif %}\n")
    with open(os.path.join(gr, "x.cpp.j2"), "w") as fh:
        fh.write("void f() { {{ project_upper }}_WIDGET = 1.0; }\n")

    orig_dir = template_gate._TEMPLATES_DIR
    orig_map = template_gate.build_template_map
    try:
        template_gate._TEMPLATES_DIR = __import__("pathlib").Path(tmpdir)
        template_gate.build_template_map = lambda ctx: {
            "solver/x.h": "gr/x.h.j2", "solver/x.cpp": "gr/x.cpp.j2"}
        findings = template_gate.scan()
    finally:
        template_gate._TEMPLATES_DIR = orig_dir
        template_gate.build_template_map = orig_map

    assert "SOLVER_WIDGET" in findings, f"gate missed the injected bug: {findings}"
    assert findings["SOLVER_WIDGET"]["needs"] == ["enable_tpid"], findings


if __name__ == "__main__":
    import tempfile

    check("templates have no guard mismatch", test_tree_is_clean)
    with tempfile.TemporaryDirectory() as td:
        check("gate detects an injected mismatch", lambda: test_gate_still_bites(td))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all template-gate checks passed")
