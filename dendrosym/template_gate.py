"""Guard-mismatch gate for the Jinja2 solver templates.

The bug class: a symbol whose *definition* sits inside a feature guard while a
*use* of it sits outside, or inside a different guard. The generated solver then
fails to compile for anyone who builds with that flag off -- and nobody notices
until someone actually does. Three instances have shipped:

  AMR_R_RATIO         defined under enable_bh_tracking, printed unguarded
  computeWTolDCoords  defined under enable_tpid, called unconditionally
  BH_LOC              defined under enable_bh_tracking, assigned under enable_tpid

All three were found by a human happening to build with the wrong flag. This
renders every template across the flag cross-product and reports any symbol that
is still referenced in a combination where its definition disappeared -- which
takes seconds and needs no toolchain.

It is a *render* gate, not a compile gate: it sees only what Jinja emits. In
particular every ``{% if X is defined %}`` block renders as absent under the stub
context, so nothing gencode emits is covered here. That is a real blind spot, and
it is stated rather than papered over: passing this gate means the guards line
up, not that the solver builds.

Usage:
    python -m dendrosym.template_gate            # report, exit 1 on findings
    python -m dendrosym.template_gate --verbose  # also list every combination
"""

import itertools
import re
import sys

import jinja2

from dendrosym.project_generator import _TEMPLATES_DIR, build_template_map

# The feature flags that gate template blocks. Kept explicit rather than scraped
# from the templates so that adding a flag is a deliberate act -- a scraped list
# would silently stop covering a flag someone renamed.
GUARD_FLAGS = (
    "enable_tpid",
    "enable_bh_tracking",
    "enable_analytical",
    "enable_gw_extraction",
    "enable_ah",
    "enable_profiling",
    "use_dendro_derivs",
)

# C++ keywords that can open a line and be followed by `name(`, which would
# otherwise read as a function definition. Missing these masks real findings:
# `return computeWTolDCoords(...)` parsed as *defining* computeWTolDCoords is
# what hid one of the three known bugs from an earlier version of this gate.
_STMT_KEYWORDS = frozenset("""
return if else for while switch case do goto throw new delete sizeof
""".split())

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")

# a definition: optional leading qualifiers/type, then NAME, then =, [, ; or (
_DEF_RE = re.compile(
    r"^\s*(?:extern\s+|static\s+|const\s+|inline\s+|constexpr\s+|struct\s+|"
    r"unsigned\s+|signed\s+|template\s*<[^>]*>\s*)*"
    r"(?:[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*\s*[*&]?\s+)"
    r"([A-Za-z_]\w*)\s*(?=[\[=;(])",
    re.M,
)
_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)", re.M)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class _StubUndefined(jinja2.ChainableUndefined):
    """An undefined that survives being iterated, compared and printed.

    The gate renders with a stub context -- no config, no gencode -- so every
    non-flag value is undefined. ChainableUndefined alone raises on `range(x)`
    and `x > 0`, which stops the render before it reaches the code being gated.
    """

    __bool__ = lambda s: False           # noqa: E731
    __iter__ = lambda s: iter(())        # noqa: E731
    __len__ = lambda s: 0                # noqa: E731
    __int__ = lambda s: 0                # noqa: E731
    __index__ = lambda s: 0              # noqa: E731
    __str__ = lambda s: ""               # noqa: E731
    __eq__ = lambda s, o: False          # noqa: E731
    __ne__ = lambda s, o: True           # noqa: E731
    __lt__ = lambda s, o: False          # noqa: E731
    __le__ = lambda s, o: False          # noqa: E731
    __gt__ = lambda s, o: False          # noqa: E731
    __ge__ = lambda s, o: False          # noqa: E731
    __hash__ = lambda s: 0               # noqa: E731


def _decomment(src):
    return _STRING_RE.sub('""', _COMMENT_RE.sub(" ", src))


def _defined(src):
    out = set(_DEFINE_RE.findall(src))
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = _IDENT_RE.match(stripped)
        if first and first.group(0) in _STMT_KEYWORDS:
            continue
        m = _DEF_RE.match(line)
        if m:
            out.add(m.group(1))
    return out


def _used(src):
    return set(_IDENT_RE.findall(src))


def _called(src):
    return set(_CALL_RE.findall(src))


def _render_all(flags, project_name="solver"):
    """Render every mapped template under one flag assignment; return one blob."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=_StubUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    ctx = dict(flags)
    ctx.update(
        project_name=project_name,
        project_upper=project_name.upper(),
        namespace=project_name,
    )
    parts = []
    for out_rel, tmpl_name in build_template_map(ctx).items():
        if not (_TEMPLATES_DIR / tmpl_name).exists():
            continue
        if not out_rel.endswith((".h", ".cpp")):
            continue  # C++ only: the toml/markdown templates have no symbols
        parts.append(env.get_template(tmpl_name).render(**ctx))
    return "\n".join(parts)


def scan(strict=True, project_name="solver"):
    """Render the flag cross-product; return {symbol: finding} for mismatches.

    A finding is a symbol that some combination *uses* while its definition is
    absent in that same combination but present in another -- i.e. a definition
    hidden behind a guard its use does not share.

    `strict` keeps only project-prefixed / ALL_CAPS globals and names used as a
    call. Without it the report is dominated by locals and parameters whose names
    happen to collide with a guard-local symbol elsewhere: on the known-good tree
    that is 9 findings for 1 real, against 1 for 1 with the filter on.
    """
    combos = list(itertools.product([True, False], repeat=len(GUARD_FLAGS)))
    rendered, defined, used, called = {}, {}, {}, {}
    for combo in combos:
        flags = dict(zip(GUARD_FLAGS, combo))
        key = tuple(sorted(k for k, v in flags.items() if v))
        blob = _decomment(_render_all(flags, project_name))
        rendered[key] = flags
        defined[key] = _defined(blob)
        used[key] = _used(blob)
        called[key] = _called(blob)

    defined_anywhere = set().union(*defined.values())
    prefix = project_name.upper() + "_"

    findings = {}
    for key, flags in rendered.items():
        missing = (used[key] & defined_anywhere) - defined[key]
        for sym in missing:
            if strict and not (
                sym.startswith(prefix)
                or _ALLCAPS_RE.match(sym)
                or sym in called[key]
            ):
                continue
            needed = sorted(
                f for f in GUARD_FLAGS
                if all(sym in defined[k] for k in defined if f in k)
                and not any(sym in defined[k] for k in defined if f not in k)
            )
            rec = findings.setdefault(
                sym, {"symbol": sym, "combos": 0, "needs": needed}
            )
            rec["combos"] += 1
    return findings


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    strict = "--no-strict" not in argv
    findings = scan(strict=strict)
    total = 2 ** len(GUARD_FLAGS)
    if not findings:
        print(f"template gate: OK -- {total} flag combinations, no guard mismatches")
        return 0
    print(f"template gate: {len(findings)} guard mismatch(es) over {total} combinations\n")
    for rec in sorted(findings.values(), key=lambda r: -r["combos"]):
        needs = ", ".join(rec["needs"]) or "<unclear>"
        print(f"  {rec['symbol']:<28} used but undefined in {rec['combos']}/{total} "
              f"combinations; defined only under: {needs}")
    print("\nA definition guarded by one flag is being used under another. "
          "Guard the use to match, or hoist the definition.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
