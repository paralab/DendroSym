"""project_generator.py

Generates a complete Dendro solver project from an NRConfig (or DendroConfiguration).

Instead of using cog to embed Python inside C++ files, this module runs all symbolic
computation once and then renders Jinja2 templates to produce a complete, buildable
C++ project.

Usage:
    from dendrosym.project_generator import DendroProjectGenerator
    from emda_configs import dendroConfigs

    gen = DendroProjectGenerator(dendroConfigs)
    gen.generate("./output/emda-solver/")
"""

import hashlib
import json
import sys
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import dendrosym


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _rewrite_deriv_calls(code: str, deriv_obj: str) -> str:
    """Rewrite legacy derivative calls to use a DendroDerivatives object.

    Transforms:
        deriv_x(out, in, hx, sz, bflag)   ->  DERIVS->grad_x(out, in, hx, sz, bflag)
        deriv_xx(out, in, hx, sz, bflag)   ->  DERIVS->grad_xx(out, in, hx, sz, bflag)
        deriv_yy(...)                       ->  DERIVS->grad_yy(...)

    Leaves advective derivatives (adv_deriv_x) unchanged.
    """
    import re
    # Match deriv_x, deriv_y, deriv_z, deriv_xx, deriv_yy, deriv_zz, deriv_xy, etc.
    # but NOT adv_deriv_x or ko_deriv_x
    pattern = r'(?<![a-zA-Z_])deriv_(x{1,2}|y{1,2}|z{1,2}|xy|xz|yz)\('
    def replacer(m):
        suffix = m.group(1)
        return f'{deriv_obj}->grad_{suffix}('
    return re.sub(pattern, replacer, code)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class DendroProjectGenerator:
    """Generates a complete Dendro-based solver project from a configuration.

    Parameters
    ----------
    config : dendrosym.DendroConfiguration or dendrosym.NRConfig
        The fully-configured equation system (variables, parameters, RHS
        functions, BCS info, etc. must already be set on this object).
    """

    def __init__(self, config):
        self.config = config
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_dir: str, *, skip_gencode: bool = False,
                 gencode_only: bool = False):
        """Generate the full solver project into *output_dir*.

        Parameters
        ----------
        output_dir : str or Path
            Where to write the project.  Created if it doesn't exist.
        skip_gencode : bool
            If True, skip the expensive CSE / equation generation step and
            only regenerate templates (useful when iterating on C++ boilerplate).
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # 1. Build the context dict that all templates share
        print("Building template context...", file=sys.stderr)
        ctx = self._build_context()

        # 2. Generate the expensive CSE-optimised .cpp.inc files
        if not skip_gencode:
            print("Generating equation code (gencode/)...", file=sys.stderr)
            self._generate_gencode(output / "gencode", ctx)
        else:
            print("Skipping gencode (skip_gencode=True)", file=sys.stderr)

        if not gencode_only:
            # 3. Render templates -> src/ and include/
            print("Rendering templates...", file=sys.stderr)
            self._render_templates(output, ctx)

            # 4. Copy static files (derivs, etc.)
            print("Copying static files...", file=sys.stderr)
            self._copy_static_files(output)
        else:
            print("Skipping templates (gencode_only=True)", file=sys.stderr)

        print(f"Project generated in {output}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(self) -> dict:
        """Run all cheap symbolic queries and return a template context dict."""
        c = self.config
        ctx = {}

        # -- project-level names
        ctx["project_name"] = c.project_name
        ctx["project_upper"] = c.project_upper
        ctx["namespace"] = c.project_name

        # -- variable types present (e.g. evolution, constraint)
        var_types = [k for k in c.all_var_names if k != "parameter"]
        ctx["var_types"] = var_types

        # -- per-variable-type info
        for vt in var_types:
            names = c.all_var_names.get(vt, [])
            enum_names = c.get_enum_var_names(vt)

            ctx[f"{vt}_var_names"] = names
            ctx[f"{vt}_enum_names"] = enum_names
            ctx[f"{vt}_num_vars"] = len(names)

            # enum code block
            enum_name = "VAR" if vt in ("evolution", "general") else "VAR_CONSTRAINT"
            ctx[f"{vt}_enum_code"] = c.gen_enum_code(vt, enum_name=enum_name)
            ctx[f"{vt}_enum_name_array"] = c.gen_enum_names(vt, enum_name=enum_name)
            ctx[f"{vt}_iterable_list"] = c.gen_enum_iterable_list(
                vt, enum_name=enum_name
            )

            # variable extraction code
            ctx[f"{vt}_var_extraction"] = c.generate_variable_extraction(vt)
            ctx[f"{vt}_rhs_var_extraction"] = c.generate_rhs_var_extraction(vt)

            # For constraint output in physcon, use a different zip var name
            if vt == "constraint":
                ctx[f"{vt}_output_extraction"] = c.generate_rhs_var_extraction(
                    vt, zip_var_name="unzipConsVars"
                )

        # -- parameter extraction code (per param-subtype like evolution, constraint)
        for param_subtype in c.all_vars.get("parameter", {}):
            ctx[f"parameter_code_{param_subtype}"] = c.gen_parameter_code(param_subtype)

        # -- physics parameter metadata (for parameters.h/cpp and sample TOML)
        physics_params = []
        seen_param_names = set()
        for param_subtype, param_list in c.all_vars.get("parameter", {}).items():
            for pvar in param_list:
                if pvar.var_name in seen_param_names:
                    continue
                seen_param_names.add(pvar.var_name)
                toml_key = pvar.var_name
                cpp_var = f"{c.project_upper}_{pvar.var_name.upper()}"

                # Format default for TOML
                if pvar.num_params > 1:
                    toml_default = repr(pvar.default)
                else:
                    default_val = pvar.default
                    if isinstance(default_val, bool):
                        toml_default = "true" if default_val else "false"
                    elif isinstance(default_val, float):
                        toml_default = str(default_val)
                    else:
                        toml_default = str(default_val)

                physics_params.append({
                    "var_name": pvar.var_name,
                    "toml_key": toml_key,
                    "cpp_var": cpp_var,
                    "dtype": pvar.dtype,
                    "num_params": pvar.num_params,
                    "default": pvar.default,
                    "toml_default": toml_default,
                    "description": pvar.description or f"Parameter: {pvar.var_name}",
                    "required": "OPTIONAL",  # physics params default optional
                })
        ctx["physics_params"] = physics_params

        # NOTE: BCS, evolution constraints, and KO dissipation are deferred
        # to _generate_gencode because they can trigger expensive derivative
        # expansion and CSE as a side effect. We set empty defaults here so
        # templates can always reference them.
        for vt in var_types:
            ctx[f"{vt}_bcs_code"] = ""
            ctx[f"{vt}_ko_code"] = ""
        ctx["evolution_constraint_enforcement"] = ""

        # -- feature flags (templates use these to conditionally include code)
        ctx["enable_bh_tracking"] = getattr(c, "enable_bh_tracking", False)
        ctx["enable_gw_extraction"] = getattr(c, "enable_gw_extraction", False)
        ctx["enable_tpid"] = getattr(c, "enable_tpid", False)

        # -- derivative system: if set, emits DendroDerivatives method calls
        # e.g. "SOLVER_DERIVS" -> SOLVER_DERIVS->grad_x(...)
        ctx["deriv_obj"] = getattr(c, "deriv_obj", "")
        ctx["use_dendro_derivs"] = ctx["deriv_obj"] != ""

        return ctx

    # ------------------------------------------------------------------
    # Gencode (expensive: CSE, derivatives, equation printing)
    # ------------------------------------------------------------------

    def _generate_gencode(self, gencode_dir: Path, ctx: dict):
        """Generate .cpp.inc files for each variable type."""
        gencode_dir.mkdir(parents=True, exist_ok=True)
        c = self.config
        prefix = c.project_name

        for vt in ctx["var_types"]:
            if c.all_rhs_functions.get(vt) is None:
                print(f"  skipping {vt} (no RHS function set)", file=sys.stderr)
                continue

            print(f"  processing {vt}...", file=sys.stderr)

            # -- must find/expand derivatives first (this populates stored_rhs_function)
            print(f"    finding and expanding derivatives...", file=sys.stderr)
            c.find_derivatives(vt)

            # -- derivative allocation, calculation, deallocation
            print(f"    generating derivative code...", file=sys.stderr)
            deriv_alloc, deriv_calc, deriv_dealloc = (
                c.generate_deriv_allocation_and_calc(
                    vt, include_byte_declaration=False
                )
            )

            alloc_file = f"{prefix}_{vt}_deriv_memalloc.cpp.inc"
            calc_file = f"{prefix}_{vt}_deriv_calc.cpp.inc"
            dealloc_file = f"{prefix}_{vt}_deriv_memdealloc.cpp.inc"

            # If using DendroDerivatives object, rewrite the legacy
            # deriv_x(...) calls to DERIVS->grad_x(...) style
            deriv_obj_name = ctx.get("deriv_obj", "")
            if deriv_obj_name:
                deriv_calc = _rewrite_deriv_calls(deriv_calc, deriv_obj_name)

            (gencode_dir / alloc_file).write_text(deriv_alloc)
            (gencode_dir / calc_file).write_text(deriv_calc)
            (gencode_dir / dealloc_file).write_text(deriv_dealloc)

            # -- intermediate / advanced derivatives
            print(f"    generating intermediate derivatives...", file=sys.stderr)
            try:
                intermediate_str, dealloc_intermediate_str = (
                    c.generate_pre_necessary_derivatives(
                        vt, dtype="double", include_byte_declaration=False
                    )
                )
            except Exception:
                intermediate_str = ""
                dealloc_intermediate_str = ""

            intermediate_file = f"{prefix}_{vt}_intermediate_grad.cpp.inc"
            intermediate_dealloc_file = (
                f"{prefix}_{vt}_intermediate_grad_dealloc.cpp.inc"
            )
            (gencode_dir / intermediate_file).write_text(intermediate_str)
            (gencode_dir / intermediate_dealloc_file).write_text(
                dealloc_intermediate_str
            )

            # -- RHS equations (the big one)
            print(f"    generating RHS equations...", file=sys.stderr)
            rhs_code = c.generate_rhs_code(vt)
            rhs_file = f"{prefix}_{vt}_rhs_eqns.cpp.inc"
            (gencode_dir / rhs_file).write_text(rhs_code)

            # -- KO derivative calculations
            print(f"    generating KO derivative code...", file=sys.stderr)
            try:
                ko_deriv_alloc, ko_deriv_calc, ko_deriv_dealloc = (
                    c.generate_ko_deriv_allocation_and_calc(vt)
                )
            except Exception:
                ko_deriv_alloc = ""
                ko_deriv_calc = ""
                ko_deriv_dealloc = ""

            ko_calc_file = f"{prefix}_{vt}_ko_deriv_calc.cpp.inc"
            (gencode_dir / ko_calc_file).write_text(ko_deriv_calc)

            # store filenames in context for template use
            ctx[f"{vt}_gencode"] = {
                "deriv_alloc": alloc_file,
                "deriv_calc": calc_file,
                "deriv_dealloc": dealloc_file,
                "intermediate_grad": intermediate_file,
                "intermediate_grad_dealloc": intermediate_dealloc_file,
                "rhs_eqns": rhs_file,
                "ko_deriv_calc": ko_calc_file,
            }

            # -- BCS code (deferred from _build_context to avoid premature CSE)
            print(f"    generating BCS code...", file=sys.stderr)
            try:
                ctx[f"{vt}_bcs_code"] = c.generate_bcs_calculations(vt)
            except Exception:
                ctx[f"{vt}_bcs_code"] = ""

            # -- KO dissipation code
            print(f"    generating KO code...", file=sys.stderr)
            try:
                ctx[f"{vt}_ko_code"] = c.generate_ko_calculations(vt)
            except Exception:
                ctx[f"{vt}_ko_code"] = ""

            print(f"  {vt} done.", file=sys.stderr)

        # -- evolution constraints (deferred from _build_context)
        if hasattr(c, "generate_evolution_constraints"):
            print("  generating evolution constraint enforcement...", file=sys.stderr)
            try:
                ctx["evolution_constraint_enforcement"] = (
                    c.generate_evolution_constraints()
                )
            except Exception:
                ctx["evolution_constraint_enforcement"] = ""

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_templates(self, output: Path, ctx: dict):
        """Render all Jinja2 templates into the output directory."""

        # Map of output path -> template path (relative to templates/)
        template_map = {
            # GR solver templates
            "include/grDef.h": "gr/grDef.h.j2",
            "include/rhs.h": "gr/rhs.h.j2",
            "include/physcon.h": "gr/physcon.h.j2",
            "include/parameters.h": "gr/parameters.h.j2",
            f"include/{ctx['project_name']}_constraints.h": "gr/constraints.h.j2",
            f"include/{ctx['project_name']}Ctx.h": "gr/solver_ctx.h.j2",
            "include/grUtils.h": "gr/grUtils.h.j2",
            "src/rhs.cpp": "gr/rhs.cpp.j2",
            "src/physcon.cpp": "gr/physcon.cpp.j2",
            "src/parameters.cpp": "gr/parameters.cpp.j2",
            f"src/{ctx['project_name']}Ctx.cpp": "gr/solver_ctx.cpp.j2",
            "src/grUtils.cpp": "gr/grUtils.cpp.j2",
            f"{ctx['project_name']}_main.cpp": "gr/main.cpp.j2",
            # Sample parameter file
            f"{ctx['project_name']}_parameters.sample.toml": "gr/sample_params.toml.j2",
            # Common templates
            "CMakeLists.txt": "common/CMakeLists.txt.j2",
            "solver/CMakeLists.txt": "common/solver_CMakeLists.txt.j2",
        }

        for out_rel, tmpl_name in template_map.items():
            tmpl_path = _TEMPLATES_DIR / tmpl_name
            if not tmpl_path.exists():
                print(
                    f"  WARNING: template {tmpl_name} not found, skipping",
                    file=sys.stderr,
                )
                continue

            tmpl = self.jinja_env.get_template(tmpl_name)
            rendered = tmpl.render(**ctx)

            out_path = output / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered)
            print(f"  wrote {out_rel}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Static file copying
    # ------------------------------------------------------------------

    def _copy_static_files(self, output: Path):
        """Copy static (non-templated) files into the output directory."""
        static_dir = _TEMPLATES_DIR / "static"
        if not static_dir.exists():
            return

        for src_file in static_dir.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(static_dir)
                dst = output / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst)
                print(f"  copied {rel}", file=sys.stderr)
