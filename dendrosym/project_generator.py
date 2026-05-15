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


def _rewrite_deriv_calls(code: str, deriv_obj: str,
                         *, use_advective: bool = False) -> str:
    """Rewrite legacy derivative calls to use a DendroDerivatives object.

    Non-advective: deriv_{x,xx,xy,...}(...) -> DERIVS.grad_{x,xx,xy,...}(...).
    Advective: adv_deriv_{x,y,z}(...,betax,bflag) -> DERIVS.grad_x(...,bflag)
    when use_advective=False (drops upwinding; dendrolib Derivs has no
    advective method yet). use_advective=True leaves the calls in place.
    """
    import re
    # plain centered FD: deriv_x / deriv_xx / deriv_xy / ...
    pattern = r'(?<![a-zA-Z_])deriv_(x{1,2}|y{1,2}|z{1,2}|xy|xz|yz)\('
    code = re.sub(pattern, lambda m: f'{deriv_obj}.grad_{m.group(1)}(', code)

    # advective FD: adv_deriv_x / adv_deriv_y / adv_deriv_z
    if not use_advective:
        # collapse onto non-advective by dropping the 5th arg (betax):
        # adv_deriv_x(out,in,hx,sz,betax,bflag) -> DERIVS.grad_x(out,in,hx,sz,bflag)
        adv_call_pattern = re.compile(
            r'(?<![a-zA-Z_])adv_deriv_([xyz])\('
            r'\s*([^,]+?)\s*,'   # out
            r'\s*([^,]+?)\s*,'   # in
            r'\s*([^,]+?)\s*,'   # hx
            r'\s*([^,]+?)\s*,'   # sz
            r'\s*[^,]+?\s*,'     # betax (discarded)
            r'\s*([^)]+?)\s*\)'  # bflag
        )
        code = adv_call_pattern.sub(
            lambda m: (
                f'{deriv_obj}.grad_{m.group(1)}'
                f'({m.group(2)}, {m.group(3)}, {m.group(4)}, '
                f'{m.group(5)}, {m.group(6)})'
            ),
            code,
        )
    return code


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
            self._generate_gencode(output / "solver" / "gencode", ctx)
        else:
            print("Skipping gencode (skip_gencode=True)", file=sys.stderr)
            # still set the expected gencode filenames so templates can include them
            prefix = self.config.project_name
            for vt in ctx["var_types"]:
                ctx[f"{vt}_gencode"] = {
                    "deriv_alloc": f"{prefix}_{vt}_deriv_memalloc.cpp.inc",
                    "deriv_calc": f"{prefix}_{vt}_deriv_calc.cpp.inc",
                    "deriv_dealloc": f"{prefix}_{vt}_deriv_memdealloc.cpp.inc",
                    "intermediate_grad": f"{prefix}_{vt}_intermediate_grad.cpp.inc",
                    "intermediate_grad_dealloc": f"{prefix}_{vt}_intermediate_grad_dealloc.cpp.inc",
                    "rhs_eqns": f"{prefix}_{vt}_rhs_eqns.cpp.inc",
                    "ko_deriv_calc": f"{prefix}_{vt}_ko_deriv_calc.cpp.inc",
                }

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
            ctx[f"{vt}_rhs_var_extraction"] = c.generate_rhs_var_extraction(
                vt, zip_var_name="unzipVarsRHS"
            )

            # For constraint output in physcon, use a different zip var name
            if vt == "constraint":
                # need to use the right enum name for constraints
                named_enums_c = c.get_enum_var_names(vt)
                rhs_names_c = c.get_rhs_var_names(vt)
                ctx[f"{vt}_output_extraction"] = dendrosym.codegen.gen_var_info(
                    rhs_names_c,
                    zip_var_name="unzipConsVars",
                    use_const=False,
                    enum_name="VAR_CONSTRAINT",
                    enum_var_names=named_enums_c,
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

                # map dtypes to C++ types
                cpp_dtype = pvar.dtype
                if "unsigned int" in cpp_dtype:
                    cpp_dtype_base = "unsigned int"
                elif "int" in cpp_dtype:
                    cpp_dtype_base = "int"
                elif "double" in cpp_dtype:
                    cpp_dtype_base = "double"
                elif "float" in cpp_dtype:
                    cpp_dtype_base = "float"
                else:
                    cpp_dtype_base = "double"

                # build extern declaration and definition
                if pvar.num_params > 1:
                    extern_decl = f"extern {cpp_dtype_base} {cpp_var}[{pvar.num_params}];"
                    default_str = "{" + ", ".join(str(d) for d in pvar.default) + "}"
                    definition = f"{cpp_dtype_base} {cpp_var}[{pvar.num_params}] = {default_str};"
                else:
                    extern_decl = f"extern {cpp_dtype_base} {cpp_var};"
                    definition = f"{cpp_dtype_base} {cpp_var} = {pvar.default};"

                physics_params.append({
                    "var_name": pvar.var_name,
                    "toml_key": toml_key,
                    "cpp_var": cpp_var,
                    "dtype": pvar.dtype,
                    "cpp_dtype": cpp_dtype_base,
                    "num_params": pvar.num_params,
                    "default": pvar.default,
                    "toml_default": toml_default,
                    "description": pvar.description or f"Parameter: {pvar.var_name}",
                    "required": "OPTIONAL",
                    "extern_decl": extern_decl,
                    "definition": definition,
                })
        ctx["physics_params"] = physics_params

        # NOTE: BCS, evolution constraints, and KO dissipation are deferred
        # to _generate_gencode because they can trigger expensive derivative
        # expansion and CSE as a side effect. We set empty defaults here so
        # templates can always reference them.
        for vt in var_types:
            ctx[f"{vt}_bcs_code"] = ""
            ctx[f"{vt}_ko_code"] = ""

        # evolution-constraint enforcement is just string-gen (no CSE), so
        # populate it here -- means --skip-gencode regens still emit it
        if hasattr(c, "generate_evolution_constraints"):
            try:
                ctx["evolution_constraint_enforcement"] = (
                    c.generate_evolution_constraints()
                )
            except Exception:
                ctx["evolution_constraint_enforcement"] = ""
        else:
            ctx["evolution_constraint_enforcement"] = ""

        # pos_floor vars: feed `<VAR>_FLOOR` constants to parameters.h/cpp so
        # constraints.h enforcement compiles for any conformal-factor name
        pos_floor_vars = []
        ec_info = getattr(c, "evolution_constraint_info", None)
        if ec_info and "pos_floor" in ec_info:
            for sym_obj in ec_info["pos_floor"]:
                vname = str(sym_obj)
                if vname.endswith(c.idx_str):
                    vname = vname[: -len(c.idx_str)]
                pos_floor_vars.append(vname)
        ctx["pos_floor_vars"] = pos_floor_vars

        # -- feature flags (templates use these to conditionally include code)
        ctx["enable_bh_tracking"] = getattr(c, "enable_bh_tracking", False)
        ctx["enable_gw_extraction"] = getattr(c, "enable_gw_extraction", False)
        ctx["enable_tpid"] = getattr(c, "enable_tpid", False)
        # AH finder: on by default for GR, gated by a compile-time cmake flag
        ctx["enable_ah"] = getattr(c, "enable_ah", True)

        # derivative system: if set, emit DendroDerivatives method calls
        ctx["deriv_obj"] = getattr(c, "deriv_obj", "")
        ctx["use_dendro_derivs"] = ctx["deriv_obj"] != ""

        # initial data types -- entries carry "code" (raw C) or "sympy_exprs"
        # ({var: expr}); sympy_exprs are converted to C after param_subs is built
        ctx["_raw_initial_data_types"] = getattr(c, "initial_data_types", [])

        ctx["enable_analytical"] = getattr(c, "enable_analytical", False)

        # symbolic IC + analytical -> C code via DendroCPrinter
        import sympy as sym
        from dendrosym.code_printer import DendroCPrinter

        cprinter = DendroCPrinter()

        # param symbols -> global C++ names, e.g. wave_speed -> WAVE_WAVE_SPEED
        param_subs = {}
        for param_subtype, param_list in c.all_vars.get("parameter", {}).items():
            for pvar in param_list:
                for s in (pvar.var_symbols if isinstance(pvar.var_symbols, tuple)
                          else [pvar.var_symbols]):
                    global_name = f"{c.project_upper}_{pvar.var_name.upper()}"
                    if pvar.num_params > 1:
                        # array params like lambda[0] -> PROJECT_LAMBDA[0]
                        idx_str = str(s)
                        if "[" in idx_str:
                            idx = idx_str.split("[")[1].rstrip("]")
                            param_subs[s] = sym.Symbol(f"{global_name}[{idx}]")
                        else:
                            param_subs[s] = sym.Symbol(global_name)
                    else:
                        param_subs[s] = sym.Symbol(global_name)

        # tag id types needing sympy->C conversion; we convert below once
        # param_subs is finalized (runtime_symbol_map gets merged in next)
        processed_id_types = []
        for idt in ctx.pop("_raw_initial_data_types", []):
            idt = dict(idt)
            if "sympy_exprs" in idt and idt["sympy_exprs"]:
                idt["_needs_sympy_conversion"] = True
            processed_id_types.append(idt)
        ctx["_pending_initial_data_types"] = processed_id_types

        # user-defined runtime symbol mappings -- sympy symbols -> C++ runtime
        # values (e.g. BH mass, coordinates)
        runtime_subs = getattr(c, "runtime_symbol_map", {})
        param_subs.update(runtime_subs)

        # finalize id types: sympy_exprs -> C lines
        final_id_types = []
        for idt in ctx.pop("_pending_initial_data_types", []):
            if idt.pop("_needs_sympy_conversion", False):
                sympy_exprs = idt.pop("sympy_exprs")
                lines = []
                for var_sym, expr in sympy_exprs.items():
                    var_name = str(var_sym).replace(c.idx_str, "")
                    expr = sym.sympify(expr)
                    expr_sub = expr.subs(param_subs)
                    ccode = cprinter.doprint(expr_sub)
                    lines.append(
                        f"    var[VAR::U_{var_name.upper()}] = {ccode};"
                    )
                idt["code"] = "\n".join(lines)
            final_id_types.append(idt)
        ctx["initial_data_types"] = final_id_types

        # symbolic initial data: dict of {var_symbol: sympy_expr}
        sym_init_data = getattr(c, "symbolic_initial_data", {})
        if sym_init_data:
            init_lines = []
            for var_sym, expr in sym_init_data.items():
                var_name = str(var_sym).replace(c.idx_str, "")
                expr_sub = expr.subs(param_subs)
                ccode = cprinter.doprint(expr_sub)
                init_lines.append(
                    f"var[VAR::U_{var_name.upper()}] = {ccode};"
                )
            ctx["symbolic_init_code"] = "\n".join(init_lines)
            ctx["symbolic_init_name"] = getattr(
                c, "symbolic_initial_data_name", "symbolicInit"
            )
        else:
            ctx["symbolic_init_code"] = ""

        # symbolic analytical solution: dict of {var_symbol: sympy_expr(x,y,z,t)}
        sym_analytical = getattr(c, "symbolic_analytical_solution", {})
        if sym_analytical:
            ana_lines = []
            for var_sym, expr in sym_analytical.items():
                var_name = str(var_sym).replace(c.idx_str, "")
                expr_sub = expr.subs(param_subs)
                ccode = cprinter.doprint(expr_sub)
                ana_lines.append(
                    f"var[VAR::U_{var_name.upper()}] = {ccode};"
                )
            ctx["symbolic_analytical_code"] = "\n".join(ana_lines)
            # auto-enable when the user provides a symbolic analytical solution
            ctx["enable_analytical"] = True
        else:
            ctx["symbolic_analytical_code"] = ""

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
                use_advective = getattr(c, "use_advective_derivs", False)
                deriv_calc = _rewrite_deriv_calls(
                    deriv_calc, deriv_obj_name, use_advective=use_advective
                )

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
            "solver/include/grDef.h": "gr/grDef.h.j2",
            **({"solver/include/bh.h": "gr/bh.h.j2"} if ctx.get("enable_bh_tracking") else {}),
            "solver/include/rhs.h": "gr/rhs.h.j2",
            "solver/include/physcon.h": "gr/physcon.h.j2",
            "solver/include/parameters.h": "gr/parameters.h.j2",
            f"solver/include/{ctx['project_name']}_constraints.h": "gr/constraints.h.j2",
            f"solver/include/{ctx['project_name']}Ctx.h": "gr/solver_ctx.h.j2",
            "solver/include/grUtils.h": "gr/grUtils.h.j2",
            "solver/include/profile_params.h": "gr/profile_params.h.j2",
            "solver/src/rhs.cpp": "gr/rhs.cpp.j2",
            "solver/src/profile_params.cpp": "gr/profile_params.cpp.j2",
            "solver/src/physcon.cpp": "gr/physcon.cpp.j2",
            "solver/src/parameters.cpp": "gr/parameters.cpp.j2",
            f"solver/src/{ctx['project_name']}Ctx.cpp": "gr/solver_ctx.cpp.j2",
            "solver/src/grUtils.cpp": "gr/grUtils.cpp.j2",
            f"solver/{ctx['project_name']}_main.cpp": "gr/main.cpp.j2",
            # sample parameter file (at project root for easy access)
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
