"""nr_configs.py

This file contains the configuration class to generate
and store the necessary pieces of information regarding
the set-up and configuration for a numerical relativity
project using the Dendro framework.
"""

from dendrosym.general_configs import ImproperInitalization
import sympy as sym

import dendrosym


class AdvectiveDerivativeNotSet(Exception):
    pass


class NRConfig(dendrosym.DendroConfiguration):
    """A class to store numerical relativity configurations

    In particular, this class can store infomration about
    the numerical relativity project such as RHS equations,
    store variables, and then generate the C++ code
    from the symbolic equations.
    """

    def __init__(self, project_name: str, project_description: str = ""):
        super().__init__(
            project_name=project_name, project_description=project_description
        )

        # make all vars a dictionary in case there are
        # other types to store, but by default we will have
        # "constraint", "evolution", and "parameter" which are initialized
        # by default
        self.all_vars = {
            "constraint": [],
            "evolution": [],
            "parameter": {"constraint": [], "evolution": []},
        }

        # also create the same type of dictionary but
        # just for string representations
        self.all_var_names = {
            "constraint": [],
            "evolution": [],
            "parameter": {"constraint": [], "evolution": []},
        }

        self.enum_prefixes = {"constraint": "C", "evolution": "U"}

        # then also store the functions that return the rhs
        # portions
        self.all_rhs_functions = {"constraint": None, "evolution": None}

        # initialize the advective derivative variable as well
        self.advective_der_var = None

        self.idx_str = "[pp]"

        self.bcs_info = {"constraint": {}, "evolution": {}}

        self.metric_var = None

        self.evolution_constraint_info = {"trace_zero": [], "pos_floor": []}

        # NOTE: stored_rhs_function is the only thing that is restored by the pickle!
        # self.stored_rhs_function = {}

        # and initialize the intial data functions
        self.all_initial_data_functions = {"general": [], "evolution": []}

    def add_evolution_variables(self, in_vars: list):
        return super().add_variable(in_vars, "evolution")

    def add_constraint_variables(self, in_vars: list):
        return super().add_variable(in_vars, "constraint")

    def add_parameter_variables(self, in_vars: list, eqn_type: str = "evolution"):
        return super().add_parameter_variables(in_vars, eqn_type)

    def add_initial_data_params(self, in_vars: list):
        """Register params used only by initial data, not by the RHS.

        Kept in a separate "initial_data" group so a reader can tell at a
        glance which params actually feed the equations vs. which only affect
        ICs. Params still get emitted as global C++ macros via the standard
        parameters.cpp iteration; only rhs.cpp's local-copy block skips them.
        """
        # framework validates eqn_type against all_vars.keys() (outer dict),
        # but stores into all_vars["parameter"][eqn_type] (inner). pre-declare
        # the slot at both levels. do NOT touch all_var_names outer -- that
        # would trigger empty-group variable-extraction codegen.
        self.all_vars.setdefault("initial_data", [])
        self.all_vars["parameter"].setdefault("initial_data", [])
        self.all_var_names["parameter"].setdefault("initial_data", [])
        return super().add_parameter_variables(in_vars, "initial_data")

    def set_advective_derivative_var(self, in_var):
        # this should be a 3x1 variable
        if isinstance(in_var, sym.Matrix):
            if in_var.shape[0] != 3:
                raise ImproperInitalization(
                    "Invalid type of variable for use as advective derivative"
                )
        elif isinstance(in_var, tuple):
            if len(in_var) != 3:
                raise ImproperInitalization(
                    "Invalid type of variable for use as advective derivative"
                )
        self.advective_der_var = str(in_var[0]).split("[")[0][:-1]

    def set_metric(self, in_var):
        assert in_var in self.all_vars.get(
            "evolution", []
        ), "Incoming metric variable is not in evolution variables"
        self.metric_var = in_var
        # also set the metric in the NR code for use there
        dendrosym.nr.set_metric(in_var)

    def add_evolution_constraint(self, var_adjust, constraint_info):
        """Adds an evolution constraint of some kind

        Takes in the input variable that needs to have some kind of constraint
        and then information about the constraint. The constraint info
        currently supports 'trace_zero', 'pos_floor'

        Trace of zero requires the metric to have been set, and the
        generation code will ensure that it has been properly extracted
        and updated due to the metric's determinant being 1.

        Positive threshold only works for single values and requires
        a parameter be set *elsewhere* (in parameters.cpp for example)
        with the name "{VAR_NAME}_FLOOR" where 'VAR_NAME' is the same
        name that was assigned to the variable but in all capitals.
        """

        if constraint_info == "trace_zero":
            assert var_adjust in self.all_vars.get(
                "evolution", []
            ), "Variable not in set evolution variables"

            if type(var_adjust) is not sym.Matrix:
                raise ValueError(
                    "For zero trace, the incoming variable needs to be a matrix"
                )

            self.evolution_constraint_info["trace_zero"].append(var_adjust)
        elif constraint_info == "pos_floor":
            assert var_adjust in self.all_vars.get(
                "evolution", []
            ), "Variable not in set evolution variables"

            if type(var_adjust) is not sym.Symbol:
                raise ValueError(
                    "For positive floor, the variable needs to not have dimensions"
                )

            self.evolution_constraint_info["pos_floor"].append(var_adjust)
        else:
            raise NotImplementedError(
                "This type of constraint has not been implemented"
            )

    def generate_evolution_constraints(self):
        if self.metric_var is None:
            raise ImproperInitalization(
                "Metric was not set, cannot generate evolution constraints"
            )

        return_str = ""

        metric_var_names = self.clean_var_names([self.metric_var])

        metric_name = metric_var_names[0][:-2]

        # then we also need to get the metric enum information
        enum_prefix = self.enum_prefixes.get("evolution", "")
        metric_enums = [
            f"VAR::{enum_prefix}_{mvar.upper()}" for mvar in metric_var_names
        ]

        # first things first, the metric determinant needs to be zero,
        # so we generate that code first
        return_str += "////\n//// CODE TO REQUIRE METRIC DETERMINANT TO BE ONE\n"
        return_str += dendrosym.codegen.generate_update_sym_mat_extract(
            metric_name, metric_enums
        )

        # then we need to generate the determinant code
        return_str += dendrosym.codegen.generate_force_sym_matrix_det_to_one(
            metric_name, metric_enums
        )

        # now we can start adding our other constraints
        for var_use in self.evolution_constraint_info.get("trace_zero", []):
            return_str += "////\n////APPLYING TRACE ZERO TO "
            # for trace of zero, we need to get all of the variables
            var_names = self.clean_var_names([var_use])
            # then get the name of the variable
            single_var_name = var_names[0][:-2]
            return_str += single_var_name + "\n"
            # then generate the enums
            var_enums = [f"VAR::{enum_prefix}_{ivar.upper()}" for ivar in var_names]

            # then add the code to extract the sym matrix
            return_str += dendrosym.codegen.generate_update_sym_mat_extract(
                single_var_name, var_enums
            )

            # then the code to force the traceless symmat
            return_str += dendrosym.codegen.generate_force_symmat_traceless(
                single_var_name, metric_name
            )

            # then the code to actually update the matrix
            return_str += dendrosym.codegen.generate_update_sym_mat_code(
                single_var_name, var_enums, include_up=False
            )
            return_str += "//// END APPLICATION OF TRACE ZERO\n////\n\n"

        for var_use in self.evolution_constraint_info.get("pos_floor"):
            return_str += "////\n////APPLYING POSITIVE FLOOR TO "
            single_var_name = self.clean_var_names([var_use])[0]
            return_str += single_var_name + "\n"

            var_enum = f"VAR::{enum_prefix}_{single_var_name.upper()}"

            return_str += dendrosym.codegen.generate_variable_always_positive(
                var_enum, floor_var=f"{single_var_name.upper()}_FLOOR"
            )
            return_str += "//// END APPLICATION OF POSITIVE FLOOR\n////\n\n"

        # now generate the metric update code
        return_str += "//// NOW UPDATING THE METRIC VALUES\n"
        return_str += dendrosym.codegen.generate_update_sym_mat_code(
            metric_name, metric_enums
        )

        return return_str

    def __repr__(self):
        return f"<NRConfigs for '{self.project_name}'>"
