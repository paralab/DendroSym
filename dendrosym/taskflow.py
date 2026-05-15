# generate some of the code
import textwrap

import sympy as sym
import dendrosym

DIR_TO_NUM = {"x": 0, "y": 1, "z": 2}


def indent(text, amount, ch=" "):
    return textwrap.indent(text, amount * ch)


def single_var_derivatives(var_name, deriv_list=["x", "y", "z"], order=1):
    if len(deriv_list) == 1:
        task_name = f"deriv {var_name} {deriv_list[0]}"
        task_var = f"deriv_{var_name}_{deriv_list[0]}"
    else:
        task_name = "fused deriv " + var_name
        task_var = f"fused_derivative_{var_name}_deriv{order}"

    text = f"""
auto {task_var} = tf.placeholder("{task_name}");

{task_var}.work([{task_var}]() {{

    auto d = *static_cast<tfdendro::dendrotf_rhs_data *>(
        {task_var}.data());

    unsigned int sz[3] = {{d.nx, d.ny, d.nz}};

"""

    for deriv in deriv_list:
        dir = deriv[0]
        dir_num = DIR_TO_NUM[dir]

        text += (
            f"    dendro_derivs::deriv_{deriv}"
            + f"(d.grad{'' if order==1 else str(order)}_{dir_num}_{var_name}, "
            + f"d.{var_name}, d.h{dir}, sz, d.bflag);\n"
        )

    text += "});\n"

    return task_var, text


def rhs_computation_singlevar(var_name, cse_list, original_ops):
    task_var = f"{var_name}_rhs_task"
    task_name = f"RHS Computation {var_name}"

    text = f"""auto {task_var} = tf.placeholder("{task_name}")
{task_var}.work([{task_var}]() {{
    auto d = *static_cast<tfdendro::dendrotf_rhs_data *>(
        {task_var}.data());
    DendroRegister double x;
    DendroRegister double y;
    DendroRegister double z;

    DendroRegister unsigned int pp;

    double r;

    for (unsigned int k = d.PW; k < d.nz - d.PW; k++) {{
        z = d.pmin[2] + k * d.hz;
        for (unsigned int j = d.PW; j < d.ny - d.PW; j++) {{
            y = d.pmin[1] + j * d.hy;
            for (unsigned int i = d.PW; i < d.nx - d.PW; i++) {{
                x = d.pmin[0] + i * d.hx;
                pp = i + d.nx + (j + d.ny * k);
"""

    # then generate for the CSE and stuff

    output_str = dendrosym.codegen.generate_cpu_preextracted(
        cse_list, [f"d.{var_name}_rhs"], "[pp]", original_ops
    )

    text += indent(output_str, 16, " ")

    text += "            }\n        }\n    }\n"
    text += "});\n"

    return task_var, text


task_var, text = single_var_derivatives("E0", ["x", "y", "z"])


cse_list = [[], [sym.symbols("x") + 1]]

task_var_rhs, rhs_text = rhs_computation_singlevar("E0", cse_list, 1)

print(task_var)
print(text)


print(task_var_rhs)
print(rhs_text)
