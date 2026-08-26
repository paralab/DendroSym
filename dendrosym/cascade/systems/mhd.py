"""mhd.py -- Ideal MHD x-direction flux: SymPy spec.

Pointwise operator F : (conserved variables) -> (x-direction flux of those
variables) for ideal magnetohydrodynamics with an ideal-gas EOS.

Inputs (8 conserved variables per grid point):
    rho                     -- mass density
    mom_x, mom_y, mom_z     -- momentum density components
    E                       -- total energy density
    Bx, By, Bz              -- magnetic field components

Plus one parameter:
    gamma_minus_1           -- (gamma - 1) for the ideal-gas EoS
                               (e.g. 2/3 for gamma = 5/3)

Outputs (8 flux components in the x-direction):
    F_rho, F_mom_x, F_mom_y, F_mom_z, F_E, F_Bx, F_By, F_Bz

The flux formulas come from the standard ideal-MHD conservation form:

    F = | rho v_x                                |
        | rho v_x^2 + p_total - Bx^2             |
        | rho v_x v_y - Bx By                    |
        | rho v_x v_z - Bx Bz                    |
        | (E + p_total) v_x - Bx (v . B)         |
        | 0                                      |
        | v_x By - v_y Bx                        |
        | v_x Bz - v_z Bx                        |

with primitives v_i = mom_i / rho and EOS:

    p       = (gamma - 1) * (E - rho*v^2/2 - B^2/2)
    p_total = p + B^2/2

This is a medium-complexity cascade: 4-5 layers, ~20 named intermediates,
peak-live ~25. Sits between Neo-Hookean (~12-38 peak-live, flat U-shape)
and BSSN (~89 peak-live, clear U-shape) in the system-size dependence
study.

The talk's slide-12 catalog lists ideal MHD at L=5; depending on how
finely you split the primitive recovery vs the energy/pressure layer,
you land at 4 or 5. We use 5 layers natural-depth here to match the talk.
"""

from collections import OrderedDict
import sympy as sym


def mhd_flux_spec(per_point: bool = True):
    """Build the ideal-MHD x-direction flux as a 5-layer named-tensor cascade.

    Parameters
    ----------
    per_point : bool, default True
        If True, leaves are named `rho[pp]`, `mom_x[pp]`, ..., and outputs
        are `F_rho_out`, etc. -- so cascade_emit's auto-classifier picks
        them up as VLOAD'd input arrays / VSTORE'd output arrays.
        If False, bare names (matches a scalar bench wrapper that declares
        the values as locals per iteration).

    Returns
    -------
    chunks : list of (chunk_name, OrderedDict[output_name, sym.Expr])
    leaves : set of free Symbols
        Order matches build_cascade_ir(chunks, leaves, ...).
    """
    if per_point:
        rho   = sym.Symbol('rho[pp]',   real=True, positive=True)
        mom_x = sym.Symbol('mom_x[pp]', real=True)
        mom_y = sym.Symbol('mom_y[pp]', real=True)
        mom_z = sym.Symbol('mom_z[pp]', real=True)
        E     = sym.Symbol('E[pp]',     real=True, positive=True)
        Bx    = sym.Symbol('Bx[pp]',    real=True)
        By    = sym.Symbol('By[pp]',    real=True)
        Bz    = sym.Symbol('Bz[pp]',    real=True)
        out_names = (
            'F_rho_out', 'F_mom_x_out', 'F_mom_y_out', 'F_mom_z_out',
            'F_E_out', 'F_Bx_out', 'F_By_out', 'F_Bz_out',
        )
    else:
        rho, mom_x, mom_y, mom_z, E, Bx, By, Bz = sym.symbols(
            'rho mom_x mom_y mom_z E Bx By Bz', real=True
        )
        out_names = ('F_rho', 'F_mom_x', 'F_mom_y', 'F_mom_z',
                     'F_E', 'F_Bx', 'F_By', 'F_Bz')
    gm1 = sym.Symbol('gamma_minus_1', real=True, positive=True)
    leaves = {rho, mom_x, mom_y, mom_z, E, Bx, By, Bz, gm1}

    # ---- L1: 1/rho (single inversion) ----
    L1_outputs = OrderedDict([
        ('rho_inv', 1 / rho),
    ])
    rho_inv = sym.Symbol('rho_inv')

    # ---- L2: primitives v_x, v_y, v_z (linear in mom and rho_inv) ----
    L2_outputs = OrderedDict([
        ('v_x', mom_x * rho_inv),
        ('v_y', mom_y * rho_inv),
        ('v_z', mom_z * rho_inv),
    ])
    v_x = sym.Symbol('v_x')
    v_y = sym.Symbol('v_y')
    v_z = sym.Symbol('v_z')

    # ---- L3: energy/pressure invariants (quadratic in primitives) ----
    # E_kin = rho * v^2 / 2 = (mom . v) / 2
    # E_mag = B^2 / 2
    # p     = (gamma - 1) * (E - E_kin - E_mag)
    # vdotB = v . B  (used by energy + induction fluxes)
    L3_outputs = OrderedDict([
        ('E_kin', sym.Rational(1, 2) * (mom_x * v_x + mom_y * v_y + mom_z * v_z)),
        ('E_mag', sym.Rational(1, 2) * (Bx * Bx + By * By + Bz * Bz)),
        ('vdotB', v_x * Bx + v_y * By + v_z * Bz),
    ])
    sE_kin = sym.Symbol('E_kin')
    sE_mag = sym.Symbol('E_mag')
    svdotB = sym.Symbol('vdotB')

    # ---- L4: pressures (linear in L3 outputs) ----
    L4_outputs = OrderedDict([
        ('p_gas',   gm1 * (E - sE_kin - sE_mag)),
        ('p_total', gm1 * (E - sE_kin - sE_mag) + sE_mag),    # p + B^2/2
        ('enthalpy', E + gm1 * (E - sE_kin - sE_mag) + sE_mag),  # E + p_total
    ])
    sp_total  = sym.Symbol('p_total')
    senthalpy = sym.Symbol('enthalpy')

    # ---- L5: flux components (quadratic in L2-L4) ----
    L5_outputs = OrderedDict([
        (out_names[0], mom_x),                                      # F_rho   = rho v_x
        (out_names[1], mom_x * v_x + sp_total - Bx * Bx),           # F_mom_x
        (out_names[2], mom_x * v_y - Bx * By),                      # F_mom_y
        (out_names[3], mom_x * v_z - Bx * Bz),                      # F_mom_z
        (out_names[4], senthalpy * v_x - Bx * svdotB),              # F_E
        (out_names[5], sym.Integer(0)),                             # F_Bx = 0
        (out_names[6], v_x * By - v_y * Bx),                        # F_By
        (out_names[7], v_x * Bz - v_z * Bx),                        # F_Bz
    ])

    chunks = [
        ('rho_inv',     L1_outputs),
        ('primitives',  L2_outputs),
        ('invariants',  L3_outputs),
        ('pressures',   L4_outputs),
        ('flux',        L5_outputs),
    ]

    return chunks, leaves


if __name__ == "__main__":
    chunks, leaves = mhd_flux_spec()
    print(f"leaves: {sorted(s.name for s in leaves)}")
    print()
    for name, outputs in chunks:
        print(f"chunk {name!r}:")
        for k, v in outputs.items():
            print(f"  {k} = {v}")
        print()
