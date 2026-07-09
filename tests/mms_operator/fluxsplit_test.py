#!/usr/bin/env python3
"""MMS convergence test for Div_a_Grad_perp_fluxsplit_flows.

The operator rebuilds the X face flux as
    F = Dunl_f * G * (N_face + R*N_donor) / (1+R)^2 ,
    R = Dunl_f * g_reg / avth_f ,
    g_reg = sqrt( g_phys^2 C^2 / (g_phys^2 + C^2) + F^2 ) ,  g_phys = sqrt(g11)|df|/dx ,
with central Y/Z fluxes using the separate coefficient `a`.

The continuum limit of the X flux is  Dunl*Nch/(1+R) * J*g11*df/dx  for ANY R
(the donor shift is O(dx)), so one exact symbolic target covers all regimes;
only the expected convergence ORDER differs:

  1. diffusive   (avth huge, R->0):        order 2, must equal the central form
  2. regularised (moderate R, donor OFF):  order 2, tests the g_reg/R machinery
  3. saturated   (R>>1, donor ON):         order 1, max-order bound 1.7 catches
                                           a silently-inactive donor branch

The manufactured f is strictly monotone in x (df/dx > 0 everywhere) so the
symbolic |df/dx| needs no abs() and the donor sign is uniform: this tests
consistency, not the sign-flap smoothness (which is a separate, dynamical
question). Baseline entry: Div_a_Grad_perp_flows re-verified on the same
geometry, so a harness/geometry mistake fails the baseline, not just the new
operator.
"""
from job_functions import run_manufactured_solutions_test
from perpendicular_laplacian import Div_a_Grad_perp_f, GeneralMetric
from boutdata.mms import x, y, z
from sympy import sin, cos, sqrt, diff

# ---------------------------------------------------------------------------
# geometry (orthogonal: g12 = g13 = 0, as required by the operator; g23 != 0
# exercises the central Y cross-terms)
g11 = 1.1 + 0.16 * x * cos(y)
g22 = 0.9 + 0.09 * x * cos(y)
g33 = 1.2 + 0.2 * x * cos(y)
g12 = 0.0
g23 = 0.5 + 0.15 * x * cos(y)
g13 = 0.0
metric = GeneralMetric(g11=g11, g12=g12, g13=g13, g22=g22, g23=g23, g33=g33)

# manufactured fields. f monotone-increasing in x on [0, 1].
f = (x**2 + x) * (1.5 + 0.5 * sin(y) * sin(z))
a = 1.0 + 0.1 * x**3 * sin(2 * y) * sin(2 * z)
Nch = 1.5 + 0.5 * x**2 * sin(y) * sin(z)
Dunl = 1.0 + 0.1 * x**3 * sin(2 * y) * sin(2 * z)


def fluxsplit_target(a, Nch, Dunl, avth, f, C, F, metric):
    """Continuum limit of Div_a_Grad_perp_fluxsplit_flows (see module docstring).

    X flux uses the face-split effective coefficient Dunl*Nch/(1+R); Y and Z
    fluxes are the central ones of Div_a_Grad_perp_f with coefficient a
    (orthogonal metric assumed: g12 = g13 = 0)."""
    J = metric.J
    dfdx = diff(f, metric.x)
    dfdy = diff(f, metric.y)
    dfdz = diff(f, metric.z)

    gphys = sqrt(metric.g11) * dfdx  # dfdx > 0 by construction, no abs needed
    greg = sqrt(gphys**2 * C**2 / (gphys**2 + C**2) + F**2)
    R = Dunl * greg / avth
    flux_x = (Dunl * Nch / (1 + R)) * J * metric.g11 * dfdx

    df3 = dfdz - (metric.g_23 / metric.g_22) * dfdy
    flux_y = a * J * metric.g23 * df3
    flux_z = a * J * metric.g33 * df3

    return (1 / J) * (
        diff(flux_x, metric.x) + diff(flux_y, metric.y) + diff(flux_z, metric.z)
    )


def extra_mesh_string(avth, eps, C, F):
    """[mesh] entries for the operator's auxiliary inputs (written with the
    same **->^ substitution as the standard entries)."""
    return f"""
   Nch = {Nch}
   Dunl = {Dunl}
   avth = {avth}
   fluxsplit_eps = {eps}
   fluxsplit_grad_ceiling = {C}
   fluxsplit_grad_floor = {F}
   """


# central target for the baseline entry
central = Div_a_Grad_perp_f(a, f, metric=metric)

common = {
    "ntest": 3,
    "ngrid": 20,
    "a_string": str(a),
    "f_string": str(f),
    "g11_string": str(g11),
    "g22_string": str(g22),
    "g33_string": str(g33),
    "g12_string": str(g12),
    "g13_string": str(g13),
    "g23_string": str(g23),
    "interactive_plots": False,
}

# regime -> (avth, eps, C, F, expected order, optional max order, a_scale)
# a_scale shrinks the (always order-2) central Y/Z contribution so the X-flux
# error dominates the fit. NB deep saturation (avth << 1) is a BAD donor probe:
# the donor error carries the donor weight R/(1+R)^2 ~ 1/R, and the whole
# X-flux scales with avth, so the O(dx) donor signal drowns under the O(dx^2)
# bulk at reachable resolutions (this exact insensitivity produced a spurious
# order-2.1 "fail-is-really-pass" in the first version of this test). The
# donor weight is MAXIMAL at R ~ 1, so the saturated regime pins R there.
REGIMES = {
    "fluxsplit_diffusive": (1.0e8, 1.0e-2, 1.0e6, 0.0, 2, None, 1.0),
    "fluxsplit_regularised": (1.0, 1.0e9, 0.5, 1.0e-3, 2, None, 1.0),
    "fluxsplit_saturated": (2.0, 1.0e-3, 1.0e3, 0.0, 1, 1.7, 0.1),
    # discriminator: same R~O(1) raw-gradient regime, donor OFF. Order 2 here
    # localises any saturated-regime failure to the donor term; failure here
    # too implicates the raw (unceilinged) R itself.
    "fluxsplit_saturated_nodonor": (2.0, 1.0e9, 1.0e3, 0.0, 2, None, 0.1),
}

if __name__ == "__main__":
    all_success = True
    messages = []
    for test_dir, (avth, eps, C, F, order, max_order, a_scale) in REGIMES.items():
        a_r = a_scale * a
        target = fluxsplit_target(a_r, Nch, Dunl, avth, f, C, F, metric)
        central_r = Div_a_Grad_perp_f(a_r, f, metric=metric)
        entry = ["Div_a_Grad_perp_fluxsplit_flows(a, f)", str(target), order]
        if max_order is not None:
            entry.append(max_order)
        test_input = dict(common)
        test_input["a_string"] = str(a_r)
        test_input["test_dir"] = test_dir
        test_input["extra_mesh_string"] = extra_mesh_string(avth, eps, C, F)
        test_input["differential_operator_list"] = [
            entry,
            # baseline: central operator on the same geometry/fields
            ["Div_a_Grad_perp_flows(a, f)", str(central_r), 2],
        ]
        success, message = run_manufactured_solutions_test(test_input)
        all_success = all_success and success
        messages.append(f"=== {test_dir} ===\n{message}")

    print("\n".join(messages))
    exit(0 if all_success else 1)
