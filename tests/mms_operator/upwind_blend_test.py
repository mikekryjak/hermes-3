#!/usr/bin/env python3
"""MMS convergence test for Div_ab_Grad_perp_upwind_blend_flows.

The operator forms each X face flux as

    ab_face = (1 - w) * mean(a*b) + w * mean(a) * b_upwind

and uses the product a*b centrally for the Y and Z fluxes. Both branches of the
X blend converge to a*b as the grid refines (the upwind shift is O(dx)), so for
ANY weight field w the continuum limit is exactly

    Div( a*b Grad_perp(f) ) ,

i.e. the central operator with coefficient a*b. One symbolic target therefore
covers every regime, and only the expected convergence ORDER changes with w:

  1. w = 0            : order 2, and must be bit-identical to
                        Div_a_Grad_perp_flows with the same coefficient
  2. w = 1, donor on  : order 1 (donor-cell), with an upper bound on the fitted
                        order to catch a silently inactive upwind branch
  3. w = 1, donor off : order 2 (eps huge => s -> 0 => upwind value collapses
                        to the face average). Localises any failure in 2 to the
                        upwind term rather than to the blend itself
  4. w varying in x   : order 1, checks a non-constant weight is handled

The order-1 regimes set the y/z modulation of f to zero, so df/dy = df/dz = 0,
the (always second-order) Y and Z fluxes vanish identically, and the fitted
order measures the X scheme alone. The order-2 regimes keep the modulation, so
the Y/Z path and the product coefficient are exercised there.

The manufactured f is strictly monotone in x (df/dx > 0 everywhere), so the
upwind direction is uniform: this tests consistency, not the smoothness of the
sign changeover, which is a separate dynamical question. Each regime also runs
Div_a_Grad_perp_flows on the same geometry as a baseline, so a harness or
geometry mistake fails the baseline rather than being blamed on the operator.
"""
import os

from job_functions import run_manufactured_solutions_test
from perpendicular_laplacian import Div_a_Grad_perp_f, GeneralMetric
from boutdata.mms import x, y, z
from sympy import sin, cos

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

# manufactured fields. f is monotone-increasing in x on [0, 1]; yz_amp switches
# the y/z structure of f on and off (see module docstring).
a = 1.0 + 0.1 * x**3 * sin(2 * y) * sin(2 * z)
b = 1.5 + 0.5 * x**2 * sin(y) * sin(z)


def f_of(yz_amp):
    return (x**2 + x) * (1.5 + yz_amp * sin(y) * sin(z))


def extra_mesh_string(b_expr, w_expr, eps):
    """[mesh] entries for the operator's inputs beyond the standard a and f."""
    return f"""
   b = {b_expr}
   w = {w_expr}
   blend_eps = {eps}
   """


def check_identity(test_dir, ntest):
    """Confirm the w=0 blend is bit-identical to the central operator.

    Entry 0 is the blend and entry 1 the central operator, run on the same
    fields in the same job, so their dumped results must agree exactly.
    """
    try:
        import numpy as np
        import xarray as xr
    except ImportError as err:  # pragma: no cover
        return False, f"identity check could not import its dependencies: {err}"

    worst = 0.0
    for i in range(ntest):
        path = os.path.join(test_dir, f"slab-mms-test-{i}", "BOUT.0.nc")
        if not os.path.isfile(path):
            return False, f"identity check: missing {path}"
        with xr.open_dataset(path) as ds:
            if "result_0" not in ds or "result_1" not in ds:
                return False, f"identity check: result_0/result_1 not in {path}"
            diff = np.abs(ds["result_0"].values - ds["result_1"].values).max()
        worst = max(worst, float(diff))

    if worst == 0.0:
        return True, "w=0 identity: bit-identical to Div_a_Grad_perp_flows"
    return False, f"w=0 identity FAILED: max abs difference {worst:.3e}, expected 0"


common = {
    "ntest": 3,
    "ngrid": 20,
    "g11_string": str(g11),
    "g22_string": str(g22),
    "g33_string": str(g33),
    "g12_string": str(g12),
    "g13_string": str(g13),
    "g23_string": str(g23),
    "interactive_plots": False,
}

# regime -> (b expression, w expression, eps, yz_amp, expected order, max order)
# With w=0 the second coefficient is unused by the X scheme, so that regime sets
# b = 1 and the blend must then reproduce the central operator exactly, on the
# identical coefficient field. The other regimes carry a non-trivial b.
REGIMES = {
    "blend_w0_identity": (1.0, 0.0, 1.0e-2, 0.5, 2, None),
    "blend_w1_upwind": (b, 1.0, 1.0e-3, 0.0, 1, 1.7),
    "blend_w1_nodonor": (b, 1.0, 1.0e9, 0.5, 2, None),
    "blend_w_varying": (b, 0.3 + 0.5 * x, 1.0e-3, 0.0, 1, 1.7),
}

if __name__ == "__main__":
    all_success = True
    messages = []
    for test_dir, (b_expr, w_expr, eps, yz_amp, order, max_order) in REGIMES.items():
        f = f_of(yz_amp)
        # Continuum limit of the blended operator, for any w
        target = Div_a_Grad_perp_f(a * b_expr, f, metric=metric)
        # Baseline: the central operator on the same geometry with coefficient a
        central = Div_a_Grad_perp_f(a, f, metric=metric)

        entry = ["Div_ab_Grad_perp_upwind_blend_flows(a, b, w, f)", str(target), order]
        if max_order is not None:
            entry.append(max_order)

        test_input = dict(common)
        test_input["a_string"] = str(a)
        test_input["f_string"] = str(f)
        test_input["test_dir"] = test_dir
        test_input["extra_mesh_string"] = extra_mesh_string(b_expr, w_expr, eps)
        test_input["differential_operator_list"] = [
            entry,
            ["Div_a_Grad_perp_flows(a, f)", str(central), 2],
        ]
        success, message = run_manufactured_solutions_test(test_input)

        # With b = 1 the two entries share a coefficient, so at w = 0 the blend
        # must reproduce the central operator to the last bit, not just to
        # second order.
        if test_dir == "blend_w0_identity":
            id_success, id_message = check_identity(test_dir, common["ntest"])
            success = success and id_success
            message = message + "\n" + id_message

        all_success = all_success and success
        messages.append(f"=== {test_dir} ===\n{message}")

    print("\n".join(messages))
    exit(0 if all_success else 1)
