#!/usr/bin/env python3
# MMS test of Div_a_Grad_perp_weighted_upwind_flows in its two limits.
#
# The operator blends the face coefficient between the central average and
# the donor-cell (upwind) value, weighted per face by the field w:
#
#   a_face = (a_i + a_{i+1})/2 + (w_face * s / 2) * (a_{i+1} - a_i)
#
# with s a smoothed sign of the difference of f across the face. The
# continuum operator is Div(a Grad_perp f) for ANY smooth w -- the donor
# correction is O(dx) -- so both regimes share the same symbolic target and
# differ only in the expected convergence order:
#
#   w = 0 (diffusive regime): identical to Div_a_Grad_perp_flows.
#       Expected order 2. This is the no-regression guarantee that the
#       upwind machinery cannot perturb the unsaturated regime.
#   w = 1 (saturated regime): donor-cell everywhere (eps chosen << |df|).
#       Expected order 1 from the O(dx) donor dissipation. An UPPER bound
#       on the order is also enforced: order ~2 here would mean the upwind
#       branch is silently inactive (e.g. mis-scaled eps), which the usual
#       lower-bound-only check would not catch.
#
# Div_a_Grad_perp_flows runs alongside in the w = 0 case as a reference.
from job_functions import run_manufactured_solutions_test
from boutdata.mms import x, y, z
from perpendicular_laplacian import Div_a_Grad_perp_f, GeneralMetric
from sympy import sin, cos

# specify symbolic inputs
# contravariant metric coeffs: same orthogonal-family metric (g23 != 0) as
# orthogonal_test.py, so results are directly comparable
g11 = 1.1 + 0.16 * x * cos(y)
g22 = 0.9 + 0.09 * x * cos(y)
g33 = 1.2 + 0.2 * x * cos(y)
g12 = 0.0
g23 = 0.5 + 0.15 * x * cos(y)
g13 = 0.0
# f and a, as in orthogonal_test.py. The O(dx) donor error scales with the
# variation of a across faces, which dominates the O(dx^2) terms over the
# tested resolution range, so the saturated case measures a clean order ~1.
f = (x**2) * sin(y) * sin(z)
a = 1.0 + 0.1 * x**3 * sin(2 * y) * sin(2 * z)
# the metric object
metric = GeneralMetric(g11=g11, g12=g12, g13=g13, g22=g22, g23=g23, g33=g33)
# Div . ( a Grad_perp f ) -- the shared symbolic target for both regimes
div_a_grad_perp_f = Div_a_Grad_perp_f(a, f, metric=metric)

# inputs shared by both regime tests
base_input = {
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
    # << the smallest |df| resolved by the manufactured f on the tested
    # grids, so s = +-1 to high accuracy in the w = 1 case
    "upwind_eps": 1.0e-8,
    "interactive_plots": False,
}

# regime 1: w = 0 everywhere -> must behave exactly like the central
# operator, converging at order 2
diffusive_input = dict(
    base_input,
    test_dir="weighted_upwind_diffusive",
    w_string="0.0",
    differential_operator_list=[
        ["Div_a_Grad_perp_weighted_upwind_flows(a, f, w)", str(div_a_grad_perp_f), 2],
        # reference: the central operator this limit must reproduce
        ["Div_a_Grad_perp_flows(a, f)", str(div_a_grad_perp_f), 2],
    ],
)

# regime 2: w = 1 everywhere -> donor-cell coefficient on every face.
# Expected order 1; upper bound 1.7 catches a silently-central scheme
# (which would measure ~2)
saturated_input = dict(
    base_input,
    test_dir="weighted_upwind_saturated",
    w_string="1.0",
    differential_operator_list=[
        [
            "Div_a_Grad_perp_weighted_upwind_flows(a, f, w)",
            str(div_a_grad_perp_f),
            1,
            1.7,
        ],
    ],
)

success = True
output_message = ""
for test_input in [diffusive_input, saturated_input]:
    this_success, this_message = run_manufactured_solutions_test(test_input)
    success = success and this_success
    output_message += f"[{test_input['test_dir']}]\n{this_message}"

print(output_message)
if success:
    exit(0)
else:
    exit(1)
