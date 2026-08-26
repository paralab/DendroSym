#!/usr/bin/env python3
"""What a USER writes: their own file, one import, their system, run it.

This is the toy system of demo_autolayer_toy.py expressed the way an
adopter would: no knowledge of chunks, builders, or emitters required.
"""
from collections import OrderedDict
import sympy as sym

from dendrosym.cascade.api import compile_system, report

# --- my system: inputs and named intermediates, in derivation order ---
a, x, y, z = sym.symbols("a x y z")
inv, g0, g1, g2, f0, f1, h0, q0 = sym.symbols("inv g0 g1 g2 f0 f1 h0 q0")

my_system = [
    ("inv", OrderedDict(inv=1/a)),
    ("g",   OrderedDict(g0=x*x + y*y, g1=x*y + z*z, g2=y*z + x*z)),
    ("f",   OrderedDict(f0=g0*inv + g1*inv, f1=g1*inv + g2*inv)),
    ("h",   OrderedDict(h0=f0*f0 + f1*f1 + f0*f1)),
    ("q",   OrderedDict(q0=(x + y + z)**2 + (x - y)*(y - z))),
    ("out", OrderedDict(out0=h0 + q0*inv, out1=h0*g2 + g0*inv)),
]

# --- compile: boundaries chosen automatically, depth chosen automatically ---
code, ir = compile_system(my_system, {a, x, y, z}, out="toy_kernel.cpp",
                          verbose=True)
report(ir)

# --- or: pin the depth ---
code3, ir3 = compile_system(my_system, {a, x, y, z}, L=3)
print("\nwith L=3:")
report(ir3)
