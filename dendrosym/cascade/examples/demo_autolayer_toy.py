#!/usr/bin/env python3
"""Toy walkthrough of automatic layer selection on a 6-object system.

The system (deliberately shaped like a miniature BSSN):
    inv  = 1/a                     one expensive scalar, used EVERYWHERE
    g    = 3 quadratic combos      used soon, then again much later (a skip)
    f    = g * inv contractions    consumed immediately by h
    h    = f^2 terms               consumed immediately by out1
    q    = big local scratch       used ONLY by out1 (dies immediately)
    out  = 2 outputs mixing h, q, g (the skip lands here), and inv

Run:  python demo_autolayer_toy.py
Shows the carry table (what crossing each boundary costs), the DP's cuts
for L = 2 and 3, and the true measured peak of every candidate.
"""
from collections import OrderedDict
import sympy as sym

from dendrosym.cascade.builder import build_cascade_ir
from dendrosym.cascade.autolayer import (explode, finest_build, dependency_carry,
                               dp_partition, merge)

a, x, y, z = sym.symbols("a x y z")
leaves = {a, x, y, z}

# BY-SYMBOL style (the recommended spec convention): downstream expressions
# reference earlier outputs through their NAMES, so dependencies are exact.
# (By-value style -- pasting the full trees -- breaks under sympy's
# Add-flattening: the subtree vanishes, matching fails, and the machinery
# silently recomputes. Correct but slower, and it smears the carry table.)
inv, g0, g1, g2, f0, f1, h0, q0 = sym.symbols("inv g0 g1 g2 f0 f1 h0 q0")

specs = [
    ("inv",  OrderedDict(inv=1/a)),
    ("g",    OrderedDict(g0=x*x + y*y, g1=x*y + z*z, g2=y*z + x*z)),
    ("f",    OrderedDict(f0=g0*inv + g1*inv, f1=g1*inv + g2*inv)),
    ("h",    OrderedDict(h0=f0*f0 + f1*f1 + f0*f1)),
    ("q",    OrderedDict(q0=(x + y + z)**2 + (x - y)*(y - z))),
    ("out",  OrderedDict(out0=h0 + q0*inv,
                         out1=h0*g2 + g0*inv)),  # g0,g2 reused: the skip
]

objects = explode(specs)
fin = finest_build(objects, leaves)
W, width, last_use = dependency_carry(objects, fin)

print("object   width  last used by        (skips show as last_use far ahead)")
for i, (name, outs) in enumerate(objects):
    print(f"  {name:6s} {width[i]:4d}   {objects[last_use[i]][0]}")

print("\ncarry W(b) across each possible boundary:")
for b in range(1, len(objects)):
    crossers = [objects[i][0] for i in range(b) if last_use[i] >= b]
    print(f"  before {objects[b][0]:6s} W={W[b]:2d}   carried: {crossers}")

temps = [c.n_temps for c in fin.chunks]
for L in (2, 3):
    cuts = dp_partition(objects, W, temps, L)
    layers = merge(objects, cuts)
    res = build_cascade_ir(layers, leaves, cse_prefix="AUTO_")
    print(f"\nL={L}: cuts before {[objects[c][0] for c in cuts]}"
          f" -> layers {[c.name for c in res.chunks]}")
