# DendroSym

A symbolic framework for generating efficient C++ code with the Dendro framework for complicated simulation systems.

## About DendroSym

To come.

## Installation

This repository is installed as a Python package. The project is not on PyPi.org, so the repository will need to be cloned and installed. It is recommended that this package is installed in a virtual environment and not on the system as a whole.

1. Clone the repository and enter the folder:

    ```sh
    git clone https://github.com/paralab/SymPyGR
    cd SymPyGR
    ```

    This step can be skipped if you follow Method 3 for installing the repository.
2. Install the repository:

    **Method 1** (easiest, but any changes require reinstallation)
    ```sh
    pip3 install .
    ```

    **Method 2** (any changes made to the files in the folder are immediately recognized and used)
    ```sh
    pip3 install -e .
    ```
    The `-e` flag tells `pip` that this is an "editable" install which allows active development of the project.

    **Method 3**
    ```sh
    pip3 install git+https://github.com/paralab/SymPyGR
    ```

    For this method, the `-e` flag can also be placed after the word `install` to mimic Method 2's editable mode.

    Installation of the repository should handle all dependencies as well.


## Usage

More information to come. A sample Python file will be included soon.

## Polynomial cascade code generation

`dendrosym.cascade` is the *polynomial cascade* code generator: you declare a system as **named
objects in evaluation order** (for GR: inverse metric → Christoffel symbols → Ricci → right-hand
sides), each layer is common-subexpression-eliminated **separately** and references earlier layers
by name, and the result is emitted as a scalar kernel or a SIMD (AVX2 / AVX-512) VEC-macro kernel,
optionally with the 6th-order derivative stencils fused into the kernel. Per-layer CSE keeps every
layer a small named blob instead of one flat body the compiler cannot vectorize; the same machinery
drives the `--cascade` option of the generated solvers (see a generated project's `CUSTOMIZE.md`).
The module depends only on `sympy`, `numpy` and `networkx` and imports without the rest of the
solver toolkit.

**Quickstart** (`dendrosym/cascade/examples/demo_user_api.py`):

```python
from collections import OrderedDict
import sympy as sym
from dendrosym.cascade import compile_system, report

a, x, y, z = sym.symbols("a x y z")
inv, g0, g1, g2, f0, f1, h0, q0 = sym.symbols("inv g0 g1 g2 f0 f1 h0 q0")

my_system = [                       # your derivation, in the order you would write it
    ("inv", OrderedDict(inv=1/a)),
    ("g",   OrderedDict(g0=x*x + y*y, g1=x*y + z*z, g2=y*z + x*z)),
    ("f",   OrderedDict(f0=g0*inv + g1*inv, f1=g1*inv + g2*inv)),
    ("h",   OrderedDict(h0=f0*f0 + f1*f1 + f0*f1)),
    ("q",   OrderedDict(q0=(x + y + z)**2 + (x - y)*(y - z))),
    ("out", OrderedDict(out0=h0 + q0*inv, out1=h0*g2 + g0*inv)),   # reference earlier outputs BY SYMBOL
]

code, ir = compile_system(my_system, {a, x, y, z}, out="toy_kernel.cpp", verbose=True)
report(ir)                                  # layers, widths, temps
code3, ir3 = compile_system(my_system, {a, x, y, z}, L=3)   # or pin the depth
```

Every knob lives in one dataclass: `compile_system(specs, leaves, options=CascadeOptions(simd="avx512",
L=7, fma_tree=True, inline_threshold=2, ...))` returns a complete standalone kernel; `build()` +
`emit_body()` give the bare VEC body for your own loop. Naming drives classification: `name[pp]`
is a per-point array, `name[N]` an indexed constant, a bare `name` a scalar the caller provides;
outputs are the objects whose names contain `_rhs` (or end in `_out`).

**Cost model before emitting:** `predict(specs, leaves, options)` builds the layered IR and
returns the source-level peak of simultaneously live values (the register-pressure predictor),
temps and layer widths — without writing a file.

**Command line:** `dendro-cascade <subcommand>` (or `python -m dendrosym.cascade`) — `compile
<spec.py>` for any user spec plus the worked systems (`bssn`, `emda`, `bssn-looped`), the layering
analysis (`autolayer`, `order-check`) and the object-level cost tools (`metrics`).
`tests/cascade/run_all.py [--slow]` is the test suite.

**Pinned sympy:** the emitted kernels depend on the sympy version (per-layer CSE output changes
between releases — the code is still correct to machine precision, but not byte-reproducible).
Regenerate under `requirements-cascade.txt` (sympy 1.13.3, `PYTHONHASHSEED=0`);
`scripts/regen_vikr_kernels.sh` is the byte-identical regression oracle.

**Used by the paper:** the kernels, ablations and cost model in *[CITATION PLACEHOLDER — David:
paper title / arXiv id at submission]* were generated with this module; the deployed
configuration is `simd="avx512", L=7, fused=True` with the BSSN `ssl`/`cahd` gauge terms.

---

## Legacy SymPyGR 

What follows is the original documentation of SymPyGR. The original code can be found in the `master-legacy` branch since the project has taken a new direction.

---

SymPy based framework for optimized code generation for BSSN formulation of Einstein equation for heterogeneous platforms. 

Dendro is an adaptive meshing framework that enables solving large-scale
computational problems on octree-refined meshes. The current version of dendro and
dendro_sym , handle adaptivity by decomposing the domain into a collection of small
regular blocks (uniformly refined), on which the code corresponding to the PDE are
automatically generated (C/C++) from the symbolic expressions. While the overall
framework works currently, there are several areas for improving performance and
portability and these are the possible research topics. Note that although dendro
supports distributed computing on large clusters, these improvements are only
required at the single-node and possibly single-thread level, as the blocks generated by
dendro are typically 16^3 to 256^3 in size. The topics listed below are tagged with the key
focus areas, gpu, openmp, simd, graph. 

* graph: Sympy produces an expression tree that is sufficiently simplified. But,
there are several repeated expressions within this tree that can be simplified by
factoring out these expressions and evaluating them only once. The main focus of
this task would be to develop algorithms for extracting common sub-expressions
within sympy. This would most likely be done in python.
* openmp, gpu Dendro produces a list of blocks that need to be scheduled across
threads and GPUs. We currently support simple scheduling via openmp.
Performance and load-balancing can be improved significantly by better
scheduling of the blocks. Of particular interest is to device methods to scheduled
blocks dynamically between GPUs and CPUs. This might have to be done in C/C++
to schedule the blocks. This C++ code can potentially be autogenerated from the
python code.
* gpu, simd As previously mentioned, the current implementation generates C++
code from the sympy expression trees. A very important contribution would be to
add python code to generate for different architecture targets, such as
1. default: pure C/C++ code
2. avx2: SIMD code targeting 256-bit wide AVX2 architectures
3. avx512: SIMD code targeting 512-bit wide architectures, such as the Xeon PHI
4. cuda: CUDA code targeting nVidia GPUs, possibly with options targeting
different generations.
5. openCL, openACC : and others.
