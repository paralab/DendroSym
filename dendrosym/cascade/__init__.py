"""dendrosym.cascade -- the polynomial-cascade code generator.

Spec of named objects in evaluation order -> per-layer CSE IR -> scalar / AVX2 /
AVX-512 C++ kernels. Ported verbatim from the vikr research repo (codegen/); the
checked-in vikr kernels regenerate byte-identical from this package
(scripts/regen_vikr_kernels.sh). See findings/cascade_api_guide.md (vikr) for the
spec contract, and `dendrosym.cascade.api` for the one-call entry points.
"""
from dendrosym.cascade.builder import (  # noqa: F401
    CascadeBuilder, CascadeResult, ChunkResult, build_cascade_ir,
    flatten_sym33, flatten_vec3, flatten_tensor3, flatten_scalar,
    expand_integer_pows, E_I, E_IJ, E_IJ_SYM,
)
from dendrosym.cascade.emit import (  # noqa: F401
    emit_kernel_function_cpp, emit_jax_kernel, emit_body, emit_standalone_kernel,
    macro_block, scalar_macro_block, undef_block, kernel_signature,
)
from dendrosym.cascade.options import CascadeOptions  # noqa: F401
from dendrosym.cascade.api import compile_system, predict, report, build, warn_if_unpinned  # noqa: F401
