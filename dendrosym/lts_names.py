"""Names the generated LTS code takes from dendrolib, in one table.

The library is still being renamed under review (ws_tidy, naming sweep due
2026-11-20); every template reads these through `lts.<key>` so a rename is a
one-line edit here. Keys are the contract-document names
(dendrolib_dfvk_ws_tidy/docs/lts_solver_contract.md).
"""

LTS_NAMES = {
    # headers: the stepper and the contract helpers
    "stepper_header": "enuts.h",
    "helpers_header": "lts.h",
    # types and free functions in namespace ts
    "stepper": "ExplicitNUTS",
    "options": "LtsOptions",
    "block_view": "BlockView",
    "clock": "Clock",
    "cadence": "Cadence",
    "enforced_nodes": "for_each_enforced_node",
    "level_offset": "level_offset",
    "apply_partitioning": "apply_partitioning",
    "remesh_if_needed": "remesh_if_needed",
    "checkpoint": "checkpoint",
    "tableau_supported": "lts_tableau_supported",
    "options_valid": "lts_options_valid",
    # stepper members that carry the stage vector across a restart
    "write_carried_g": "write_carried_g",
    "read_carried_g": "read_carried_g",
    # Clock members
    "clock_due": "due",
    "clock_seed": "seed",
    "clock_resume": "resume",
    "clock_slot": "checkpoint_slot",
}
