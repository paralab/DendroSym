"""mkpar.py <out.toml> KEY=VAL ... : wave gate par from the generated sample (sections ignored; keys are unique)."""
import re, sys
s = open("/home/denv/research/gen_lts_trial/wave_lts/wave_parameters.sample.toml").read()
base = dict(maxdepth=8, mindepth=4, compd_min="[-16.0,-16.0,-16.0]", compd_max="[16.0,16.0,16.0]",
            rk_time_begin="0.0", rk_time_end="2.0", io_output_freq=1000000, time_step_output_freq=1,
            checkpt_freq=1000000, remesh_test_freq=4, dendro_grain_sz=20, blk_min_x="-16.0",
            blk_min_y="-16.0", blk_min_z="-16.0", blk_max_x="16.0", blk_max_y="16.0", blk_max_z="16.0", id_type=2)
over = dict(a.split("=", 1) for a in sys.argv[2:])
base.update(over)
for k, v in base.items():
    s, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v}", s, count=1, flags=re.M)
    assert n == 1, k
open(sys.argv[1], "w").write(s)
