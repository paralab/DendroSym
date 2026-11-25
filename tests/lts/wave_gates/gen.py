import sys, runpy, os
sys.path.insert(0, "/home/denv/research/SymPyGR")
sys.path.insert(0, "/home/denv/research/SymPyGR_lts")
import dendrosym
ns = runpy.run_path("/home/denv/research/SymPyGR/wave_eqn_config.py")
cfg = ns["dendroConfigs"]
out, flag = sys.argv[1], sys.argv[2] == "on"
if flag:
    cfg.enable_lts = True
dendrosym.run(cfg, default_output_dir=out)
