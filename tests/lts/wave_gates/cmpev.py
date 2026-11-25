#!/usr/bin/env python3
"""cmpev.py <dirA> <dirB>: compare DENDRO_EVAR_DUMP outputs by node position. Prints max |diff|, node counts, bitwise flag."""
import sys, glob, numpy as np
def load(d):
    pos, val = [], []
    for f in sorted(glob.glob(d + "/ev_rank*.bin"), key=lambda s: int(s.rsplit("rank", 1)[1][:-4])):
        r = int(f.rsplit("rank", 1)[1][:-4])
        raw = open(f, "rb").read()
        nloc, dof = (int(x) for x in np.frombuffer(raw[:8], dtype=np.uint32))
        v = np.frombuffer(raw[8:8 + 8 * nloc * dof], dtype=np.float64).reshape(dof, nloc)
        pr = open(f"{d}/ev_pos_rank{r}.bin", "rb").read()
        rec = np.dtype([("x", "<u8"), ("y", "<u8"), ("z", "<u8"), ("dup", "<u4"), ("lev", "<u4")])
        p = np.frombuffer(pr, dtype=rec)
        assert len(p) == nloc, (f, len(p), nloc)
        pos.append(p); val.append(v)
    p = np.concatenate(pos); v = np.concatenate(val, axis=1)
    keep = p["dup"] == 0
    p, v = p[keep], v[:, keep]
    key = np.lexsort((p["z"], p["y"], p["x"]))
    return p[key], v[:, key]
pa, va = load(sys.argv[1]); pb, vb = load(sys.argv[2])
if len(pa) != len(pb) or not all((pa[k] == pb[k]).all() for k in ("x", "y", "z")):
    print(f"MESH_DIFF nodes {len(pa)} vs {len(pb)}"); sys.exit(1)
d = np.abs(va - vb)
mx = float(d.max()) if d.size else 0.0
scale = float(np.abs(va).max())
print(f"nodes={len(pa)} dof={va.shape[0]} max|diff|={mx:.3e} max|u|={scale:.3e} rel={mx/scale if scale else 0:.3e} bitwise={'YES' if mx == 0 else 'NO'} nonfinite={int((~np.isfinite(va)).sum()+(~np.isfinite(vb)).sum())}")
sys.exit(0 if mx == 0 else 2)
