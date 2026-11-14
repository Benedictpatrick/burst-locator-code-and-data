import sys, itertools
import numpy as np
import multiprocessing as mp
import pilot_corr as P
from decoders import make_decoder
from robust import robust_posterior

CFGS = [(eps, dl) for eps in (1e-3, 3e-3, 1e-2, 3e-2) for dl in (0.3, 0.6)]
ITERS = (1, 3, 6)

def init():
    P.init("bb144", 0.003, 0.9, 0.5, "bplsd0", 1)

def shot(seed):
    S = P.S; H, O, noise = S["H"], S["O"], S["noise"]
    rng = np.random.default_rng(seed)
    e, g = noise.sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8); o = O @ e % 2
    def dec(q): return P._errors(make_decoder("bplsd0", H, O, np.clip(q, 1e-12, .49)), s)
    bad = lambda eh: int(((O @ eh) % 2 != o).any())
    e_nom = P._errors(S["nominal"], s)
    out = {"nominal": bad(e_nom), "oracle": bad(dec(g))}
    for eps, dl in CFGS:
        eh = e_nom
        for it in range(1, max(ITERS) + 1):
            eh = dec(robust_posterior(noise, eh, eps, dl))
            if it in ITERS: out[f"eps{eps}_d{dl}_it{it}"] = bad(eh)
    return out

if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 280
    with mp.Pool(14, initializer=init) as pool:
        res = pool.map(shot, range(5000, 5000 + shots), chunksize=2)
    print("failures per", shots)
    for k in res[0]: print(f"  {k:24s} {sum(r[k] for r in res)}")
