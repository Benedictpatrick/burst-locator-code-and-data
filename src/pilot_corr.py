"""Pilot: can the hidden correlated-noise field be estimated from the syndrome alone?

Compares, on identical shots of the gross code under correlated noise:
  nominal    decoder uses the i.i.d. marginals p_j (prior mismatch)
  oracle     decoder uses g_j(J) for the realised field J (upper bound, not a decoder)
  adaptive   stage 1: nominal decode -> e_hat; HMM posterior of the field given e_hat
             -> re-weighted priors; stage 2: decode again (repeat `iters` times)

  python src/pilot_corr.py [shots] [theta] [gamma] [p] [final_decoder] [iters] [workers]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

from bbcodes import BB_PARAMS, build_code, keep_basis_detectors, memory_circuit
from corrnoise import CorrelatedNoise, mechanism_locations
from decoders import dem_columns, dem_to_matrices, make_decoder

S = {}


def init(code_name, p, theta, gamma, final, iters):
    code = build_code(code_name)
    rounds = BB_PARAMS[code_name][2]
    circ = keep_basis_detectors(memory_circuit(code, p, rounds, "Z"), "Z")
    dem = circ.detector_error_model()
    H, O, priors = dem_to_matrices(dem)
    t_of, q_of = mechanism_locations(circ, dem_columns(dem), ticks_per_round=8, rounds=rounds)
    S.update(H=H, O=O, priors=priors, final=final, iters=iters,
             noise=CorrelatedNoise(priors, t_of, q_of, rounds, circ.num_qubits, theta, gamma),
             nominal=make_decoder(final, H, O, priors))


def _errors(dec, s):
    return np.asarray(dec._errors(s), dtype=np.uint8)


def shot(seed):
    rng = np.random.default_rng(seed)
    H, O, noise, final = S["H"], S["O"], S["noise"], S["final"]
    e, g = noise.sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8)
    o = O @ e % 2
    out = {}
    t0 = time.perf_counter()
    e_nom = _errors(S["nominal"], s)
    out["nominal"] = int(((O @ e_nom) % 2 != o).any())
    out["t_nominal"] = time.perf_counter() - t0
    out["oracle"] = int(((O @ _errors(make_decoder(final, H, O, np.clip(g, 1e-12, 0.49)), s)) % 2 != o).any())
    t0 = time.perf_counter()
    e_hat = e_nom
    for _ in range(S["iters"]):
        q = np.clip(noise.posterior_priors(e_hat), 1e-12, 0.49)
        e_hat = _errors(make_decoder(final, H, O, q), s)
    out["adaptive"] = int(((O @ e_hat) % 2 != o).any())
    out["t_adaptive"] = time.perf_counter() - t0
    # how well does the estimate track the truth? (correlation of log-odds shifts)
    out["corr"] = float(np.corrcoef(np.log(q / noise.p), np.log(np.clip(g, 1e-12, 1) / noise.p))[0, 1])
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    shots = int(a[0]) if len(a) > 0 else 200
    theta = float(a[1]) if len(a) > 1 else 0.9
    gamma = float(a[2]) if len(a) > 2 else 0.5
    p = float(a[3]) if len(a) > 3 else 0.003
    final = a[4] if len(a) > 4 else "bplsd0"
    iters = int(a[5]) if len(a) > 5 else 1
    workers = int(a[6]) if len(a) > 6 else 14
    t0 = time.time()
    with mp.Pool(workers, initializer=init, initargs=("bb144", p, theta, gamma, final, iters)) as pool:
        res = pool.map(shot, range(5000, 5000 + shots), chunksize=4)
    f = {k: sum(r[k] for r in res) for k in ("nominal", "oracle", "adaptive")}
    print(f"bb144 p={p} theta={theta} gamma={gamma} decoder={final} iters={iters} shots={shots} "
          f"({time.time() - t0:.0f}s)")
    print(f"  failures  nominal {f['nominal']}   oracle {f['oracle']}   adaptive {f['adaptive']}")
    gap = f["nominal"] - f["oracle"]
    if gap > 0:
        print(f"  oracle gap closed: {100 * (f['nominal'] - f['adaptive']) / gap:.0f}%")
    print(f"  mean field-estimate correlation with truth: {np.mean([r['corr'] for r in res]):.2f}")
    print(f"  mean time/shot  nominal {1e3 * np.mean([r['t_nominal'] for r in res]):.0f} ms   "
          f"adaptive stage {1e3 * np.mean([r['t_adaptive'] for r in res]):.0f} ms")
