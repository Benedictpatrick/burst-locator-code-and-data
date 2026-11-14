"""Quick end-to-end smoke test + speed probe. Usage: python src/smoke.py [code] [p] [shots]"""
import sys
import time

import numpy as np

from bbcodes import BB_PARAMS, build_code, keep_basis_detectors, memory_circuit
from decoders import DECODERS, dem_to_matrices, make_decoder

name = sys.argv[1] if len(sys.argv) > 1 else "bb72"
p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
shots = int(sys.argv[3]) if len(sys.argv) > 3 else 300
only = sys.argv[4].split(",") if len(sys.argv) > 4 else list(DECODERS)

code = build_code(name)
rounds = BB_PARAMS[name][2]
circ = keep_basis_detectors(memory_circuit(code, p, rounds, "Z"), "Z")
dem = circ.detector_error_model()
H, O, priors = dem_to_matrices(dem)
print(f"{name} p={p} rounds={rounds}  H={H.shape}  nnz={H.nnz}  max col weight={H.sum(axis=0).max()}")
dets, obs = circ.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
for dn in only:
    t0 = time.perf_counter()
    dec = make_decoder(dn, H, O, priors)
    t_build = time.perf_counter() - t0
    fails, times = 0, []
    for s, o in zip(dets, obs):
        t0 = time.perf_counter()
        pred = dec.decode(s)
        times.append(time.perf_counter() - t0)
        fails += int((pred != o).any())
    times = np.array(times) * 1e3
    print(f"{dn:11s} build {t_build:5.1f}s  fails {fails:4d}/{shots}  mean {times.mean():8.2f} ms  "
          f"median {np.median(times):8.2f} ms  max {times.max():9.1f} ms")
