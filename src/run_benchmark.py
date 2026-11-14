"""Monte Carlo benchmark of decoders on BB-code memory experiments.

Each (code, p, decoder) point is sampled in independent seeded batches spread
over worker processes until max_errors logical failures or max_shots shots.
Every batch is appended to a CSV, so runs can be interrupted and resumed and
the raw record is fully auditable.

Example:
  python src/run_benchmark.py --codes bb144 --ps 0.003 0.004 --decoders relay1 bplsd0 \
      --max-shots 200000 --max-errors 100 --out results/raw.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np

from bbcodes import BB_PARAMS, build_code, keep_basis_detectors, memory_circuit
from decoders import dem_to_matrices, make_decoder

FIELDS = ["code", "basis", "detectors", "rounds", "p", "decoder", "seed", "shots", "errors",
          "decode_seconds", "timestamp"]
_state = {}


def _init(code_name, p, rounds, basis, detectors, decoder_name):
    code = build_code(code_name)
    circ = memory_circuit(code, p, rounds, basis)
    if detectors == "basis":
        circ = keep_basis_detectors(circ, basis)
    H, O, priors = dem_to_matrices(circ.detector_error_model())
    _state["circ"] = circ
    _state["dec"] = make_decoder(decoder_name, H, O, priors)


def _batch(args):
    seed, shots = args
    dets, obs = _state["circ"].compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
    dec = _state["dec"]
    errors = 0
    t0 = time.perf_counter()
    for s, o in zip(dets, obs):
        errors += int((dec.decode(s) != o).any())
    return seed, shots, errors, time.perf_counter() - t0


def _done(path):
    """Existing totals per point and the set of seeds already used."""
    tot, seeds = {}, {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["code"], r["basis"], r["detectors"], int(r["rounds"]), float(r["p"]), r["decoder"])
                s, e = tot.get(key, (0, 0))
                tot[key] = (s + int(r["shots"]), e + int(r["errors"]))
                seeds.setdefault(key, set()).add(int(r["seed"]))
    return tot, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", default=["bb144"])
    ap.add_argument("--ps", nargs="+", type=float, required=True)
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--detectors", default="basis", choices=["basis", "all"],
                    help="basis: decode with same-basis detectors only (XZ decoding); all: full DEM (XYZ)")
    ap.add_argument("--rounds", type=int, default=0, help="0 = code distance")
    ap.add_argument("--max-shots", type=int, default=100000)
    ap.add_argument("--max-errors", type=int, default=100)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="results/raw.csv")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    tot, used = _done(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for code in a.codes:
            rounds = a.rounds or BB_PARAMS[code][2]
            for p in a.ps:
                for dn in a.decoders:
                    key = (code, a.basis, a.detectors, rounds, p, dn)
                    shots, errors = tot.get(key, (0, 0))
                    seeds_used = used.get(key, set())
                    if shots >= a.max_shots or errors >= a.max_errors:
                        print(f"skip {key}: {errors}/{shots} already", flush=True)
                        continue
                    t0 = time.time()
                    next_seed = (max(seeds_used) + 1) if seeds_used else 1000

                    def gen():
                        s = next_seed
                        while True:
                            yield (s, a.batch)
                            s += 1

                    with mp.Pool(a.workers, initializer=_init,
                                 initargs=(code, p, rounds, a.basis, a.detectors, dn)) as pool:
                        # ordered imap: batches are consumed in seed order, so the stopping
                        # rule cannot favour batches that happened to decode quickly
                        for seed, n, e, secs in pool.imap(_batch, gen()):
                            shots += n
                            errors += e
                            w.writerow(dict(code=code, basis=a.basis, detectors=a.detectors, rounds=rounds,
                                            p=p, decoder=dn, seed=seed, shots=n, errors=e,
                                            decode_seconds=f"{secs:.4f}", timestamp=int(time.time())))
                            f.flush()
                            if shots >= a.max_shots or errors >= a.max_errors:
                                pool.terminate()
                                break
                    print(f"{code} p={p} {dn}: {errors}/{shots} = {errors / shots:.3e} "
                          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
