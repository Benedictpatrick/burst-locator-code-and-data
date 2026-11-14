"""Evaluate the neural burst locator on fresh shots, against the same baselines.

  python src/eval_nn.py [net.pt] [shots] [workers] [out.csv]

Per shot (regenerated from its seed in every process, so all methods see the
same errors):  nominal | oracle | robust (hard EM, 3 passes) | soft (soft EM, 6
passes) | nn (network priors -> BP+LSD) | nn_refine (network priors, then one
robust estimate/redecode pass).
Evaluation seeds (>= 900000) are disjoint from the training stream.
"""
import csv
import multiprocessing as mp
import sys
import time

import numpy as np

import nn_burst as B
from decoders import make_decoder
from robust import robust_posterior

METHODS = ["nominal", "oracle", "robust", "soft", "nn", "nn_refine"]
S = {}


def sample(noise, H, O, seed):
    rng = np.random.default_rng(seed)
    e, g = noise.sample_errors(rng)
    return e, g, (H @ e % 2).astype(np.uint8), O @ e % 2


def init(theta, gamma):
    H, O, priors, noise = B.setup(theta, gamma)
    S.update(H=H, O=O, priors=priors, noise=noise, nominal=make_decoder("bplsd0", H, O, priors))


def shot(args):
    from ldpc import BpDecoder
    seed, q_nn = args
    H, O, noise = S["H"], S["O"], S["noise"]
    e, g, s, o = sample(noise, H, O, seed)

    def dec(q):
        return np.asarray(make_decoder("bplsd0", H, O, np.clip(q, 1e-12, 0.49))._errors(s), dtype=np.uint8)

    def marg(q):
        d = BpDecoder(H, error_channel=list(np.clip(q, 1e-12, 0.49)), max_iter=30, bp_method="minimum_sum",
                      ms_scaling_factor=0.625, schedule="parallel")
        d.decode(s)
        return 1.0 / (1.0 + np.exp(np.clip(np.asarray(d.log_prob_ratios, dtype=np.float64), -40, 40)))

    bad = lambda eh: int(((O @ eh) % 2 != o).any())
    out = {"seed": seed, "n_errors": int(e.sum())}
    e_nom = np.asarray(S["nominal"]._errors(s), dtype=np.uint8)
    out["nominal"] = bad(e_nom)
    out["oracle"] = bad(dec(g))
    eh = e_nom
    for _ in range(3):
        eh = dec(robust_posterior(noise, eh, 0.03, 0.3))
    out["robust"] = bad(eh)
    q = S["priors"]
    for _ in range(6):
        q = noise.posterior_priors_soft(marg(q))
    out["soft"] = bad(dec(q))
    e_nn = dec(q_nn.astype(np.float64))
    out["nn"] = bad(e_nn)
    out["nn_refine"] = bad(dec(robust_posterior(noise, e_nn, 0.03, 0.3)))
    return out


if __name__ == "__main__":
    import torch

    a = sys.argv[1:]
    path = a[0] if len(a) > 0 else "results/burst_net.pt"
    shots = int(a[1]) if len(a) > 1 else 2000
    workers = int(a[2]) if len(a) > 2 else 14
    out = a[3] if len(a) > 3 else "results/nn_shots.csv"
    ck = torch.load(path, weights_only=False)
    theta, gamma = ck["theta"], ck["gamma"]
    H, O, priors, noise = B.setup(theta, gamma)
    table = B.bin_table(noise)                      # (n_mech, K)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = B.Net(H.shape[0], noise.rounds * noise.nq).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    seeds = list(range(900000, 900000 + shots))
    t0 = time.time()
    X = np.stack([sample(noise, H, O, sd)[2] for sd in seeds]).astype(np.float32)
    Q = np.empty((shots, priors.size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, shots, 256):
            post = torch.softmax(net(torch.from_numpy(X[i:i + 256]).to(dev)), dim=-1).cpu().numpy()  # (b, site, K)
            Q[i:i + 256] = np.einsum("bjk,jk->bj", post[:, noise.flat, :], table)
    print(f"network inference on {shots} shots: {time.time() - t0:.1f}s total (incl. sampling)", flush=True)
    with open(out, "w", newline="") as f, mp.Pool(workers, initializer=init, initargs=(theta, gamma)) as pool:
        w = csv.DictWriter(f, fieldnames=["seed", "n_errors"] + METHODS)
        w.writeheader()
        tot = dict.fromkeys(METHODS, 0)
        for r in pool.imap(shot, zip(seeds, Q), chunksize=4):
            w.writerow(r)
            for m in METHODS:
                tot[m] += r[m]
    print(f"theta={theta} gamma={gamma} shots={shots} ({time.time() - t0:.0f}s)")
    for m in METHODS:
        print(f"  {m:10s} {tot[m]}")
    gap = tot["nominal"] - tot["oracle"]
    if gap > 0:
        print("  oracle gap closed: " + ", ".join(f"{m} {100 * (tot['nominal'] - tot[m]) / gap:.0f}%"
                                                  for m in ("robust", "soft", "nn", "nn_refine")))
