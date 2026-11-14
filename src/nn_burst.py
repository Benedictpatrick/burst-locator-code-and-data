"""Neural burst locator: syndrome -> posterior over the hidden field at every site.

Input   detection events of one shot (flat vector of Z-type detectors)
Output  for each site (round, qubit) a categorical distribution over K field bins
Trained on simulator samples (syndrome, true field), cross-entropy loss. The
decoder priors are then q_j = sum_k P(bin k at site of j) * E[g_j | bin k].

  python src/nn_burst.py train [theta] [gamma] [minutes] [out.pt]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from bbcodes import BB_PARAMS, build_code, keep_basis_detectors, memory_circuit
from corrnoise import CorrelatedNoise, mechanism_locations
from decoders import dem_columns, dem_to_matrices

# field bins: the link function is negligible below J ~ 1.5, so resolution goes to the upper tail
EDGES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
K = len(EDGES) + 1


def setup(theta, gamma, p=0.003):
    code = build_code("bb144")
    rounds = BB_PARAMS["bb144"][2]
    circ = keep_basis_detectors(memory_circuit(code, p, rounds, "Z"), "Z")
    dem = circ.detector_error_model()
    H, O, priors = dem_to_matrices(dem)
    t_of, q_of = mechanism_locations(circ, dem_columns(dem), ticks_per_round=8, rounds=rounds)
    noise = CorrelatedNoise(priors, t_of, q_of, rounds, circ.num_qubits, theta, gamma)
    return H, O, priors, noise


def bin_table(noise, n=400000, seed=7):
    """E[g_j(J) | J in bin k] for every mechanism: (n_mech, K)."""
    from scipy.stats import norm
    J = np.random.default_rng(seed).standard_normal(n)
    b = np.digitize(J, EDGES)
    s = np.sqrt(1 - noise.theta ** 2)
    out = np.zeros((noise.p.size, K))
    for k in range(K):
        Jk = J[b == k][:2000]
        out[:, k] = norm.cdf((noise.zp[:, None] + noise.theta * Jk[None, :]) / s).mean(axis=1)
    return out


class Net(nn.Module):
    def __init__(self, n_det, n_site, width=3072):
        super().__init__()
        self.n_site = n_site
        self.body = nn.Sequential(nn.Linear(n_det, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_site * K))

    def forward(self, x):
        return self.body(x).view(-1, self.n_site, K)


def make_batch(noise, H, rng, n):
    """Vectorised over the batch: n fields, n error patterns, n syndromes at once."""
    from scipy.special import ndtr
    R, nq = noise.rounds, noise.nq
    J = np.empty((n, R, nq), dtype=np.float32)
    J[:, :, 0] = rng.standard_normal((n, R), dtype=np.float32)
    sd = np.float32(np.sqrt(1 - noise.rho ** 2))
    xi = rng.standard_normal((n, R, nq), dtype=np.float32)
    for k in range(1, nq):
        J[:, :, k] = noise.rho * J[:, :, k - 1] + sd * xi[:, :, k]
    Jf = J.reshape(n, R * nq)
    g = ndtr((noise.zp[None, :] + noise.theta * Jf[:, noise.flat]) / np.sqrt(1 - noise.theta ** 2))
    E = (rng.random(g.shape, dtype=np.float32) < g).astype(np.float32)
    X = np.asarray((H @ E.T).T % 2, dtype=np.float32)
    Y = np.digitize(Jf, EDGES).astype(np.int64)
    return X, Y


class GpuSampler:
    """Same noise model as CorrelatedNoise.sample_errors, generated on the GPU in float32."""

    def __init__(self, noise, H, dev, seed):
        self.R, self.nq, self.rho, self.theta = noise.rounds, noise.nq, noise.rho, noise.theta
        self.zp = torch.tensor(noise.zp, dtype=torch.float32, device=dev)
        self.flat = torch.tensor(noise.flat, dtype=torch.long, device=dev)
        self.Ht = torch.tensor(np.asarray(H.todense()), dtype=torch.float32, device=dev).T.contiguous()
        self.edges = torch.tensor(EDGES, dtype=torch.float32, device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.dev = dev

    @torch.no_grad()
    def batch(self, n):
        J = torch.empty((n, self.R, self.nq), device=self.dev)
        xi = torch.randn((n, self.R, self.nq), device=self.dev, generator=self.gen)
        sd = float(np.sqrt(1 - self.rho ** 2))
        J[:, :, 0] = xi[:, :, 0]
        for k in range(1, self.nq):
            J[:, :, k] = self.rho * J[:, :, k - 1] + sd * xi[:, :, k]
        Jf = J.reshape(n, -1)
        g = torch.special.ndtr((self.zp[None, :] + self.theta * Jf[:, self.flat]) / float(np.sqrt(1 - self.theta ** 2)))
        E = (torch.rand(g.shape, device=self.dev, generator=self.gen) < g).float()
        X = torch.remainder(E @ self.Ht, 2.0)
        Y = torch.bucketize(Jf, self.edges)
        return X, Y, E.sum(1).mean().item()


def train(theta=0.9, gamma=0.5, minutes=20.0, out="results/burst_net.pt", batch=256, seed=0):
    H, O, priors, noise = setup(theta, gamma)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = GpuSampler(noise, H, dev, seed)
    net = Net(H.shape[0], noise.rounds * noise.nq).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    # class weights: high-field bins are rare but are the ones that matter
    w = torch.tensor([1, 2, 4, 8, 8, 8, 8], dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)
    t0, step = time.time(), 0
    while time.time() - t0 < minutes * 60:
        x, y, mean_err = sampler.batch(batch)
        if step == 0:
            print(f"sampler check: mean errors/shot in first batch {mean_err:.1f} "
                  f"(i.i.d. expectation {noise.p.sum():.1f})", flush=True)
        logits = net(x)
        loss = loss_fn(logits.reshape(-1, K), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                hi = y >= 3  # J > 2: burst sites
                pred_hi = logits.argmax(-1) >= 3
                rec = (pred_hi & hi).sum().item() / max(hi.sum().item(), 1)
                prec = (pred_hi & hi).sum().item() / max(pred_hi.sum().item(), 1)
            print(f"step {step} ({time.time() - t0:.0f}s, {step * batch} shots) loss {loss.item():.4f} "
                  f"burst recall {rec:.2f} precision {prec:.2f}", flush=True)
    torch.save({"state": net.state_dict(), "theta": theta, "gamma": gamma, "steps": step, "batch": batch,
                "seed": seed, "edges": EDGES}, out)
    print(f"saved {out} after {step} steps ({step * batch} training shots)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "train":
        train(float(a[1]) if len(a) > 1 else 0.9, float(a[2]) if len(a) > 2 else 0.5,
              float(a[3]) if len(a) > 3 else 20.0, a[4] if len(a) > 4 else "results/burst_net.pt")
