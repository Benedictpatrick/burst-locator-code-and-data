"""Soft-EM variant: BP posterior marginals (not hard decisions) drive the HMM field estimate."""
import sys
import numpy as np
from ldpc import BpDecoder
import pilot_corr as P
from decoders import make_decoder

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 120
method = sys.argv[2] if len(sys.argv) > 2 else "product_sum"
bp_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 30
em_iters = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,3,6").split(",")]
P.init("bb144", 0.003, 0.9, 0.5, "bplsd0", 1)
S = P.S
H, O, noise = S["H"], S["O"], S["noise"]

def marginals(priors, s):
    d = BpDecoder(H, error_channel=list(priors), max_iter=bp_iters, bp_method=method,
                  ms_scaling_factor=0.625, schedule="parallel")
    d.decode(s)
    llr = np.clip(np.asarray(d.log_prob_ratios, dtype=np.float64), -40, 40)
    return 1.0 / (1.0 + np.exp(llr))

def soft_posterior(m):
    # same HMM, but emissions use expected counts m_j instead of hard e_j
    G = noise.levels.size
    emis = noise.base.copy()
    np.add.at(emis, noise.flat, m[:, None] * (noise.log_g - noise.log_1mg))
    emis = emis.reshape(noise.rounds, noise.nq, G); emis -= emis.max(axis=2, keepdims=True)
    E = np.exp(emis); alpha = np.empty_like(E); beta = np.empty_like(E)
    a = noise.pi0[None, :] * E[:, 0]; alpha[:, 0] = a / a.sum(1, keepdims=True)
    for k in range(1, noise.nq):
        a = (alpha[:, k - 1] @ noise.T) * E[:, k]; alpha[:, k] = a / a.sum(1, keepdims=True)
    beta[:, -1] = 1.0
    for k in range(noise.nq - 2, -1, -1):
        b = (beta[:, k + 1] * E[:, k + 1]) @ noise.T.T; beta[:, k] = b / b.sum(1, keepdims=True)
    post = alpha * beta; post /= post.sum(2, keepdims=True)
    return np.einsum("jg,jg->j", post.reshape(-1, G)[noise.flat], noise.g_lv)

fails = {"nominal": 0, "oracle": 0, **{f"soft{k}": 0 for k in em_iters}}
corr = {k: [] for k in em_iters}
for seed in range(5000, 5000 + shots):
    rng = np.random.default_rng(seed)
    e, g = noise.sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8); o = O @ e % 2
    lg = np.log(np.clip(g, 1e-12, 1) / noise.p)
    fails["nominal"] += int(((O @ P._errors(S["nominal"], s)) % 2 != o).any())
    fails["oracle"] += int(((O @ P._errors(make_decoder("bplsd0", H, O, np.clip(g, 1e-12, .49)), s)) % 2 != o).any())
    q = noise.p.copy()
    for it in range(1, max(em_iters) + 1):
        q = np.clip(soft_posterior(marginals(q, s)), 1e-12, .49)
        if it in em_iters:
            fails[f"soft{it}"] += int(((O @ P._errors(make_decoder("bplsd0", H, O, q), s)) % 2 != o).any())
            corr[it].append(np.corrcoef(np.log(q / noise.p), lg)[0, 1])
print(f"{method} bp_iters={bp_iters} failures per {shots}: {fails}")
print("field corr:", {k: round(float(np.mean(v)), 2) for k, v in corr.items()})
