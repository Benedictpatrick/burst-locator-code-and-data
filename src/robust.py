"""Robust field posterior: the decoder's error estimate is modelled as a noisy
observation of the true errors (false-positive rate eps, miss rate delta), so
isolated mis-attributed errors do not force spurious field spikes."""
import numpy as np


def robust_posterior(noise, m, eps, delta):
    """m: hard (0/1) or soft (in [0,1]) per-mechanism firing estimate."""
    G = noise.levels.size
    h = np.clip(noise.g_lv * (1 - delta) + (1 - noise.g_lv) * eps, 1e-300, 1 - 1e-12)
    lh, l1h = np.log(h), np.log1p(-h)
    emis = np.zeros((noise.rounds * noise.nq, G))
    np.add.at(emis, noise.flat, l1h + np.asarray(m, dtype=np.float64)[:, None] * (lh - l1h))
    emis = emis.reshape(noise.rounds, noise.nq, G)
    emis -= emis.max(axis=2, keepdims=True)
    E = np.exp(emis)
    alpha = np.empty_like(E)
    beta = np.empty_like(E)
    a = noise.pi0[None, :] * E[:, 0]
    alpha[:, 0] = a / a.sum(1, keepdims=True)
    for k in range(1, noise.nq):
        a = (alpha[:, k - 1] @ noise.T) * E[:, k]
        alpha[:, k] = a / a.sum(1, keepdims=True)
    beta[:, -1] = 1.0
    for k in range(noise.nq - 2, -1, -1):
        b = (beta[:, k + 1] * E[:, k + 1]) @ noise.T.T
        beta[:, k] = b / b.sum(1, keepdims=True)
    post = alpha * beta
    post /= post.sum(2, keepdims=True)
    return np.einsum("jg,jg->j", post.reshape(-1, G)[noise.flat], noise.g_lv)
