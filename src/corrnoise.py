"""Spatially correlated noise and syndrome-only estimation of the hidden field.

Noise model (Dangi & Chatterjee, arXiv:2611.10796): for each syndrome round t
an independent stationary Gaussian AR(1) chain J[t, k] over the qubit index k,
    J[t,0] ~ N(0,1),  J[t,k+1] = rho J[t,k] + sqrt(1-rho^2) xi,   rho = exp(-1/L), L = n_qubits^gamma.
Error mechanism j, attached to a representative (round t_j, qubit q_j), fires
independently given J with probability
    g_j(J) = Phi((Phi^-1(p_j) + theta J[t_j, q_j]) / sqrt(1 - theta^2)),
which keeps every marginal E_J[g_j] = p_j.

Estimator (this work): given a hard estimate e_hat of which mechanisms fired,
each round is a hidden Markov chain over k with Bernoulli emissions g_j(J_k);
forward-backward on a discretised J gives the exact posterior over J_k and
hence posterior-mean priors E[g_j | e_hat] for a second decoding pass.
"""
from __future__ import annotations

import numpy as np
import stim
from scipy.stats import norm


def mechanism_locations(circuit: stim.Circuit, keys, ticks_per_round: int, rounds: int):
    """Representative (round, qubit) for each merged DEM column.

    keys: list of (detector tuple, observable tuple) in column order, as produced
    by decoders.dem_columns."""
    index = {k: i for i, k in enumerate(keys)}
    t_of = np.zeros(len(keys), dtype=np.int64)
    q_of = np.zeros(len(keys), dtype=np.int64)
    seen = np.zeros(len(keys), dtype=bool)
    for ex in circuit.explain_detector_error_model_errors(reduce_to_one_representative_error=True):
        dets, obs = set(), set()
        for term in ex.dem_error_terms:
            t = term.dem_target
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        col = index.get((tuple(sorted(dets)), tuple(sorted(obs))))
        if col is None or seen[col]:
            continue
        loc = ex.circuit_error_locations[0]
        if loc.flipped_pauli_product:
            q = loc.flipped_pauli_product[0].gate_target.value
        else:  # measurement flip
            q = loc.flipped_measurement.observable[0].gate_target.value
        t_of[col] = min(max((loc.tick_offset - 1) // ticks_per_round, 0), rounds - 1)
        q_of[col] = q
        seen[col] = True
    assert seen.all(), f"{(~seen).sum()} mechanisms without a location"
    return t_of, q_of


class CorrelatedNoise:
    def __init__(self, priors, t_of, q_of, rounds, n_qubits, theta, gamma, grid=41, span=4.0):
        self.p = np.asarray(priors, dtype=np.float64)
        self.t_of, self.q_of = t_of, q_of
        self.rounds, self.nq = rounds, n_qubits
        self.theta = theta
        self.rho = float(np.exp(-1.0 / n_qubits ** gamma))
        self.zp = norm.ppf(self.p)
        self.flat = t_of * n_qubits + q_of  # site index of each mechanism
        # discretised field for inference
        self.levels = np.linspace(-span, span, grid)
        w = norm.pdf(self.levels)
        self.pi0 = w / w.sum()
        sd = np.sqrt(1 - self.rho ** 2)
        T = norm.pdf((self.levels[None, :] - self.rho * self.levels[:, None]) / max(sd, 1e-9))
        self.T = T / T.sum(axis=1, keepdims=True)
        # g[j, level]: firing probability of mechanism j at each field level
        s = np.sqrt(1 - theta ** 2)
        self.g_lv = norm.cdf((self.zp[:, None] + theta * self.levels[None, :]) / s)
        self.log_g = np.log(np.clip(self.g_lv, 1e-300, 1))
        self.log_1mg = np.log1p(-np.clip(self.g_lv, 0, 1 - 1e-16))
        # per-site sum of log(1-g) over all mechanisms (the "nothing fired" emission)
        self.base = np.zeros((rounds * n_qubits, grid))
        np.add.at(self.base, self.flat, self.log_1mg)

    # ------------------------------------------------------------ sampling
    def sample_field(self, rng):
        J = np.empty((self.rounds, self.nq))
        J[:, 0] = rng.standard_normal(self.rounds)
        sd = np.sqrt(1 - self.rho ** 2)
        xi = rng.standard_normal((self.rounds, self.nq))
        for k in range(1, self.nq):
            J[:, k] = self.rho * J[:, k - 1] + sd * xi[:, k]
        return J

    def conditional_priors(self, J):
        """Oracle priors g_j(J)."""
        return norm.cdf((self.zp + self.theta * J.reshape(-1)[self.flat]) / np.sqrt(1 - self.theta ** 2))

    def sample_errors(self, rng):
        J = self.sample_field(rng)
        g = self.conditional_priors(J)
        return (rng.random(g.size) < g).astype(np.uint8), g

    # ----------------------------------------------------------- inference
    def posterior_priors(self, e_hat):
        """Posterior-mean firing probabilities E[g_j(J) | e_hat] by forward-backward
        over the qubit index within each round."""
        return self.posterior_priors_soft(np.asarray(e_hat, dtype=np.float64))

    def posterior_priors_soft(self, m):
        """As posterior_priors, with per-mechanism firing estimates m in [0, 1]
        (hard decisions or soft marginals) entering the emissions as expected counts."""
        G = self.levels.size
        emis = self.base.copy()
        np.add.at(emis, self.flat, np.asarray(m, dtype=np.float64)[:, None] * (self.log_g - self.log_1mg))
        emis = emis.reshape(self.rounds, self.nq, G)
        emis -= emis.max(axis=2, keepdims=True)
        E = np.exp(emis)
        alpha = np.empty_like(E)
        beta = np.empty_like(E)
        a = self.pi0[None, :] * E[:, 0]
        alpha[:, 0] = a / a.sum(axis=1, keepdims=True)
        for k in range(1, self.nq):
            a = (alpha[:, k - 1] @ self.T) * E[:, k]
            alpha[:, k] = a / a.sum(axis=1, keepdims=True)
        beta[:, -1] = 1.0
        for k in range(self.nq - 2, -1, -1):
            b = (beta[:, k + 1] * E[:, k + 1]) @ self.T.T
            beta[:, k] = b / b.sum(axis=1, keepdims=True)
        post = alpha * beta
        post /= post.sum(axis=2, keepdims=True)
        post = post.reshape(self.rounds * self.nq, G)
        return np.einsum("jg,jg->j", post[self.flat], self.g_lv)
