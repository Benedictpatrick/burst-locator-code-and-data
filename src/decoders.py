"""Uniform wrappers around the open-source decoders compared in this study.

Every decoder is built from the same (check matrix, observable matrix, priors)
triple extracted from a Stim detector error model, and exposes

    decode(syndrome: np.ndarray[uint8]) -> np.ndarray[uint8]   (predicted observable flips)

so that all decoders see byte-identical inputs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import stim


# --------------------------------------------------------------- DEM -> matrices
def dem_columns(dem: stim.DetectorErrorModel):
    """Column keys (detector tuple, observable tuple) in the order used by dem_to_matrices."""
    return dem_to_matrices(dem, return_keys=True)


def dem_to_matrices(dem: stim.DetectorErrorModel, return_keys: bool = False):
    """Return (H, O, priors): detector-by-error and observable-by-error sparse
    binary matrices and per-error probabilities. Error mechanisms with identical
    (detector set, observable set) are merged, combining probabilities as
    independent events."""
    merged: dict[tuple, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - p) + p * (1 - q)
    keys = sorted(merged)
    if return_keys:
        return keys
    rows_h, cols_h, rows_o, cols_o = [], [], [], []
    for j, (dets, obs) in enumerate(keys):
        rows_h += dets
        cols_h += [j] * len(dets)
        rows_o += obs
        cols_o += [j] * len(obs)
    shape_h = (dem.num_detectors, len(keys))
    shape_o = (dem.num_observables, len(keys))
    H = sp.csr_matrix((np.ones(len(rows_h), dtype=np.uint8), (rows_h, cols_h)), shape=shape_h)
    O = sp.csr_matrix((np.ones(len(rows_o), dtype=np.uint8), (rows_o, cols_o)), shape=shape_o)
    priors = np.array([merged[k] for k in keys], dtype=np.float64)
    return H, O, priors


# -------------------------------------------------------------------- decoders
class _ErrorDecoder:
    """Base: subclasses implement _errors(syndrome) -> error vector."""

    def __init__(self, H, O, priors):
        self.H, self.O, self.priors = H, O, priors

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        e = np.asarray(self._errors(syndrome), dtype=np.uint8)
        return (self.O @ e) % 2


class BP(_ErrorDecoder):
    """Plain min-sum belief propagation (no post-processing)."""

    def __init__(self, H, O, priors, max_iter=1000, ms_scaling_factor=0.625):
        super().__init__(H, O, priors)
        from ldpc import BpDecoder

        self.d = BpDecoder(H, error_channel=list(priors), max_iter=max_iter, bp_method="minimum_sum",
                           ms_scaling_factor=ms_scaling_factor, schedule="parallel")

    def _errors(self, s):
        return self.d.decode(s)


class BPOSD(_ErrorDecoder):
    """BP + ordered statistics decoding with combination sweep (ldpc package)."""

    def __init__(self, H, O, priors, max_iter=10000, osd_order=10, osd_method="osd_cs",
                 bp_method="minimum_sum", ms_scaling_factor=0.625):
        super().__init__(H, O, priors)
        from ldpc import BpOsdDecoder

        self.d = BpOsdDecoder(H, error_channel=list(priors), max_iter=max_iter, bp_method=bp_method,
                              ms_scaling_factor=ms_scaling_factor, schedule="parallel",
                              osd_method=osd_method, osd_order=osd_order)

    def _errors(self, s):
        return self.d.decode(s)


class BPLSD(_ErrorDecoder):
    """BP + localized statistics decoding (ldpc package)."""

    def __init__(self, H, O, priors, max_iter=30, lsd_order=0, lsd_method="lsd_cs",
                 bp_method="minimum_sum", ms_scaling_factor=0.625):
        super().__init__(H, O, priors)
        from ldpc.bplsd_decoder import BpLsdDecoder

        self.d = BpLsdDecoder(H, error_channel=list(priors), max_iter=max_iter, bp_method=bp_method,
                              ms_scaling_factor=ms_scaling_factor, schedule="parallel",
                              lsd_method=lsd_method, lsd_order=lsd_order)

    def _errors(self, s):
        return self.d.decode(s)


class Relay(_ErrorDecoder):
    """Relay-BP (IBM, arXiv:2506.01779) via the relay-bp Rust package."""

    def __init__(self, H, O, priors, gamma0=0.125, pre_iter=80, num_sets=300, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66), stop_nconv=1, seed=0):
        super().__init__(H, O, priors)
        import relay_bp

        self.d = relay_bp.RelayDecoderF64(
            sp.csr_matrix(H), error_priors=np.asarray(priors, dtype=np.float64),
            gamma0=gamma0, pre_iter=pre_iter, num_sets=num_sets, set_max_iter=set_max_iter,
            gamma_dist_interval=tuple(gamma_dist_interval), stop_nconv=stop_nconv, seed=seed)

    def _errors(self, s):
        return self.d.decode(np.ascontiguousarray(s, dtype=np.uint8))


DECODERS = {
    "bp": (BP, {}),
    "bposd_cs10": (BPOSD, {}),  # 10,000 BP iterations, the setting used as baseline in arXiv:2506.01779
    "bposd_cs10_i1000": (BPOSD, {"max_iter": 1000}),
    "bposd_cs10_i100": (BPOSD, {"max_iter": 100}),
    "bplsd0": (BPLSD, {}),
    "relay1": (Relay, {"stop_nconv": 1, "num_sets": 300}),
    "relay5": (Relay, {"stop_nconv": 5, "num_sets": 600}),
}


def make_decoder(name: str, H, O, priors, **overrides):
    cls, kwargs = DECODERS[name]
    return cls(H, O, priors, **{**kwargs, **overrides})
