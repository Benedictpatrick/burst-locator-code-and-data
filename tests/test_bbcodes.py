"""Validation of the BB code construction and memory circuits.

Run:  python -m pytest tests -q      (or: python tests/test_bbcodes.py)
"""
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bbcodes import BB_PARAMS, build_code, gf2_rank, keep_basis_detectors, memory_circuit  # noqa: E402


def test_code_parameters():
    for name, (n, k, _d) in BB_PARAMS.items():
        code = build_code(name)
        assert code.n == n
        assert code.k == k, (name, code.k)
        assert code.lx.shape[0] == k
        # logicals commute with the opposite-type stabilisers
        assert not ((code.hx @ code.lz.T) % 2).any()
        assert not ((code.hz @ code.lx.T) % 2).any()
        # logical X and Z pair up non-degenerately
        assert gf2_rank((code.lx @ code.lz.T) % 2) == k
        # weight-6 checks
        assert (code.hx.sum(axis=1) == 6).all() and (code.hz.sum(axis=1) == 6).all()


def test_circuits_are_deterministic_without_noise():
    for name in ("bb72", "bb144"):
        code = build_code(name)
        for basis in ("Z", "X"):
            circ = memory_circuit(code, p=0.0, rounds=3, basis=basis)
            dets, obs = circ.compile_detector_sampler().sample(64, separate_observables=True)
            assert not dets.any(), (name, basis)
            assert not obs.any(), (name, basis)


def test_dem_exists_and_counts():
    code = build_code("bb72")
    rounds = 6
    circ = memory_circuit(code, p=1e-3, rounds=rounds, basis="Z")
    # raises if any detector or observable is non-deterministic
    dem = circ.detector_error_model(decompose_errors=False)
    n2 = code.n // 2
    assert dem.num_observables == code.k
    assert dem.num_detectors == n2 * rounds + n2 * (rounds - 1) + n2
    zonly = keep_basis_detectors(circ, "Z")
    assert zonly.detector_error_model().num_detectors == n2 * (rounds + 1)


def test_single_data_error_gives_code_syndrome():
    """An X error on one data qubit before a noiseless run must fire exactly the
    Z checks given by the corresponding column of H_Z."""
    code = build_code("bb72")
    n2 = code.n // 2
    base = memory_circuit(code, p=0.0, rounds=2, basis="Z")
    for q in (0, 17, 40, 71):
        circ = stim.Circuit()
        injected = False
        for inst in base:
            circ.append(inst)
            if not injected and inst.name == "TICK":
                circ.append("X_ERROR", [n2 + q], 1.0)
                injected = True
        circ = keep_basis_detectors(circ, "Z")
        dets, _ = circ.compile_detector_sampler().sample(1, separate_observables=True)
        first_round = dets[0][:n2].astype(np.uint8)
        assert (first_round == code.hz[:, q]).all(), q


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
