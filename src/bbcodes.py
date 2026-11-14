"""Bivariate bicycle (BB) codes and their syndrome-extraction memory circuits.

Code construction and the depth-7 CNOT schedule follow
Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, Nature 627, 778 (2024).

Everything here is an independent implementation; correctness is checked by
Stim (deterministic detectors/observables) and by the tests in tests/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

# name -> (l, m, A exponents, B exponents); a term is ("x", power) or ("y", power)
BB_CODES = {
    "bb72": (6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]),
    "bb90": (15, 3, [("x", 9), ("y", 1), ("y", 2)], [("x", 0), ("x", 2), ("x", 7)]),
    "bb108": (9, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]),
    "bb144": (12, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]),
    "bb288": (12, 12, [("x", 3), ("y", 2), ("y", 7)], [("y", 3), ("x", 1), ("x", 2)]),
}
# Published [[n, k, d]] for reference / tests.
BB_PARAMS = {
    "bb72": (72, 12, 6),
    "bb90": (90, 8, 10),
    "bb108": (108, 8, 10),
    "bb144": (144, 12, 12),
    "bb288": (288, 12, 18),
}


# ----------------------------------------------------------------- GF(2) tools
def gf2_row_reduce(mat: np.ndarray):
    """Return (reduced matrix, pivot columns) of a binary matrix over GF(2)."""
    m = (np.array(mat, dtype=np.uint8) & 1).copy()
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        others = np.nonzero(m[:, c])[0]
        others = others[others != r]
        m[others] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def gf2_rank(mat: np.ndarray) -> int:
    return len(gf2_row_reduce(mat)[1])


def gf2_nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis (as rows) of the right nullspace of mat over GF(2)."""
    red, pivots = gf2_row_reduce(mat)
    cols = mat.shape[1]
    free = [c for c in range(cols) if c not in set(pivots)]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            if red[r, f]:
                basis[i, p] = 1
    return basis


# ------------------------------------------------------------------- the code
@dataclass
class BBCode:
    name: str
    l: int
    m: int
    A: list  # three n/2 x n/2 permutation matrices
    B: list
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray  # k x n logical X operators
    lz: np.ndarray  # k x n logical Z operators

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def k(self) -> int:
        return self.lz.shape[0]


def _logicals(h_commute: np.ndarray, h_stab: np.ndarray) -> np.ndarray:
    """Operators in ker(h_commute) that are independent modulo rowspace(h_stab)."""
    ker = gf2_nullspace(h_commute)
    stack = gf2_row_reduce(h_stab)[0]
    rank = stack.shape[0]
    out = []
    for v in ker:
        cand = np.vstack([stack, v[None, :]])
        r = gf2_rank(cand)
        if r > rank:
            stack, rank = gf2_row_reduce(cand)[0], r
            out.append(v)
    return np.array(out, dtype=np.uint8)


def build_code(name: str) -> BBCode:
    l, m, a_terms, b_terms = BB_CODES[name]
    shift_l = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    shift_m = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(shift_l, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), shift_m)

    def term(t):
        base = x if t[0] == "x" else y
        return np.linalg.matrix_power(base.astype(np.int64), t[1]).astype(np.uint8) & 1

    A = [term(t) for t in a_terms]
    B = [term(t) for t in b_terms]
    a = (A[0] + A[1] + A[2]) % 2
    b = (B[0] + B[1] + B[2]) % 2
    hx = np.hstack([a, b]).astype(np.uint8)
    hz = np.hstack([b.T, a.T]).astype(np.uint8)
    assert not ((hx @ hz.T) % 2).any(), "CSS condition violated"
    lz = _logicals(hx, hz)
    lx = _logicals(hz, hx)
    return BBCode(name, l, m, A, B, hx, hz, lx, lz)


# ---------------------------------------------------------------- the circuit
_SX = ["idle", 1, 4, 3, 5, 0, 2]
_SZ = [3, 5, 0, 1, 2, 4, "idle"]


def memory_circuit(
    code: BBCode,
    p: float,
    rounds: int,
    basis: str = "Z",
    noisy_boundaries: bool = False,
) -> stim.Circuit:
    """Memory experiment for a BB code under uniform circuit-level depolarizing noise.

    Noise (strength p everywhere): two-qubit depolarizing after each CNOT,
    single-qubit depolarizing on idling data qubits, flip after each ancilla
    reset, flip of each ancilla measurement outcome.

    By default the data-qubit initialisation and final data readout are
    noiseless (the convention of Bravyi et al.: noisy cycles closed by an
    ideal readout). Set noisy_boundaries=True to add flips of strength p there.

    Qubit layout: X checks [0, n/2), left data [n/2, n), right data
    [n, 3n/2), Z checks [3n/2, 2n).
    Detectors carry coordinates (check index, 0 for Z-type / 1 for X-type, round).
    """
    assert basis in ("Z", "X")
    n2 = code.n // 2
    xc = list(range(0, n2))
    dl = list(range(n2, 2 * n2))
    dr = list(range(2 * n2, 3 * n2))
    zc = list(range(3 * n2, 4 * n2))
    data = dl + dr

    def nb(mat, i):
        (cols,) = np.nonzero(mat[i])
        assert cols.size == 1
        return int(cols[0])

    # neighbours: direction 0..2 -> left data, 3..5 -> right data
    xn = [[dl[nb(code.A[d], i)] for d in range(3)] + [dr[nb(code.B[d], i)] for d in range(3)] for i in range(n2)]
    zn = [[dl[nb(code.B[d].T, i)] for d in range(3)] + [dr[nb(code.A[d].T, i)] for d in range(3)] for i in range(n2)]

    def cycle() -> stim.Circuit:
        c = stim.Circuit()
        for t in range(7):
            pairs = []
            if t == 0:
                c.append("RX", xc)
                c.append("Z_ERROR", xc, p)
            if _SZ[t] != "idle":
                for i in range(n2):
                    pairs += [zn[i][_SZ[t]], zc[i]]  # data controls, Z check is target
            if _SX[t] != "idle":
                for i in range(n2):
                    pairs += [xc[i], xn[i][_SX[t]]]  # X check controls, data is target
            if t == 6:
                c.append("M", zc, p)
            c.append("CX", pairs)
            c.append("DEPOLARIZE2", pairs, p)
            busy = set(pairs)
            idle = [q for q in data if q not in busy]
            if idle:
                c.append("DEPOLARIZE1", idle, p)
            c.append("TICK")
        c.append("MX", xc, p)
        c.append("R", zc)
        c.append("X_ERROR", zc, p)
        c.append("DEPOLARIZE1", data, p)
        c.append("TICK")
        return c

    circ = stim.Circuit()
    circ.append("R" if basis == "Z" else "RX", data)
    if noisy_boundaries:
        circ.append("X_ERROR" if basis == "Z" else "Z_ERROR", data, p)
    circ.append("R", zc)
    circ.append("X_ERROR", zc, p)
    circ.append("TICK")

    # per cycle the record gains n2 Z-check results followed by n2 X-check results
    per = 2 * n2
    for r in range(rounds):
        circ += cycle()
        for i in range(n2):
            z_now = stim.target_rec(-per + i)
            x_now = stim.target_rec(-n2 + i)
            if r == 0:
                if basis == "Z":
                    circ.append("DETECTOR", [z_now], [i, 0, r])
                else:
                    circ.append("DETECTOR", [x_now], [i, 1, r])
            else:
                circ.append("DETECTOR", [z_now, stim.target_rec(-2 * per + i)], [i, 0, r])
                circ.append("DETECTOR", [x_now, stim.target_rec(-per - n2 + i)], [i, 1, r])

    if noisy_boundaries:
        circ.append("X_ERROR" if basis == "Z" else "Z_ERROR", data, p)
    circ.append("M" if basis == "Z" else "MX", data)
    nd = len(data)  # data results are the last nd records, in order dl + dr
    h = code.hz if basis == "Z" else code.hx
    logical = code.lz if basis == "Z" else code.lx
    for i in range(n2):
        targets = [stim.target_rec(-nd + int(q)) for q in np.nonzero(h[i])[0]]
        # last check result of matching type: Z block then X block precede the data block
        off = -nd - per + i if basis == "Z" else -nd - n2 + i
        targets.append(stim.target_rec(off))
        circ.append("DETECTOR", targets, [i, 0 if basis == "Z" else 1, rounds])
    for j in range(logical.shape[0]):
        targets = [stim.target_rec(-nd + int(q)) for q in np.nonzero(logical[j])[0]]
        circ.append("OBSERVABLE_INCLUDE", targets, j)
    return circ


def keep_basis_detectors(circuit: stim.Circuit, basis: str = "Z") -> stim.Circuit:
    """Drop detectors of the other type (the 'XZ decoding' setting: decode a
    Z-memory with Z-type detectors only)."""
    want = 0 if basis == "Z" else 1
    out = stim.Circuit()
    for inst in circuit:
        if inst.name == "DETECTOR" and int(inst.gate_args_copy()[1]) != want:
            continue
        out.append(inst)
    return out
