# An independent benchmark of open-source decoders on bivariate bicycle codes

Author: P. Benedict (independent researcher) — benedictpatrickjohn@gmail.com

Reproducible, like-for-like Monte Carlo comparison of open-source decoders for
quantum LDPC codes on memory experiments of the bivariate bicycle (BB) codes of
Bravyi et al. (Nature 627, 778, 2024), including the [[144,12,12]] gross code.
All decoders receive byte-identical syndromes sampled from the same circuits
under the same circuit-level noise model.

## Current study: locating correlated noise bursts from the syndrome

The manuscript is `paper/manuscript.md`. It tests syndrome-only estimators of a
hidden correlated noise field (model of arXiv:2611.10796) on the gross code.

| Path | Purpose |
|---|---|
| `src/corrnoise.py` | Correlated noise sampler and hidden-Markov field posterior |
| `src/robust.py` | Noisy-observation variant of the field posterior |
| `src/exp_corr.py` | Main experiment (Table 1, control) -> `results/corr_shots.csv` |
| `src/analyze_corr.py` | Summary and Figure 1 -> `results/corr_summary.csv`, `results/corr_fig.png` |
| `src/nn_burst.py` | Neural burst locator and GPU training (needs `torch`, CUDA build used: 2.6.0+cu124) |
| `src/eval_nn.py` | Network evaluation (Table 2) -> `results/nn_shots.csv` |
| `src/pilot_corr.py`, `src/diag_*.py` | Exploratory pilots that led to the final design |

```
python src/exp_corr.py 2000 14 results/corr_shots.csv
python src/analyze_corr.py results/corr_shots.csv
python src/nn_burst.py train 0.9 0.5 20 results/burst_net.pt
python src/eval_nn.py results/burst_net.pt 2000 8 results/nn_shots.csv
```

`results/raw.csv` holds an earlier, partial decoder benchmark (Relay-BP and
BP+LSD under independent noise) that is not part of the manuscript.

## Layout

| Path | Purpose |
|---|---|
| `src/bbcodes.py` | BB code construction and Stim memory circuits (own implementation) |
| `src/decoders.py` | Detector-error-model conversion and uniform decoder wrappers |
| `src/run_benchmark.py` | Parallel, resumable Monte Carlo runner (one CSV row per seeded batch) |
| `src/analyze.py` | Summary table (Wilson intervals, per-round rates) and figures |
| `src/smoke.py` | Quick end-to-end check and speed probe |
| `tests/` | Validation of codes and circuits |
| `results/` | Raw batch records and derived summaries |

## Reproduce

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python tests/test_bbcodes.py
.venv/Scripts/python src/run_benchmark.py --codes bb144 --ps 0.003 --decoders relay1 bplsd0 --out results/raw.csv
.venv/Scripts/python src/analyze.py results/raw.csv
```

`relay-bp` ships no Windows wheel; building it needs a Rust toolchain with a
working linker (MSVC build tools, or the `x86_64-pc-windows-gnullvm` toolchain
with LLVM-MinGW).

## Conventions

- Noise: uniform circuit-level depolarizing noise of strength p (two-qubit
  depolarizing after each CNOT, single-qubit depolarizing on idle data qubits,
  reset and measurement flips on ancillas). Data initialisation and final data
  readout are noiseless.
- Experiment: Z-basis memory, d syndrome rounds (d = code distance).
- A shot fails if any of the k logical observables is predicted wrongly.
- Default decoding uses Z-type detectors only ("XZ decoding").
