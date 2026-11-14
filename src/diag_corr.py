"""Diagnostic: ceiling of the HMM field estimator when given the TRUE error pattern,
plus burst statistics of the correlated noise."""
import sys
import numpy as np
import pilot_corr as P
from decoders import make_decoder

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 120
P.init("bb144", 0.003, 0.9, 0.5, "bplsd0", 1)
S = P.S
H, O, noise = S["H"], S["O"], S["noise"]
f = dict(nominal=0, oracle=0, truth_informed=0, adaptive=0)
corr_t, corr_a, nerr, top = [], [], [], []
for seed in range(5000, 5000 + shots):
    rng = np.random.default_rng(seed)
    e, g = noise.sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8); o = O @ e % 2
    lg = np.log(np.clip(g, 1e-12, 1) / noise.p)
    e_nom = P._errors(S["nominal"], s)
    f["nominal"] += int(((O @ e_nom) % 2 != o).any())
    f["oracle"] += int(((O @ P._errors(make_decoder("bplsd0", H, O, np.clip(g, 1e-12, .49)), s)) % 2 != o).any())
    qt = np.clip(noise.posterior_priors(e), 1e-12, .49)
    f["truth_informed"] += int(((O @ P._errors(make_decoder("bplsd0", H, O, qt), s)) % 2 != o).any())
    qa = np.clip(noise.posterior_priors(e_nom), 1e-12, .49)
    f["adaptive"] += int(((O @ P._errors(make_decoder("bplsd0", H, O, qa), s)) % 2 != o).any())
    corr_t.append(np.corrcoef(np.log(qt / noise.p), lg)[0, 1]); corr_a.append(np.corrcoef(np.log(qa / noise.p), lg)[0, 1])
    nerr.append(int(e.sum())); top.append(float(np.sort(g)[-50:].mean()))
    # overlap between true and estimated fired mechanisms
print("failures per", shots, f)
print("field corr: truth-informed %.2f  adaptive %.2f" % (np.mean(corr_t), np.mean(corr_a)))
print("errors/shot: mean %.1f median %.0f max %d  (iid expectation %.1f)" % (np.mean(nerr), np.median(nerr), max(nerr), noise.p.sum()))
print("mean of top-50 conditional priors per shot: %.3f" % np.mean(top))
