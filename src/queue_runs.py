"""Run the strengthening batch back to back, logging to results/queue.log.

Each step is skipped if its output file already exists, so the queue can be
restarted after an interruption.

  python src/queue_runs.py
"""
import os
import subprocess
import sys
import time

PY = sys.executable
SETTINGS = [(0.9, 0.25, 12), (0.7, 0.5, 12), (0.7, 0.25, 12)]  # theta, gamma, training minutes
EVAL_SHOTS = "2000"
RUN_MAIN = False  # the 10,000-shot rerun of exp_corr.py takes about 2 hours on 8 cores
WORKERS = "8"
LOG = "results/queue.log"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def run(name, out, cmd):
    if os.path.exists(out):
        log(f"skip {name}: {out} exists")
        return
    log(f"start {name}")
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"{'done' if rc == 0 else 'FAILED (exit %d)' % rc} {name} in {(time.time() - t0) / 60:.1f} min")
    if rc != 0 and os.path.exists(out):
        os.replace(out, out + ".failed")


if __name__ == "__main__":
    os.makedirs("results/v2", exist_ok=True)
    log("queue started")
    for th, ga, mins in SETTINGS:
        tag = f"t{th}_g{ga}"
        run(f"train {tag}", f"results/v2/net_{tag}.pt",
            [PY, "src/nn_burst.py", "train", str(th), str(ga), str(mins), f"results/v2/net_{tag}.pt"])
    for th, ga, _ in SETTINGS:
        tag = f"t{th}_g{ga}"
        if os.path.exists(f"results/v2/net_{tag}.pt"):
            run(f"eval {tag}", f"results/v2/nn_shots_{tag}.csv",
                [PY, "src/eval_nn.py", f"results/v2/net_{tag}.pt", EVAL_SHOTS, WORKERS, f"results/v2/nn_shots_{tag}.csv"])
    if RUN_MAIN:
        run("main experiment 10k", "results/v2/corr_shots.csv",
            [PY, "src/exp_corr.py", "10000", WORKERS, "results/v2/corr_shots.csv"])
    log("queue finished")
