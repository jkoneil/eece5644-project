#!/usr/bin/env python3
""" THIS IS PRIMARILY WRITTEN WTIH CLAUDE I just wanted to get a quick read on the data before investing time
I still need to go in and make edits / ensure it's actually correct and double check everything """
"""Experiment 1: can driving style alone predict terrain on BorealTC?

Usage (from the project folder):
    python3 experiment1.py --data BorealTC/data/borealtc

Needs: numpy pandas scikit-learn
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

TERRAINS = ["ASPHALT", "FLOORING", "ICE", "SANDY_LOAM", "SNOW"]  # MIXED is excluded
FEATURES = ["speed_med", "speed_iqr", "yaw_med", "yaw_iqr", "stationary_frac"]


def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25


def build_partitions(data_dir, part_s, stat_thr):
    """Cut every run into non-overlapping part_s-second partitions and compute
    the five driving-style features for each. Speed = |mean of wheel velocities|
    (6.5 Hz stream); yaw rate = |wz| from the IMU (100 Hz)."""
    rows = []
    for terr in TERRAINS:
        for imu_path in sorted((data_dir / terr).glob("imu_*.csv")):
            idx = imu_path.stem.split("_")[1]
            pro_path = data_dir / terr / f"pro_{idx}.csv"
            if not pro_path.exists():
                continue
            imu, pro = pd.read_csv(imu_path), pd.read_csv(pro_path)
            t_end = min(imu.time.max(), pro.time.max())
            speed = ((pro.velL + pro.velR) / 2).abs().to_numpy()
            yaw = imu.wz.abs().to_numpy()
            ti, tp = imu.time.to_numpy(), pro.time.to_numpy()
            for k in range(int(t_end // part_s)):
                a, b = k * part_s, (k + 1) * part_s
                s = speed[(tp >= a) & (tp < b)]
                y = yaw[(ti >= a) & (ti < b)]
                if len(s) < 2 or len(y) < 2:
                    continue
                rows.append(dict(
                    terrain=terr, run=f"{terr}_{idx}", part=k, t0=a,
                    speed_med=np.median(s), speed_iqr=iqr(s),
                    yaw_med=np.median(y), yaw_iqr=iqr(y),
                    stationary_frac=float((s < stat_thr).mean()),
                ))
    return pd.DataFrame(rows)


def make_model(name, X_tr, y_tr, g_tr):
    if name == "logreg":
        # inner CV is stratified and grouped by run so regularization is tuned
        # without leakage. If any inner training fold is missing a class (can
        # happen under label permutation), fall back to a fixed C=1.
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                          random_state=0).split(X_tr, y_tr, g_tr))
        n_cls = len(set(y_tr))
        ok = all(len(set(y_tr[a])) == n_cls and len(set(y_tr[b])) == n_cls
                 for a, b in inner)
        if ok:
            clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10, 100], cv=inner,
                                       max_iter=2000)
        else:
            clf = LogisticRegression(C=1.0, max_iter=2000)
    else:
        clf = GaussianNB()
    return make_pipeline(StandardScaler(), clf)


def cv_predict(X, y, groups, name, seed, n_splits):
    """Run-wise, terrain-stratified K-fold. Returns out-of-fold predictions."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    fold_acc = []
    for tr, te in sgkf.split(X, y, groups):
        model = make_model(name, X[tr], y[tr], groups[tr])
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        fold_acc.append(accuracy_score(y[te], pred[te]))
    return pred, fold_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to BorealTC/data/borealtc")
    ap.add_argument("--part", type=float, default=5.0, help="partition length (s)")
    ap.add_argument("--stat-thr", type=float, default=0.05,
                    help="stationary speed threshold (m/s); check the speed histogram")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", default="results_exp1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = build_partitions(Path(args.data), args.part, args.stat_thr)
    df.to_csv(out / "partitions.csv", index=False)

    table1 = df.groupby("terrain").agg(runs=("run", "nunique"), partitions=("part", "size"))
    print("\nTable 1 (runs and partitions per terrain):\n", table1)
    table1.to_csv(out / "table1.csv")

    X = df[FEATURES].to_numpy()
    y = df.terrain.to_numpy()
    groups = df.run.to_numpy()
    classes = sorted(set(y))
    summary = dict(chance=1 / len(classes),
                   majority_class=float(pd.Series(y).value_counts(normalize=True).max()))

    for name in ["logreg", "gnb"]:
        pred, fold_acc = cv_predict(X, y, groups, name, args.seed, args.folds)
        summary[name] = dict(
            accuracy=accuracy_score(y, pred),
            balanced_accuracy=balanced_accuracy_score(y, pred),
            fold_acc_mean=float(np.mean(fold_acc)),
            fold_acc_std=float(np.std(fold_acc)),
        )
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes),
                          index=classes, columns=classes)
        cm.to_csv(out / f"confusion_{name}.csv")
        print(f"\n{name}: {summary[name]}\n{cm}")

    # Permutation test (logistic regression): shuffle the terrain label of each
    # RUN (all partitions of a run get the same shuffled label), refit, repeat.
    run_terr = df.groupby("run").terrain.first()
    rng = np.random.RandomState(args.seed)
    obs = summary["logreg"]["accuracy"]
    null = []
    for _ in range(args.perms):
        shuffled = pd.Series(rng.permutation(run_terr.to_numpy()), index=run_terr.index)
        y_perm = df.run.map(shuffled).to_numpy()
        p, _ = cv_predict(X, y_perm, groups, "logreg", args.seed, args.folds)
        null.append(accuracy_score(y_perm, p))
    null = np.array(null)
    summary["permutation"] = dict(
        n=args.perms, null_mean=float(null.mean()), null_std=float(null.std()),
        null_95th=float(np.percentile(null, 95)),
        p_value=float((1 + (null >= obs).sum()) / (1 + len(null))),
    )
    np.save(out / "null_accuracies.npy", null)
    print("\nPermutation test:", summary["permutation"])

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved partitions.csv, table1.csv, confusion matrices, summary.json to {out}/")


if __name__ == "__main__":
    main()