"""Link to Claude session: https://claude.ai/share/172fb118-ffc6-47cb-be5e-6a58342295d0"""
"""Experiment 1 : can driving style alone predict terrain on BorealTC?
Usage : python3 experiment1.py --data BorealTC/data/borealtc
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# List of the 5 terrain types, list of features
TERRAINS = ["ASPHALT", "FLOORING", "ICE", "SANDY_LOAM", "SNOW"]  # exclude MIXED terrains
FEATURES = ["speed_med", "speed_iqr", "yaw_med", "yaw_iqr", "stationary_fraction"]

# Measures how spread out numbers are
def iqr(x: np.ndarray) -> float:
    # Get width of middle half
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

# This function is written with help of Claude
# Split ever run into 5s partitions + compute the five driving-style features on each
def build_partitions(data_dir, stationary_thr=0.05, part_s=5) -> pd.DataFrame:
    """ Speed comes from the pro file:  (velL + velR) / 2   [m/s]
        Yaw   comes from the imu file:  wz                  [rad/s]"""
    rows = []
    # Loop through each terrain
    for terrain_type in TERRAINS:
        # Loop through IMU file for each terrain
        for imu_path in sorted((data_dir / terrain_type).glob("imu_*.csv")):
            idx = imu_path.stem.split("_")[1] # Get run number
            pro_path = data_dir / terrain_type / f"pro_{idx}.csv" # Get path for matching wheel file

            # Load recording into tables
            imu, pro = pd.read_csv(imu_path), pd.read_csv(pro_path)
            
            t_end = min(imu.time.max(), pro.time.max()) # Take smaller end in case of inconsistency
            # Get speed and yaw
            speed = ((pro.velL + pro.velR) / 2).abs().to_numpy()
            yaw = imu.wz.abs().to_numpy()

            ti, tp = imu.time.to_numpy(), pro.time.to_numpy() # Time stamps for reading
            
            # Slice run into 5 second pieces
            for k in range(int(t_end // part_s)):
                a, b = k * part_s, (k + 1) * part_s
                # Get speed / yaw for each section
                s = speed[(tp >= a) & (tp < b)]
                y = yaw[(ti >= a) & (ti < b)]
                if len(s) < 2 or len(y) < 2:
                    continue

                # Fill in table
                rows.append(dict(
                    terrain=terrain_type, 
                    run=f"{terrain_type}_{idx}", 
                    part=k, 
                    t0=a,
                    speed_med=np.median(s), 
                    speed_iqr=iqr(s),
                    yaw_med=np.median(y), yaw_iqr=iqr(y),
                    stationary_fraction=float((s < stationary_thr).mean()),
                ))
    return pd.DataFrame(rows)

# This function is written with help of Claude
# Returns an untrained model for one outer fold
def make_model(name, X_tr, y_tr, g_tr):
    """ X_tr is the five driving numners for each training piece
        y_tr is the true terrain of each piece
        g_tr keeps track of which run each training piece came from"""
    if name == "logreg":
        # If any inner training fold is missing a class fall back to a fixed C=1.
        # Divide training tuns into 3 groups
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
    """Run-wise, terrain-stratified K-fold. Returns out-of-fold predictions.
       X is the five driving numbers for each piece
       y is the true terrain of each piece
       groups indicates which run each piece belongs to
       seed is for the purpose of reproducibility
       n_splits is how many groups to divide the runs into -- we will usually be doing 5"""
    # Divide runs into 5, make sure run is never split across parts, use stratified to get mix of terrains
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    fold_acc = [] # For scores each round
    # Loop through each round, get training positions and test positions
    for tr, te in sgkf.split(X, y, groups):
        model = make_model(name, X[tr], y[tr], groups[tr])
        # Learn
        model.fit(X[tr], y[tr])
        # Test prediction and get accuracy
        pred[te] = model.predict(X[te])
        fold_acc.append(accuracy_score(y[te], pred[te]))
    return pred, fold_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to BorealTC/data/borealtc")
    ap.add_argument("--part", type=float, default=5.0, help="partition length in seconds")
    ap.add_argument("--stationary-thr", type=float, default=0.05, help="stationary speed threshold (m/s); check the speed histogram")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--out", default="results_exp1")
    args = ap.parse_args()

    # Setup the output folder
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Save the partitions
    df = build_partitions(Path(args.data), args.stationary_thr, args.part)
    df.to_csv(out / "partitions.csv", index=False)

    table1 = df.groupby("terrain").agg(runs=("run", "nunique"), partitions=("part", "size"))
    print("\nTable 1 (runs and partitions per terrain):\n", table1)
    table1.to_csv(out / "table1.csv")

    # Split the table into the pieces the model needs, record the two reference scores
    X = df[FEATURES].to_numpy()
    y = df.terrain.to_numpy()
    groups = df.run.to_numpy()
    classes = sorted(set(y))
    # Next section written with Claude
    summary = {
        "n_partitions": len(y),
        "n_runs": len(set(groups)),
        "classes": classes,
        # Always predict the most common terrain
        "majority_baseline_acc": float(pd.Series(y).value_counts(normalize=True).max()),
        # Guesser: pick a terrain uniformly at random
        "chance_acc": 1 / len(classes),
    }

    for name in ["logreg", "gnb"]:
        pred, fold_acc = cv_predict(X, y, groups, name, args.seed, args.folds)
        summary[name] = dict(
            accuracy=accuracy_score(y, pred),
            balanced_accuracy=balanced_accuracy_score(y, pred),
            fold_acc_mean=float(np.mean(fold_acc)),
            fold_acc_std=float(np.std(fold_acc)),
            fold_accs=[float(a) for a in fold_acc]
        )
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
        cm.to_csv(out / f"confusion_{name}.csv")
        print(f"\n{name}: {summary[name]}\n{cm}")

    # Written partially with Claude
    # Permutation test for logistic regression: shuffle the terrain label of each run, refit, repeat
    rng = np.random.default_rng(args.seed)
    run_terrain = df.groupby("run")["terrain"].first()   # one terrain label per run
    runs = run_terrain.index.to_numpy()
    labels = run_terrain.to_numpy()

    null_acc, null_bacc = [], []
    for i in range(args.permutations):
        # Shuffle the labels across runs; every partition in a run gets its run's new label
        shuffled = dict(zip(runs, rng.permutation(labels)))
        y_perm = np.array([shuffled[g] for g in groups], dtype=object)

        pred_perm, _ = cv_predict(X, y_perm, groups, "logreg", args.seed, args.folds)
        null_acc.append(float(accuracy_score(y_perm, pred_perm)))
        null_bacc.append(float(balanced_accuracy_score(y_perm, pred_perm)))
        if (i + 1) % 20 == 0:
            print(f"permutation {i + 1}/{args.permutations}")

    obs_acc = summary["logreg"]["accuracy"]
    obs_bacc = summary["logreg"]["balanced_accuracy"]
    n = len(null_acc)
    summary["permutation_test"] = dict(
        model="logreg",
        n_permutations=n,
        observed_accuracy=obs_acc,
        null_acc_mean=float(np.mean(null_acc)),
        null_acc_std=float(np.std(null_acc)),
        null_acc_95th=float(np.percentile(null_acc, 95)),
        p_value_accuracy=(1 + sum(a >= obs_acc for a in null_acc)) / (n + 1),
        observed_balanced_accuracy=obs_bacc,
        null_bacc_mean=float(np.mean(null_bacc)),
        null_bacc_std=float(np.std(null_bacc)),
        null_bacc_95th=float(np.percentile(null_bacc, 95)),
        p_value_balanced_accuracy=(1 + sum(b >= obs_bacc for b in null_bacc)) / (n + 1),
    )
    print(f"\npermutation test: {summary['permutation_test']}")

    # Save every null score so you can plot the null distribution later
    (pd.DataFrame({"null_accuracy": null_acc, "null_balanced_accuracy": null_bacc})
       .to_csv(out / "null_accuracies.csv", index_label="perm"))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()