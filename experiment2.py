"""Link to Claude session: https://claude.ai/share/f4efae40-3f87-457a-a78c-535abbcb6bd9"""
"""Experiment 2: does the CNN rely on driving style?
Two stages (run `--stage both` for both):

  train    Run the authors' CNN pipeline (LaRocque et al. 2024). You can run with 
           the original partition-wise folds or run-wise folds (--split partition
           or --split run). In both cases, the model/optimizer/hyperparaemters
           are from the authors' original cnn_train.py.
           The program will save one predcition per held-out 1.7s window with its parent
           5s partition / driving-style features.

  analyze  Bins partitions into a 3x3 grid of median speed x median yaw rate
           (terciles), keeps bins with more than one terrain, and compares
           conditioned accuracy (AccCond) with pooled accuracy (AccPool).
NOTICE : You must delete the results_exp2 folder before starting if you want a fresh run !!
         Otherwise, you skip a fold where the csv alrady exists, no new training
         Originally implemented this way to save progress when a run crashed mid way through
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
import einops as ein
import torch
import lightning
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# Reused unchanged from experiment1.py: terrain list, feature names, IQR helper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment1 import TERRAINS, iqr
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------------------------------
# Settings copied from the authors' cnn_train.py (only `verbose` is
# switched off to keep the console readable)
# --------------------------------------------------------------------------
PART_WINDOW = 5      # seconds
N_FOLDS = 5
MOVING_WINDOW = 1.7  # seconds
STRIDE = 0.1         # seconds
HOMOGENEOUS_AUGMENTATION = True
CNN_PAR = {
    "time_window": 0.4, 
    "time_overlap": 0.2, 
    "filter_size": [3, 3], 
    "num_filters": 16 * 2
}

CNN_TRAIN_OPT = {
    "hamming": True, 
    "valid_perc": 0.1, 
    "init_learn_rate": 0.005, 
    "learn_drop_factor": 0.1,
    "max_epochs": 150, 
    "minibatch_size": 10, 
    "valid_patience": 8, 
    "scheduler": "plateau",
    "reduce_lr_patience": 4, 
    "valid_frequency": 1.0, 
    "gradient_threshold": 6,
    "focal_loss": False, 
    "focal_loss_alpha": 0.25, 
    "focal_loss_gamma": 2,
    "verbose": False, 
    "dropout": 0.0, 
    "use_augmentation": False,
}

COLUMNS = {
    "imu": {
        "wx": True, 
        "wy": True, 
        "wz": True, 
        "ax": True, 
        "ay": True, 
        "az": True
    },
    "pro": {
        "velL": True, 
        "velR": True, 
        "curL": True, 
        "curR": True},
}
# Column layout of the authors' partition arrays (n_partitions, time, channels):
#   0 terrain, 1 terr_idx, 2 run_idx, 3 win_idx (partition number), 4 time, then the channels
IMU_WZ_COL = 5 + list(COLUMNS["imu"]).index("wz")
PRO_VELL_COL = 5 + list(COLUMNS["pro"]).index("velL")
PRO_VELR_COL = 5 + list(COLUMNS["pro"]).index("velR")

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "BorealTC"                 # the authors' repo, with the data in place
DATA_DIR = REPO / "data" / "borealtc"
OUT_DIR = ROOT / "results_exp2"
SEED = 21                                # the authors' RANDOM_STATE
STATIONARY_THR = 0.05                    # same as experiment1.py
N_BINS = 3                               # terciles of median speed and of median yaw rate
MIN_PARTS = 10                           # held-out partitions for a terrain to count in a bin
TOLERANCE = 0.05                         # pre-set 5-point rule

# ==========================================================================
# TRAIN STAGE
# ==========================================================================
def port_code():
    """Patch the environment so the authors' models.py runs without Mamba or an
    NVIDIA GPU, and on recent PyTorch. Safe to call more than once."""

    # 1. models.py imports mamba_ssm at the top, but only MambaTerrain uses it.
    #    A stub module lets the import succeed; calling it fails loudly.
    if "mamba_ssm" not in sys.modules:
        def create_block(*args, **kwargs):
            raise RuntimeError("mamba_ssm is stubbed out; Mamba models are unavailable")
        names = ["mamba_ssm", "mamba_ssm.models", "mamba_ssm.models.mixer_seq_simple"]
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[-1]].create_block = create_block

    # 2. The authors hard-code accelerator="gpu". Fall back to CPU when there's no CUDA.
    if not getattr(lightning.Trainer, "_ported", False):
        class Trainer(lightning.Trainer):
            _ported = True
            def __init__(self, *args, **kwargs):
                if kwargs.get("accelerator") == "gpu" and not torch.cuda.is_available():
                    kwargs["accelerator"] = "cpu"
                super().__init__(*args, **kwargs)
        lightning.Trainer = Trainer

    # 3. Recent PyTorch removed ReduceLROnPlateau's `verbose` argument. Drop it.
    sched = torch.optim.lr_scheduler
    if not getattr(sched.ReduceLROnPlateau, "_ported", False):
        class ReduceLROnPlateau(sched.ReduceLROnPlateau):
            _ported = True
            def __init__(self, *args, verbose=None, **kwargs):
                super().__init__(*args, **kwargs)
        sched.ReduceLROnPlateau = ReduceLROnPlateau


# This function was adapted with Claude from the code written for Experiment 1
def partition_features(data: dict, stationary_thr: float = 0.05) -> pd.DataFrame:
    """Five driving-style features from Experiment 1, computed on the authors' 5 s partitions.
    data = {"imu": arr, "pro": arr}, arr shape (n_partitions, n_samples, 5 + channels)."""
    imu, pro = data["imu"], data["pro"]

    # Labels from the first sample: 0 terrain, 2 run_idx, 3 win_idx (partition)
    lab_imu, lab_pro = imu[:, 0, [0, 2, 3]], pro[:, 0, [0, 2, 3]]
    assert lab_imu.shape == lab_pro.shape, "IMU/pro partition count differs"
    assert (lab_imu == lab_pro).all(), "IMU/pro mismatch in terrain/run_idx/part"

    terrain = lab_imu[:, 0].astype(str)
    run_idx = lab_imu[:, 1].astype(int)

    speed = np.abs((pro[:, :, PRO_VELL_COL] + pro[:, :, PRO_VELR_COL]).astype(float) / 2)
    yaw = np.abs(imu[:, :, IMU_WZ_COL].astype(float))

    df = pd.DataFrame({
        "terrain": terrain,
        "run": [f"{t}_{r}" for t, r in zip(terrain, run_idx)],
        "run_idx": run_idx,
        "part": lab_imu[:, 2].astype(int),
        "t0": imu[:, 0, 4].astype(float),
        "speed_med": np.median(speed, axis=1),
        "speed_iqr": np.apply_along_axis(iqr, 1, speed),
        "yaw_med": np.median(yaw, axis=1),
        "yaw_iqr": np.apply_along_axis(iqr, 1, yaw),
        "stationary_fraction": (speed < stationary_thr).mean(axis=1),
    })
    assert not df.duplicated(["run", "part"]).any(), "duplicate (run, part)"
    return df

# Experiment 1 pattern, new core, written with help of Claude
def make_fold_splitter(mode, original, state, stationary_thr=0.05):
    """'partition': authors' original StratifiedKFold over partitions.
       'run':       StratifiedGroupKFold, so each run lands in exactly one fold.
        Also stores the per-partition driving-style features in state["partitions"]."""
    assert mode in ("partition", "run"), f"unknown split mode {mode!r}"

    def splitter(unified, labels, n_splits=5, random_state=None, ablation=False):
        df = partition_features(unified, stationary_thr)
        state["partitions"] = df
        if mode == "partition" or n_splits is None:
            return original(unified, labels, n_splits, random_state, ablation)

        assert not ablation, "ablation not supported in run mode"
        # The authors' pipeline needs every terrain in every train and test fold. With only 7
        # ASPHALT runs, a shuffle can leave a fold without one, so try seeds seed, seed+1, ...
        # and keep the first valid split (deterministic, and the seed used is recorded).
        for attempt in range(100):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + attempt)
            folds = list(sgkf.split(np.zeros(len(labels)), labels, df.run.to_numpy()))
            if all(set(labels[tr]) == set(labels) == set(labels[te]) for tr, te in folds):
                break
        else:
            raise RuntimeError("no run-wise split with every terrain in every fold in 100 tries")
        state["split_seed"] = random_state + attempt
        print(f"run-wise folds use seed {state['split_seed']}")
        train, test = [], []
        for tr, te in folds:
            assert not set(df.run.iloc[tr]) & set(df.run.iloc[te]), "a run appears in both train and test"
            train.append({k: v[tr] for k, v in unified.items()})
            test.append({k: v[te] for k, v in unified.items()})
        return train, test

    return splitter

# This function is a combination of the author code new code (partially Claude code)
def stage_train(args):
    out = (OUT_DIR / args.split).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    port_code()

    from utils import frequency, models, preprocessing  # the authors' code
    _dft = frequency.DFT
    frequency.DFT = lambda signal, sf, hamming=False: _dft(np.asarray(signal, dtype=float), sf, hamming)

    # Lightning writes tb_logs/ and checkpoints/ relative to cwd, so keep them in out/
    os.chdir(out)
    with open(out / "config.json", "w") as f:
        json.dump({**vars(args), "cnn_par": CNN_PAR, "cnn_train_opt": CNN_TRAIN_OPT}, f, indent=2)

    # ---- Authors' setup
    summary = pd.DataFrame({"columns": pd.Series(COLUMNS)})
    terr_dfs = preprocessing.get_recordings(DATA_DIR, summary)
    classes = sorted(map(str, np.unique(terr_dfs["imu"].terrain)))
    assert classes == sorted(TERRAINS), f"unexpected terrains {classes}"

    # ---- Authors' partitioning, with our fold splitter swapped in
    state = {}
    # Keep the authors' own splitter: stage_train can run twice in one process (--split both)
    original = getattr(preprocessing, "_orig_kfold_splits", preprocessing.kfold_splits)
    preprocessing._orig_kfold_splits = original
    preprocessing.kfold_splits = make_fold_splitter(args.split, original, state, STATIONARY_THR)
    train, test = preprocessing.partition_data(terr_dfs, summary, PART_WINDOW, N_FOLDS, random_state=SEED)
    state["partitions"].to_csv(out / "partitions.csv", index=False)
    print(f"{len(state['partitions'])} partitions from {state['partitions'].run.nunique()} runs "
          f"(split={args.split}, seed={SEED})")

    test_runs = [set(zip(t["imu"][:, 0, 0], t["imu"][:, 0, 2])) for t in test]
    if args.split == "run":
        assert sum(map(len, test_runs)) == len(set().union(*test_runs)), "a run is in >1 test fold"

    # ---- Authors' augmentation, spectrograms, and min/max over all folds' train data
    aug_train, aug_test = preprocessing.augment_data(
        train, test, summary, moving_window=MOVING_WINDOW,
        stride=STRIDE, homogeneous=HOMOGENEOUS_AUGMENTATION)
    train_mcs, test_mcs = preprocessing.apply_multichannel_spectogram(
        aug_train, aug_test, summary, MOVING_WINDOW,
        CNN_PAR["time_window"], CNN_PAR["time_overlap"], hamming=CNN_TRAIN_OPT["hamming"])
    flat = [ein.rearrange(t["data"], "a b c d -> (a b c) d") for t in train_mcs]
    maxs = np.stack([f.max(axis=0) for f in flat]).max(axis=0)
    mins = np.stack([f.min(axis=0) for f in flat]).min(axis=0)

    # ---- Train / predict per fold (folds are numbered 1..5)
    cnn_par = {**CNN_PAR, "num_classes": len(classes)}
    for k in range(1, N_FOLDS + 1):
        pred_path = out / f"predictions_fold{k}.csv"
        if pred_path.exists():
            print(f"fold {k}: {pred_path.name} exists, skipping")
            continue

        res = models.convolutional_neural_network(
            train_mcs[k - 1], test_mcs[k - 1], cnn_par, CNN_TRAIN_OPT,
            dict(mw=MOVING_WINDOW, fold=k, dataset="husky", mins=mins, maxs=maxs),
            random_state=SEED)

        # Map each held-out window back to its parent partition. The test loader does not
        # shuffle, so predictions are in the same order as the augmented test windows.
        meta = aug_test[k - 1]["imu"][:, 0, :4]  # terrain, terr_idx, run_idx, win_idx
        true = np.array(classes)[np.asarray(res["true"], dtype=int)]
        assert len(meta) == len(true), "window count mismatch"
        assert (true == meta[:, 0].astype(str)).all(), "test window order changed"

        preds = pd.DataFrame({
            "fold": k,
            "run": [f"{t}_{int(r)}" for t, r in zip(meta[:, 0], meta[:, 2])],
            "part": meta[:, 3].astype(int),
            "window": np.arange(len(meta)),
            "true": true,
            "pred": np.array(classes)[np.asarray(res["pred"], dtype=int)],
        })
        preds[[f"conf_{c}" for c in classes]] = res["conf"]
        preds.to_csv(pred_path, index=False)
        print(f"fold {k}: window accuracy = {np.mean(preds.true == preds.pred):.4f} on {len(preds)} windows")


# ==========================================================================
# ANALYZE STAGE
# ==========================================================================

def load_windows(out_dir):
    files = sorted(Path(out_dir).glob("predictions_fold*.csv"))
    if not files:
        raise SystemExit(f"no predictions_fold*.csv in {out_dir}; run --stage train first")
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    w["correct"] = (w["pred"] == w["true"]).astype(int)
    return w

# Written with help of Claude
def assign_bins(parts, windows, n_bins):
    p = parts[["run", "part", "speed_med", "yaw_med"]].copy()
    p["speed_bin"], speed_edges = pd.qcut(p.speed_med, n_bins, labels=False, retbins=True)
    p["yaw_bin"], yaw_edges = pd.qcut(p.yaw_med, n_bins, labels=False, retbins=True)
    p["bin"] = p.speed_bin * n_bins + p.yaw_bin

    w = windows.merge(p[["run", "part", "speed_bin", "yaw_bin", "bin"]], on=["run", "part"], how="left", validate="many_to_one")
    assert w["bin"].notna().all(), "some windows have no parent partition"
    return p, w, speed_edges[1:-1], yaw_edges[1:-1]

# new
def analyze(split):
    out = (OUT_DIR / split).resolve()
    parts = pd.read_csv(out / "partitions.csv")
    windows = load_windows(out)
    parts, w, speed_edges, yaw_edges = assign_bins(parts, windows, N_BINS)

    # Held-out partitions per (bin, terrain) cell; each partition is held out once
    held = w.drop_duplicates(["run", "part"])
    table = held.pivot_table(index="bin", columns="true", values="part",
                             aggfunc="size", fill_value=0)
    valid = table >= MIN_PARTS
    mixed = valid.sum(axis=1) >= 2

    # Keep only windows in valid cells of mixed bins
    cells = [(b, t) for b in table.index[mixed] for t in table.columns[valid.loc[b]]]
    m = w[pd.MultiIndex.from_arrays([w.bin, w.true]).isin(cells)]

    # Per-bin table for the Results section. Counts, recalls and the per-bin accuracy pool
    # all five folds' held-out windows (each partition is held out exactly once).
    bins = table.add_prefix("n_")
    bins.insert(0, "speed_bin", bins.index // N_BINS)
    bins.insert(1, "yaw_bin", bins.index % N_BINS)
    bins["n_valid_terrains"] = valid.sum(axis=1)
    bins["mixed"] = mixed
    bins["n_windows"] = w.groupby("bin").size()
    bins["n_windows_used"] = m.groupby("bin").size().reindex(bins.index, fill_value=0)
    recall = m.groupby(["bin", "true"]).correct.mean().unstack()
    bins = bins.join(recall.add_prefix("recall_"))
    bins["bin_balanced_acc_pooled_over_folds"] = pd.Series(
        {b: balanced_accuracy_score(g.true, g.pred) for b, g in m.groupby("bin")})
    bins.to_csv(out / "bins.csv", na_rep="NA")  # recall_* and bin acc are NA outside valid cells / mixed bins

    # Per fold: pooled vs conditioned balanced accuracy on the SAME windows (m).
    # Primary AccCond weights every usable bin equally; the window-weighted version is a
    # robustness check. A bin is usable in a fold only if >= 2 terrains appear in it there.
    folds = []
    for k, f in m.groupby("fold"):
        per_bin = [(balanced_accuracy_score(g.true, g.pred), len(g))
                   for _, g in f.groupby("bin") if g.true.nunique() >= 2]
        if not per_bin:
            continue
        accs, sizes = map(np.array, zip(*per_bin))
        pool = balanced_accuracy_score(f.true, f.pred)
        cond, cond_w = float(accs.mean()), float(np.average(accs, weights=sizes))
        folds.append(dict(fold=int(k), acc_pool=float(pool), acc_cond=cond,
                          gap=float(pool - cond), acc_cond_weighted=cond_w,
                          gap_weighted=float(pool - cond_w),
                          n_windows=len(f), n_bins_used=len(per_bin)))
    assert folds, "no fold has a bin with two valid terrains"

    fd = pd.DataFrame(folds)

    def std(x):  # sample std (ddof=1); undefined with a single fold -> null in the JSON
        return float(x.std(ddof=1)) if len(x) > 1 else None
    gap, cond = fd.gap.mean(), fd.acc_cond.mean()
    chance = float((1 / bins.n_valid_terrains[mixed]).mean())
    if gap > TOLERANCE:
        verdict = "relies on driving style: accuracy drops when driving style is held fixed"
    elif cond > chance + TOLERANCE:
        verdict = "reads terrain signal beyond driving style"
    else:
        verdict = "inconclusive: conditioned accuracy is near chance"

    summary = dict(
        split=split, bins=N_BINS, min_parts=MIN_PARTS, tolerance=TOLERANCE,
        speed_edges=speed_edges.tolist(), yaw_edges=yaw_edges.tolist(),
        n_mixed_bins=int(mixed.sum()), n_windows_used=len(m), n_windows_total=len(w),
        acc_overall=float((w.pred == w.true).mean()),  # plain accuracy on all windows (authors' metric)
        folds=folds, std_ddof=1,
        acc_pool_mean=float(fd.acc_pool.mean()), acc_pool_std=std(fd.acc_pool),
        acc_cond_mean=float(cond), acc_cond_std=std(fd.acc_cond),
        gap_mean=float(gap), gap_std=std(fd.gap),
        acc_cond_weighted_mean=float(fd.acc_cond_weighted.mean()),
        gap_weighted_mean=float(fd.gap_weighted.mean()),
        chance_cond=chance, verdict=verdict,
    )
    with open(out / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"[{split}] overall acc={summary['acc_overall']:.3f}  AccPool={summary['acc_pool_mean']:.3f}  "
          f"AccCond={cond:.3f}  gap={gap:+.3f}  (window-weighted gap={summary['gap_weighted_mean']:+.3f})  "
          f"chance={chance:.3f}\n  -> {verdict}")
    print(bins.round(3).to_string())
    return summary


def stage_analyze(args):
    splits = [args.split] if args.split != "both" else ["partition", "run"]
    for s in splits:
        if (OUT_DIR / s).exists():
            analyze(s)
        else:
            print(f"[{s}] no results folder at {OUT_DIR / s}, skipping")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Help sections are written with Claude
    ap.add_argument("--stage", choices=["train", "analyze", "both"], default="both")
    ap.add_argument("--split", choices=["run", "partition", "both"], default="run",
                    help="run = run-wise folds, each run in one fold; "
                    "partition = the authors' original partition-wise folds; "
                    "both = partition then run (default: run)")
    args = ap.parse_args()

    splits = ["partition", "run"] if args.split == "both" else [args.split]
    if args.stage in ("train", "both"):
        for s in splits:
            stage_train(argparse.Namespace(**{**vars(args), "split": s}))
    if args.stage in ("analyze", "both"):
        stage_analyze(args)


if __name__ == "__main__":
    main()