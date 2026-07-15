import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import build_feature_table

parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", nargs="+", required=True)
parser.add_argument("--out", default="model.pkl")
parser.add_argument("--model_type", choices=["gbm", "histgbm"], default="gbm")
parser.add_argument("--exclude_turns", default=None,
                     help="CSV with a turn_id/group column to exclude from training (held-out set)")
args = parser.parse_args()

X_list = []
y_list = []
group_list = []

for d in args.data_dirs:
    X, meta = build_feature_table(d)
    y = (meta["label"] == "eot").astype(int).values
    lang_tag = os.path.basename(d.rstrip("/"))
    groups = meta["turn_id"].astype(str) + "_" + lang_tag

    X_list.append(X)
    y_list.append(y)
    group_list.append(groups)

    print(d, "->", len(X), "pauses,", y.sum(), "eot,", len(y) - y.sum(), "hold")

X_all = pd.concat(X_list, ignore_index=True)
y_all = np.concatenate(y_list)
groups_all = pd.concat(group_list, ignore_index=True)

feature_cols = list(X_all.columns)
X_all = X_all[feature_cols].fillna(0.0)


if args.exclude_turns:
    holdout_df = pd.read_csv(args.exclude_turns)
    holdout_groups = set(holdout_df["group"].astype(str)) if "group" in holdout_df.columns \
        else set(holdout_df.iloc[:, 0].astype(str))
    keep_mask = ~groups_all.astype(str).isin(holdout_groups)
    n_before = len(X_all)
    X_all = X_all[keep_mask].reset_index(drop=True)
    y_all = y_all[keep_mask.values]
    groups_all = groups_all[keep_mask].reset_index(drop=True)
    print(f"excluded {n_before - len(X_all)} pauses belonging to held-out turns")


def make_model():
    if args.model_type == "histgbm":
        return HistGradientBoostingClassifier(
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=0,
        )

    return GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=8,
    random_state=0,
    )


gkf = GroupKFold(n_splits=5)
fold_aucs = []
for train_idx, val_idx in gkf.split(X_all, y_all, groups_all):
    model = make_model()
    model.fit(X_all.iloc[train_idx], y_all[train_idx])
    preds = model.predict_proba(X_all.iloc[val_idx])[:, 1]
    if len(set(y_all[val_idx])) > 1:
        fold_aucs.append(roc_auc_score(y_all[val_idx], preds))

print("cv auc mean:", np.mean(fold_aucs), "folds:", fold_aucs)

final_model = make_model()
final_model.fit(X_all, y_all)

with open(args.out, "wb") as f:
    pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)

print("saved model to", args.out) 