import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import build_feature_table

parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", nargs="+", required=True)
parser.add_argument("--out", default="model.pkl")
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

# quick cross validation just to see if the features are doing anything
gkf = GroupKFold(n_splits=5)  
fold_aucs = []
for train_idx, val_idx in gkf.split(X_all, y_all, groups_all):
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )   
    model.fit(X_all.iloc[train_idx], y_all[train_idx])
    preds = model.predict_proba(X_all.iloc[val_idx])[:, 1]
    if len(set(y_all[val_idx])) > 1:
        fold_aucs.append(roc_auc_score(y_all[val_idx], preds))

print("cv auc mean:", np.mean(fold_aucs), "folds:", fold_aucs)

# now train final model on everything
final_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=0,
)
final_model.fit(X_all, y_all)

with open(args.out, "wb") as f:
    pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)

print("saved model to", args.out)