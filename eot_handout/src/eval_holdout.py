import argparse, os, sys
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import build_feature_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--locked_file", default="holdout_turns_locked.csv")
    args = ap.parse_args()

    X_list, y_list, group_list = [], [], []
    for d in args.data_dirs:
        X, meta = build_feature_table(d)
        y = (meta["label"] == "eot").astype(int).values
        lang_tag = os.path.basename(d.rstrip("/"))
        groups = meta["turn_id"].astype(str) + "_" + lang_tag
        X_list.append(X)
        y_list.append(pd.Series(y))
        group_list.append(groups)

    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)
    groups_all = pd.concat(group_list, ignore_index=True)

    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    X_all = X_all[feature_cols].fillna(0.0)

    if os.path.exists(args.locked_file):
        locked = pd.read_csv(args.locked_file)
        locked_groups = set(locked["group"].astype(str))
        test_idx = groups_all[groups_all.astype(str).isin(locked_groups)].index.values
        print(f"reusing locked split from {args.locked_file}")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, test_idx = next(gss.split(X_all, y_all, groups_all))
        pd.DataFrame({"group": groups_all.iloc[test_idx]}).to_csv("holdout_turns.csv", index=False)
        print("wrote fresh holdout_turns.csv with", groups_all.iloc[test_idx].nunique(), "unique turns")

    p = model.predict_proba(X_all.iloc[test_idx])[:, 1]
    auc = roc_auc_score(y_all.iloc[test_idx], p)
    print(f"HELD-OUT AUC: {auc:.3f}  n_pauses={len(test_idx)}  n_turns={groups_all.iloc[test_idx].nunique()}")

if __name__ == "__main__":
    main()  