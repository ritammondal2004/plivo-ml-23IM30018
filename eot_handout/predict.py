import argparse
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from extract_features import build_feature_table

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--out", default="predictions.csv")
parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "model.pkl"))
args = parser.parse_args()

with open(args.model, "rb") as f:
    bundle = pickle.load(f)
            
model = bundle["model"]
feature_cols = bundle["feature_cols"]

X, meta = build_feature_table(args.data_dir)
X = X[feature_cols].fillna(0.0)
            
probs = model.predict_proba(X)[:, 1]

out_df = meta[["turn_id", "pause_index"]].copy()
out_df["p_eot"] = probs
out_df.to_csv(args.out, index=False)

print("wrote", len(out_df), "predictions to", args.out)