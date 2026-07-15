# Run Log

Format: after every scoring run — the score, and 1-2 lines on what changed and why.

## Run 1 — silence-only baseline (starter/baseline.py)

Commands: 

`python starter/baseline.py --data_dir eot_data/english --out preds_baseline_en.csv`
`python starter/score.py --data_dir eot_data/english --pred preds_baseline_en.csv`
  
`python starter/baseline.py --data_dir eot_data/hindi --out preds_baseline_hi.csv`
`python starter/score.py --data_dir eot_data/hindi --pred preds_baseline_hi.csv`

**Results:**
- English: AUC=0.514, mean delay=1600ms @ 0.0% interrupted turns (threshold=1.0, delay=1600ms)
- Hindi: AUC=0.501, mean delay=850ms @ 5.0% interrupted turns (threshold=0.05, delay=850ms)

**What/why:** This is the given baseline — p_eot=1.0 for every pause, so the agent has no real signal and just relies on the fixed action-delay sweep. AUC ~0.50 on both languages confirms there's no discriminative power (it's random). The English result (1600ms, 0% cutoffs) is the true "wait out the full timeout every time" behavior. The Hindi number (850ms) looks better but is not real signal — it's the scorer finding a threshold/delay pair that happens to land under the 5% cutoff budget by chance, since every prediction is identical. This is the number to beat with real prosodic features.

## Run 2 — (next: causal prosodic features + GradientBoostingClassifier)