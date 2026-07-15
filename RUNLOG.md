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


## Run 2 — prosodic features (pitch + energy trajectory) + GradientBoostingClassifier
Trained on eot_data/english + eot_data/hindi combined.
Command: 

`python src/train.py --data_dirs eot_data/english eot_data/hindi --out model.pkl`

Cross-validated AUC (GroupKFold by turn, 5 folds): mean=0.631, folds=[0.585, 0.670, 0.660, 0.597, 0.643] 
In-sample score (model scored on the same data it trained on, NOT a fair estimate):
English AUC=0.986, delay=325ms @ 2% cutoffs
Hindi AUC=0.993, delay=160ms @ 4% cutoffs

What/why: added pitch slope, energy trajectory, voiced ratio, and turn-context
features (pause_index, time since turn start), all computed only from audio
before pause_start. In-sample numbers are inflated since the model saw this
data during training. CV AUC of 0.63 is the honest estimate.

## Run 3 — cross-language generalization check
Trained on English only, tested on Hindi (worst case, simulates unseen language).
Command:
`python src/train.py --data_dirs eot_data/english --out model_en_only.pkl`

`python predict.py --data_dir eot_data/hindi --model model_en_only.pkl --out predictions_hindi_heldout.csv`

Result: AUC=0.616, delay=850ms @ 5% cutoffs (no better than baseline on Hindi)

What/why: this confirms training on both languages together matters a lot —
single-language training doesn't transfer well. Final model (model.pkl) is
trained on English+Hindi combined, which is what we submit.

## Run 4 — honest held-out evaluation

was worried the run 2 numbers (AUC 0.986/0.993) were basically the model
memorizing the 200 turns it trained on since delay was suspiciously low.
wrote a small holdout script that carves out 40 turns (88 pauses) that
never touch training at all, then checks AUC only on those.

commands:
python src/eval_holdout.py --data_dirs eot_data/english eot_data/hindi --model model_v2_gbm.pkl
cp holdout_turns.csv holdout_turns_locked.csv
python src/train.py --data_dirs eot_data/english eot_data/hindi --out model_final.pkl --model_type gbm --exclude_turns holdout_turns_locked.csv
python src/eval_holdout.py --data_dirs eot_data/english eot_data/hindi --model model_final.pkl

result: held-out AUC = 0.682 (n=88 pauses, 40 turns never seen in training)

this is the real number to trust, not the in-sample 0.98+ from run 2.
0.682 is well above the ~0.50 baseline and the 0.616 english-only-to-hindi
transfer number from run 3, so combined training + current features do
carry real signal, just not as much as the in-sample numbers suggested.

also compared GradientBoosting vs HistGradientBoosting with the same
features (CV AUC 0.629 vs 0.611) - GBM won so kept it, just regularized
harder (max_depth=2, min_samples_leaf=15, subsample=0.8) to reduce the
overfitting seen in run 2.

## Run 5 — final submission model

retrained on full english+hindi data (no exclusions) since holdout was
only for honest evaluation, not part of the actual submitted model.

command:
```
python src/train.py --data_dirs eot_data/english eot_data/hindi --out model.pkl --model_type gbm
python predict.py --data_dir eot_data/english --model model.pkl --out predictions_english.csv
python predict.py --data_dir eot_data/hindi --model model.pkl --out predictions_hindi.csv
python starter/score.py --data_dir eot_data/english --pred predictions_english.csv
python starter/score.py --data_dir eot_data/hindi --pred predictions_hindi.csv
```

results: 
English AUC=0.980, delay=400ms @ 3% cutoffs
Hindi AUC=0.996, delay=190ms @ 2% cutoffs  

note: these scores are in-sample (model trained on this exact data) so
expect them to look better than reality. true expected performance on
unseen data is closer to the 0.682 held-out AUC from run 4.