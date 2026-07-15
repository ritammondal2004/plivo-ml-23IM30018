
# Causal feature extraction for End-of-Turn detection.


import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter"))
from features import load_wav, speech_before, frame_energy_db, f0_contour 

WINDOW_S = 1.5  # how much pre-pause audio we look at
                              

def _safe_stats(arr):
    """mean/std/min/max/slope of a 1D array, 0 if empty."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if len(arr) >= 2:
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
    else:
        slope = 0.0
    return mean, std, mn, mx, slope


def extract_features_for_pause(x, sr, pause_start, turn_total_speech_s,
                                pause_index, time_since_turn_start):
    """Build one feature row for a single pause, using ONLY audio
    strictly before pause_start."""
    seg = speech_before(x, sr, pause_start, window_s=WINDOW_S)
                                    
    feats = {}

             
    if len(seg) > 0:
        e_db = frame_energy_db(seg, sr)
    else:
        e_db = np.array([])
    e_mean, e_std, e_min, e_max, e_slope = _safe_stats(e_db)
    feats["energy_mean"] = e_mean
    feats["energy_std"] = e_std
    feats["energy_min"] = e_min
    feats["energy_max"] = e_max
    feats["energy_slope"] = e_slope
    if len(e_db) > 0:
        last_n = max(1, int(0.3 * (len(e_db) / (WINDOW_S if WINDOW_S > 0 else 1))))
        feats["energy_last_vs_mean"] = float(np.mean(e_db[-last_n:]) - e_mean)
    else:
        feats["energy_last_vs_mean"] = 0.0
                     
                             
    if len(seg) > 0:                     
        f0 = f0_contour(seg, sr)        
        voiced = f0[f0 > 0]          
    else:                      
        f0 = np.array([])          
        voiced = np.array([])
    f0_mean, f0_std, f0_min, f0_max, f0_slope = _safe_stats(voiced)
    feats["f0_mean"] = f0_mean
    feats["f0_std"] = f0_std
    feats["f0_slope"] = f0_slope
    feats["voiced_ratio"] = float(len(voiced) / len(f0)) if len(f0) > 0 else 0.0
    if len(voiced) >= 4:
        half = len(voiced) // 2
        feats["f0_second_half_mean"] = float(np.mean(voiced[half:]))
        feats["f0_first_half_mean"] = float(np.mean(voiced[:half]))
        feats["f0_fall"] = feats["f0_first_half_mean"] - feats["f0_second_half_mean"]
    else:  
        feats["f0_second_half_mean"] = 0.0
        feats["f0_first_half_mean"] = 0.0
        feats["f0_fall"] = 0.0

    feats["pause_index"] = int(pause_index)
    feats["time_since_turn_start"] = float(time_since_turn_start)
    feats["turn_total_speech_s"] = float(turn_total_speech_s)

    return feats


def build_feature_table(data_dir):
    """Given a folder with audio/ + labels.csv, return (X_df, meta_df)."""
    labels_path = os.path.join(data_dir, "labels.csv")
    df = pd.read_csv(labels_path)

    rows = []
    metas = []
    wav_cache = {}  

    for turn_id, g in df.groupby("turn_id"):
        g = g.sort_values("pause_index")
        audio_file = g.iloc[0]["audio_file"]
        wav_path = os.path.join(data_dir, audio_file)
        if wav_path not in wav_cache:
            wav_cache[wav_path] = load_wav(wav_path)
        x, sr = wav_cache[wav_path]

        for _, r in g.iterrows():
            pause_start = float(r["pause_start"])
            feats = extract_features_for_pause(
                x, sr, pause_start,
                turn_total_speech_s=pause_start,
                pause_index=r["pause_index"],
                time_since_turn_start=pause_start,
            )
            rows.append(feats)
            meta = {"turn_id": r["turn_id"], "pause_index": int(r["pause_index"])}
            if "label" in r:
                meta["label"] = r["label"]
            metas.append(meta)

    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metas)
    return X, meta_df