"""Bandingkan data produksi (logs/predictions.jsonl) dengan data latih (reference.csv).
Jalankan:  python monitoring/drift_check.py
"""
import json, sys
import pandas as pd
from scipy.stats import ks_2samp

ref = pd.read_csv("monitoring/reference.csv")
rows = [json.loads(l) for l in open("logs/predictions.jsonl")]
if len(rows) < 30:
    sys.exit(f"Data produksi baru {len(rows)} baris, minimal 30.")
cur = pd.DataFrame([r["input"] for r in rows])

drift = False
for col in ref.select_dtypes("number").columns:
    p = ks_2samp(ref[col], cur[col]).pvalue
    flag = p < 0.01
    drift |= flag
    print(f"{col:18s} KS p-value={p:.4f} {'<< DRIFT' if flag else 'ok'}")
for col in ref.select_dtypes(exclude="number").columns:
    unseen = (~cur[col].isin(ref[col].unique())).mean()
    flag = unseen > 0.10
    drift |= flag
    print(f"{col:18s} kategori baru={unseen:.1%} {'<< DRIFT' if flag else 'ok'}")
print("\nHASIL:", "ADA DRIFT, pertimbangkan retraining" if drift else "aman")
sys.exit(1 if drift else 0)
