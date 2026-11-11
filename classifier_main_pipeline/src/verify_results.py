r"""
Recompute test metrics from a saved CNN-LSTM checkpoint using the SAME split the
training notebook used (train_test_split, test_size=0.2, random_state=42, stratified).

Run from the src folder, e.g.
  python verify_results.py --data ..\data\processed\epochs_all.npy ^
      --labels ..\data\processed\labels_all.npy ^
      --weights ..\models\cnn_lstm_hybrid_best.pth --out results_check.json
"""
import argparse, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from cnn_lstm_model import load_model, normalise_epochs, predict_proba, STAGE_NAMES

REFS = {
    "20-subject run (RESULTS.md, Exp 003)": (0.8014, 0.7128),
    "153-rec run, paper Fig. 4":            (0.7689, 0.6932),
    "153-rec run, Week 9 log":              (0.7559, 0.6780),
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--out", default="results_check.json")
a = ap.parse_args()

X = np.load(a.data); y = np.load(a.labels).astype(int)
print(f"Loaded {X.shape} epochs, {len(y)} labels")
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Test epochs: {len(y_te)}  (20-subject data -> ~3,646; 153-recording data -> 35,483)")

model = load_model(a.weights)
probs = predict_proba(model, normalise_epochs(X_te))
pred = probs.argmax(1)

p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=range(5), zero_division=0)
auc = [roc_auc_score((y_te == k).astype(int), probs[:, k]) for k in range(5)]
res = dict(n_test=int(len(y_te)), acc=float((pred == y_te).mean()),
           kappa=float(cohen_kappa_score(y_te, pred)), mcc=float(matthews_corrcoef(y_te, pred)),
           cm=confusion_matrix(y_te, pred, labels=range(5)).tolist(),
           precision=p.round(4).tolist(), recall=r.round(4).tolist(), f1=f.round(4).tolist(),
           support=s.tolist(), auc=[round(x, 4) for x in auc],
           macro_f1=round(float(f.mean()), 4), macro_auc=round(float(np.mean(auc)), 4))
json.dump(res, open(a.out, "w"), indent=2)

print(f"\nacc={res['acc']:.4f}  kappa={res['kappa']:.4f}  mcc={res['mcc']:.4f}")
print(f"{'Stage':6} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6} {'n':>7}")
for i, nme in enumerate(STAGE_NAMES):
    print(f"{nme:6} {p[i]:6.3f} {r[i]:6.3f} {f[i]:6.3f} {auc[i]:6.3f} {int(s[i]):7d}")
print("\nComparison with reference numbers:")
for k, (ra, rk) in REFS.items():
    ok = abs(res['acc'] - ra) < 6e-4 and abs(res['kappa'] - rk) < 1.1e-3
    print(f"  {k:40} acc {ra:.4f} kappa {rk:.4f}  {'<-- MATCH' if ok else ''}")
print(f"\nSaved {a.out}")