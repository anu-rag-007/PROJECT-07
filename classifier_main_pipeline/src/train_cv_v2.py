r"""
Subject-grouped K-fold cross-validation with a proper validation split.

* Subjects (not epochs) are split, so a person never appears in both training and test.
* Inside each fold, ~12% of the training subjects form a validation set used for early stopping
  and checkpoint choice. The test subjects are never used to make any decision.
* Every epoch of every recording receives an out-of-fold prediction from a model that never saw
  that subject. Those predictions are what the stable-N1 trigger replay should run on.
* Folds are saved as they finish, so an interrupted run resumes where it stopped.

Smoke test first:  python train_cv_v2.py --data_dir ..\data\processed_v2 --limit_subjects 10 --max_epochs 2 --out_dir cv_smoke
Full run:          python train_cv_v2.py --data_dir ..\data\processed_v2 --out_dir cv_out
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from cnn_lstm_model import CNNLSTMSleepClassifier, normalise_epochs, predict_proba
from cv_utils import assign_folds, confusion, kappa_from_cm, per_class, subject_bootstrap

STAGES = ["Wake", "N1", "N2", "N3", "REM"]


class EpochDS(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y, self.idx = X, y, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]; x = self.X[j].astype(np.float32)
        return torch.from_numpy((x - x.mean()) / (x.std() + 1e-8)), int(self.y[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="cv_out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_subjects", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    d, out = Path(a.data_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(d / "X.npy"); y = np.load(d / "y.npy").astype(np.int64); m = np.load(d / "meta.npz")
    subj, rec_idx, eidx = m["subj"], m["rec_idx"], m["epoch_idx"]
    if a.limit_subjects:
        keep = np.isin(subj, np.unique(subj)[: a.limit_subjects])
        X, y, subj, rec_idx, eidx = X[keep], y[keep], subj[keep], rec_idx[keep], eidx[keep]
    valid = y >= 0
    fold_of = assign_folds(subj[valid], y[valid], a.folds, a.seed)
    fold_arr = np.array([fold_of.get(int(s), -1) for s in subj])
    np.savez(out / "aligned.npz", y=y, subj=subj, rec_idx=rec_idx, epoch_idx=eidx, fold=fold_arr)
    print(f"{len(np.unique(subj))} subjects, {valid.sum()} labelled epochs, {a.folds} folds", flush=True)

    oof = np.full((len(y), 5), np.nan, np.float32)
    for k in range(a.folds):
        f = out / f"fold{k}.npz"
        te_all = np.where(fold_arr == k)[0]
        if f.exists():
            z = np.load(f); oof[z["idx"]] = z["probs"]; print(f"fold {k}: loaded from disk"); continue
        rest = np.array(sorted(set(np.unique(subj).tolist()) - set(np.unique(subj[te_all]).tolist())))
        rng = np.random.default_rng(a.seed + k)
        val_s = rng.choice(rest, size=max(2, int(round(0.12 * len(rest)))), replace=False)
        va = np.where(np.isin(subj, val_s) & valid)[0]
        tr = np.where(np.isin(subj, rest) & ~np.isin(subj, val_s) & valid)[0]
        model = CNNLSTMSleepClassifier()
        w = compute_class_weight("balanced", classes=np.arange(5), y=y[tr])
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
        loader = DataLoader(EpochDS(X, y, tr), batch_size=a.batch, shuffle=True, num_workers=0)
        Xva, best_k, best_state, bad, t0 = normalise_epochs(X[va]), -1.0, None, 0, time.time()
        for ep in range(a.max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); crit(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            vk = cohen_kappa_score(y[va], predict_proba(model, Xva).argmax(1))
            sch.step(vk)
            if vk > best_k:
                best_k, bad = vk, 0
                best_state = {n: p.clone() for n, p in model.state_dict().items()}
            else:
                bad += 1
            print(f"fold {k} epoch {ep + 1:2d}  val_kappa={vk:.4f}  best={best_k:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if bad >= a.patience:
                break
        model.load_state_dict(best_state)
        torch.save(best_state, out / f"fold{k}_model.pth")
        probs = predict_proba(model, normalise_epochs(X[te_all]))
        np.savez(f, idx=te_all, probs=probs, val_kappa=best_k, epochs_run=ep + 1)
        oof[te_all] = probs

    assert not np.isnan(oof[fold_arr >= 0]).any()
    np.save(out / "oof_probs.npy", oof)
    pred = oof.argmax(1)
    cms = {int(s): confusion(y[(subj == s) & valid], pred[(subj == s) & valid]) for s in np.unique(subj)}
    cm = sum(cms.values()); P, R, F = per_class(cm)
    folds = []
    for k in range(a.folds):
        mk = (fold_arr == k) & valid; c = confusion(y[mk], pred[mk])
        folds.append(dict(fold=k, n=int(c.sum()), acc=float(np.trace(c) / c.sum()), kappa=float(kappa_from_cm(c))))
    res = dict(protocol=f"{a.folds}-fold subject-grouped CV, validation-based early stopping, out-of-fold predictions",
               n_subjects=len(cms), n_epochs=int(cm.sum()), acc=float(np.trace(cm) / cm.sum()),
               kappa=float(kappa_from_cm(cm)), **subject_bootstrap(cms),
               precision=P.round(4).tolist(), recall=R.round(4).tolist(), f1=F.round(4).tolist(),
               macro_f1=round(float(F.mean()), 4), cm=cm.tolist(), folds=folds, config=vars(a))
    json.dump(res, open(out / "results_cv.json", "w"), indent=2)
    print(f"\nOut-of-fold: acc={res['acc']:.4f} (95% CI {res['acc_ci'][0]:.3f}-{res['acc_ci'][1]:.3f})  "
          f"kappa={res['kappa']:.4f} (95% CI {res['kappa_ci'][0]:.3f}-{res['kappa_ci'][1]:.3f})")
    print("per-class F1:", dict(zip(STAGES, res["f1"])))


if __name__ == "__main__":
    main()