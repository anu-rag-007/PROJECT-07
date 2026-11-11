"""Small numpy/sklearn helpers for subject-grouped cross-validation (no torch needed)."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def kappa_from_cm(cm):
    cm = np.asarray(cm, float); n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def assign_folds(subj, y, n_splits=5, seed=42):
    """subj, y: arrays over VALID epochs. Returns {subject: fold}; a subject never spans folds."""
    out = {}
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=subj)):
        for s in np.unique(subj[te]):
            out[int(s)] = k
    return out


def confusion(y, p, k=5):
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1); R = tp / np.maximum(cm.sum(1), 1)
    return P, R, 2 * P * R / np.maximum(P + R, 1e-12)


def subject_bootstrap(cms, n_boot=2000, seed=0):
    """cms: {subject: 5x5 confusion}. Cluster bootstrap over subjects -> 95% CIs."""
    arr = np.stack(list(cms.values())); rng = np.random.default_rng(seed)
    accs, kaps = [], []
    for _ in range(n_boot):
        cm = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        accs.append(np.trace(cm) / cm.sum()); kaps.append(kappa_from_cm(cm))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(acc_ci=q(accs), kappa_ci=q(kaps))