r"""
Clean preprocessing: Sleep-EDF (Fpz-Cz) -> contiguous per-recording epoch arrays with IDs.

What this fixes compared with the earlier notebooks
  * Labels are placed by annotation ONSET. Before, labels were appended in order and unknown or
    movement annotations were skipped, which shifts every later label against its signal epoch.
  * Unknown/movement epochs stay in the arrays with label -1, so sequences remain contiguous
    (needed to replay a trigger over a whole night). Training simply ignores label -1.
  * Every epoch carries recording / subject / night IDs, so splits can be made BY SUBJECT.
  * The 0.3-35 Hz 4th-order Butterworth band-pass is explicit and causal, like a live system.

Usage
  python prep_v2.py --data_dir <folder with *-PSG.edf and *Hypnogram.edf> --out_dir ..\data\processed_v2
Output: X.npy (N,3000 float32, microvolts), y.npy (int8, -1 = unknown), meta.npz, recordings.json
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt

EPOCH_S, CHANNEL = 30, "EEG Fpz-Cz"
STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def labels_from_annotations(onsets, durations, descriptions, n_epochs, offset_s=0.0):
    """One label per 30 s epoch, placed by onset time. Anything not a sleep stage stays -1."""
    lab = np.full(n_epochs, -1, dtype=np.int8)
    for on, du, d in zip(onsets, durations, descriptions):
        start, n = int(round((on + offset_s) / EPOCH_S)), int(round(du / EPOCH_S))
        if start < 0:
            n, start = n + start, 0
        if n <= 0 or start >= n_epochs:
            continue
        lab[start:start + n] = STAGE_MAP.get(d, -1)
    return lab


def bandpass_causal(x, fs, lo=0.3, hi=35.0, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x - x.mean())


def process_recording(psg_path, hyp_path):
    import mne
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if CHANNEL not in raw.ch_names:
        return None
    fs = raw.info["sfreq"]
    if abs(fs - 100.0) > 1e-6:
        print(f"  skipped: sampling rate {fs} Hz (expected 100)")
        return None
    sig = raw.get_data(picks=[CHANNEL])[0] * 1e6                      # volts -> microvolts
    sig = bandpass_causal(sig, fs)
    spe = int(EPOCH_S * fs)
    n_ep = len(sig) // spe
    X = sig[: n_ep * spe].reshape(n_ep, spe).astype(np.float32)
    ann = mne.read_annotations(hyp_path)
    off = 0.0
    if ann.orig_time is not None and raw.info["meas_date"] is not None:
        off = (ann.orig_time - raw.info["meas_date"]).total_seconds()
    y = labels_from_annotations(ann.onset, ann.duration, ann.description, n_ep, off)
    nw = np.where(y > 0)[0]                                           # crop to the sleep period
    if len(nw) == 0:
        return None
    return X[nw[0]: nw[-1] + 1], y[nw[0]: nw[-1] + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    tmp = out / "tmp"; tmp.mkdir(parents=True, exist_ok=True)
    hyps = sorted(data.glob("*Hypnogram.edf"))
    subj_ids, recs = {}, []
    for p in sorted(data.glob("*-PSG.edf")):
        m = re.match(r"S[CT]\d(\d\d)(\d)", p.name)
        if not m:
            continue
        rec = p.name[:6]
        h = [x for x in hyps if x.name.startswith(rec)]
        if not h:
            print(f"{rec}: no hypnogram"); continue
        res = process_recording(str(p), str(h[0]))
        if res is None:
            print(f"{rec}: skipped"); continue
        X, y = res
        np.save(tmp / f"{rec}_X.npy", X); np.save(tmp / f"{rec}_y.npy", y)
        key = p.name[:2] + m.group(1)                                 # e.g. SC00
        recs.append(dict(rec=rec, subject_key=key, subj=subj_ids.setdefault(key, len(subj_ids)),
                         night=int(m.group(2)), n=int(len(y))))
        print(f"{rec}: {len(y)} epochs, {int((y == -1).sum())} unknown/movement", flush=True)

    total = sum(r["n"] for r in recs)
    Xall = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(total, 3000))
    yall = np.empty(total, np.int8)
    rec_idx, subj, night, eidx = (np.empty(total, np.int32) for _ in range(4))
    pos = 0
    for i, r in enumerate(recs):
        n = r["n"]
        Xall[pos:pos + n] = np.load(tmp / f"{r['rec']}_X.npy")
        yall[pos:pos + n] = np.load(tmp / f"{r['rec']}_y.npy")
        rec_idx[pos:pos + n], subj[pos:pos + n], night[pos:pos + n] = i, r["subj"], r["night"]
        eidx[pos:pos + n] = np.arange(n)
        pos += n
        for s in ("X", "y"):
            (tmp / f"{r['rec']}_{s}.npy").unlink()
    Xall.flush()
    np.save(out / "y.npy", yall)
    np.savez(out / "meta.npz", rec_idx=rec_idx, subj=subj, night=night, epoch_idx=eidx)
    json.dump(recs, open(out / "recordings.json", "w"), indent=2)
    print(f"\nDone: {len(recs)} recordings, {len(subj_ids)} subjects, {total} epochs")


if __name__ == "__main__":
    main()