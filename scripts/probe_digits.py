"""Feature steering, step 1 (LIVE.md, 16 Nov 2026): can a linear probe read each display digit from Krea's hidden states
at its place (runner/feature_probe.py, replay_probe)? If not, steering has nothing to shift and stops here.

Rows: one per latent frame and digit place, plus a control row on the keypad (runner/feature_probe.py CONTROL);
label = the 7-segment reader's digit ("0".."9", " " blank) on the latent's frames, kept only when its frames agree.
code-probe-r1: the same numbers run on every seed and the prompt names the number, so folds hold out whole numbers
(grouped by the number shown), never only a seed. Per layer and denoising call:
- display: accuracy on held-out numbers of a multinomial logistic regression (standardised, L2), against the majority;
- control: the same on the keypad rows (labelled with the place's digit); as good as the display = not held locally;
- neighbours: a probe given only the left and right places' labels (one-hot), the same folds;
- place transfer: fitted on places 0-2, tested on places 3-4 (a shared digit direction, what steering would add);
- drawn vs typed: on rows where Krea drew another digit than the one typed there, which one the probe predicts.
Call 0 is the first step (t=1000, the block's input pure noise): a digit read there comes from the memory and the
prompt, not from the block's own pixels. Class means per layer and call (step 2's directions) go to data/probe/means.npz.
Writes data/scores/probe_digits.json.
Usage: uv run --with scikit-learn python scripts/probe_digits.py --run logs/live/<launch> [--arm replay_P_probe]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = list("0123456789") + [" "]


def load(run: Path, arm: str) -> tuple[np.ndarray, list[dict], dict]:
    feats, rows, meta = [], [], None
    for p in sorted(run.glob(f"{arm}-s*_probe.npz")):
        z = np.load(p)
        m = json.loads(str(z["meta"]))
        meta = meta or m
        rs = json.loads(str(z["rows"]))
        for r in rs:
            r["seed"] = m["seed"]
        feats.append(z["feats"])
        rows += rs
    if not rows:
        raise SystemExit(f"no {arm}-s*_probe.npz in {run}")
    return np.concatenate(feats).astype(np.float32), rows, meta


def keep(r: dict) -> bool:
    labs = [x for x in r["labels4"] if x is not None]
    return r["label"] in CLASSES and len(labs) >= 2 and all(x == r["label"] for x in labs)


def folds(groups: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """Held-out masks: the distinct numbers split into k sets."""
    names = sorted(set(groups))
    return [np.isin(groups, names[i::k]) for i in range(k)]


def cv_predict(f: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(y), "x", dtype=object)
    for te in folds(groups):
        tr = ~te
        if len(set(y[tr])) < 2 or not te.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(f[tr], y[tr])
        pred[te] = clf.predict(f[te])
    return pred


def acc(pred: np.ndarray, y: np.ndarray) -> float:
    m = pred != "x"
    return round(float((pred[m] == y[m]).mean()), 3) if m.any() else None


def majority_cv(y: np.ndarray, groups: np.ndarray) -> float:
    hits = n = 0
    for te in folds(groups):
        vals, counts = np.unique(y[~te], return_counts=True)
        hits += int((y[te] == vals[counts.argmax()]).sum())
        n += int(te.sum())
    return round(hits / max(n, 1), 3)


def neighbour_features(rows: list[dict]) -> np.ndarray:
    """One-hot of the left and right places' labels in the same latent frame (none: all zero)."""
    at = {(r["seed"], r["block"], r["latent"], r["place"]): r["label"] for r in rows}
    out = np.zeros((len(rows), 2 * len(CLASSES)), np.float32)
    for i, r in enumerate(rows):
        for side, dp in enumerate((-1, 1)):
            lab = at.get((r["seed"], r["block"], r["latent"], r["place"] + dp))
            if lab in CLASSES:
                out[i, side * len(CLASSES) + CLASSES.index(lab)] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", default="replay_P_probe")
    a = ap.parse_args()
    x, rows, meta = load(a.run, a.arm)
    ok = np.array([keep(r) for r in rows])
    x, rows = x[ok], [r for r, k in zip(rows, ok) if k]
    ctrl = np.array([bool(r.get("control")) for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["shown"] for r in rows])
    place = np.array([r["place"] for r in rows])
    typed_digit = np.array([(r["shown"].rjust(5)[r["place"]] if r["place"] < 5 else "?") for r in rows])  # r2 nit 1: the number written
    disp = ~ctrl
    layers, calls = meta["layers"], meta["calls"]
    res: dict = {"rows_display": int(disp.sum()), "rows_control": int(ctrl.sum()), "dropped": int((~ok).sum()),
                 "classes": {c: int((y[disp] == c).sum()) for c in CLASSES}, "numbers": len(set(groups)),
                 "skips": meta.get("skips"), "layers": layers, "calls": calls,
                 "majority": majority_cv(y[disp], groups[disp]), "probe": {}}
    drows = [r for r, d in zip(rows, disp) if d]
    nb = cv_predict(neighbour_features(drows), y[disp], groups[disp])
    res["neighbours_only"] = acc(nb, y[disp])
    print(f"{int(disp.sum())} display rows, {int(ctrl.sum())} control rows ({res['dropped']} dropped), "
          f"{res['numbers']} numbers; majority {res['majority']}, neighbours' labels only {res['neighbours_only']}")
    means = {}
    mism = disp & (typed_digit != y) & np.isin(typed_digit, CLASSES)
    for li, layer in enumerate(layers):
        for c in range(calls):
            f = x[:, li, c]
            pd_ = cv_predict(f[disp], y[disp], groups[disp])
            pc = cv_predict(f[ctrl], y[ctrl], groups[ctrl]) if ctrl.any() else None
            tr, te = disp & (place <= 2), disp & (place >= 3)
            tp = None
            if len(set(y[tr])) > 1 and te.any():
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                tp = round(float((make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
                                  .fit(f[tr], y[tr]).predict(f[te]) == y[te]).mean()), 3)
            full = np.full(len(y), "x", dtype=object)
            full[disp] = pd_
            m = mism & (full != "x")
            key = f"layer{layer}_call{c}"
            res["probe"][key] = {"display": acc(pd_, y[disp]), "control": acc(pc, y[ctrl]) if pc is not None else None,
                                 "place_transfer": tp, "mismatch_rows": int(m.sum()),
                                 "mismatch_predicts_drawn": round(float((full[m] == y[m]).mean()), 3) if m.any() else None,
                                 "mismatch_predicts_typed": round(float((full[m] == typed_digit[m]).mean()), 3) if m.any() else None}
            r = res["probe"][key]
            print(f"{key:16s} display {r['display']}  control {r['control']}  places 0-2 -> 3-4 {r['place_transfer']}  "
                  f"drawn/typed on {r['mismatch_rows']} mismatches {r['mismatch_predicts_drawn']}/{r['mismatch_predicts_typed']}")
            fd = f[disp]
            means[key] = np.stack([fd[y[disp] == cl].mean(0) if (y[disp] == cl).any() else np.zeros(f.shape[1])
                                   for cl in CLASSES])
    out = ROOT / "data/probe"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "means.npz", classes=np.array(CLASSES), **means)
    (ROOT / "data/scores/probe_digits.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
