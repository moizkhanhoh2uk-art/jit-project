"""Reproducible experiments. Run: python evaluate.py   (regenerates everything in results/)."""
import hashlib
import importlib.metadata
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

import core
from core import ACTIONS, REJECT, ROOT, SEED

OUT = ROOT / "results"
N_BOOT, N_PERM = 2000, 200
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=ACTIONS, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), macro_precision=pr, macro_recall=rc, macro_f1=f1, n=len(y))


def cluster_bootstrap(df, cols, rng):
    """Resample whole phrase families (the independent unit), not individual rows."""
    fam = df.groupby("family")[cols].sum()
    n = df.groupby("family").size().to_numpy()
    vals = fam.to_numpy(dtype=float)
    idx = rng.integers(0, len(fam), size=(N_BOOT, len(fam)))
    return vals[idx].sum(axis=1) / n[idx].sum(axis=1, keepdims=True)  # (N_BOOT, len(cols)) accuracies


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def confusion_png(y, p, name, title):
    cm = confusion_matrix(y, p, labels=ACTIONS)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=[a.split(":")[1] for a in ACTIONS]).plot(
        ax=ax, xticks_rotation=60, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / (name + "_confusion.png"), dpi=140)
    plt.close(fig)


def grouped_cv(data, kind):
    """Out-of-fold predictions + max probabilities; every row is predicted by a model that never saw its family."""
    data = data.copy()
    data["fold"] = data.family.map(core.fold_of)
    pred, conf = pd.Series(index=data.index, dtype=object), pd.Series(index=data.index, dtype=float)
    for k in sorted(data.fold.unique()):
        tr, te = data[data.fold != k], data[data.fold == k]
        m = core.make_model(kind).fit(tr.text, tr.label)
        proba = m.predict_proba(te.text)
        pred[te.index] = m.classes_[proba.argmax(axis=1)]
        conf[te.index] = proba.max(axis=1)
    return data.assign(pred=pred, conf=conf)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    data, model = core.train()
    test = data[data.split == "test"].copy()
    train_df = data[data.split == "train"]

    # ---- 1. approved fixed-split comparison (identical protocol to the supervisor draft) ----
    majority = train_df.label.value_counts().sort_index().index[0]
    methods = {
        "Keyword": [core.keyword(t) for t in test.text],
        "TFIDF_LR": list(model.predict(test.text)),
        "Majority_baseline": [majority] * len(test),
    }
    rows, preds, sec, chal = [], [], [], []
    for name, yp in methods.items():
        rows.append(dict(method=name, **metrics(test.label, yp)))
        if name != "Majority_baseline":
            confusion_png(test.label, yp, name, name + " held out requests")
        for (_, r), p in zip(test.iterrows(), yp):
            preds.append(dict(method=name, id=r.id, text=r.text, true_action=r.label, predicted_action=p, correct=p == r.label))
            if name != "Majority_baseline":
                for o in core.security_outcomes(r.label, p):
                    sec.append(dict(method=name, id=r.id, true_action=r.label, predicted_action=p, **o))
    for name, fn in [("Keyword", core.keyword), ("TFIDF_LR", lambda t: model.predict([t])[0])]:
        for case in json.loads((ROOT / "data/challenges.json").read_text()):
            p = fn(case["text"]); intended = case["intended_actions"]
            chal.append(dict(method=name, **case, predicted_action=p,
                             single_action_suitable=len(intended) == 1 and p == intended[0]))
    pd.DataFrame(rows).to_csv(OUT / "classification_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    sec = pd.DataFrame(sec); sec.to_csv(OUT / "security_cases.csv", index=False)
    cols = ["false_allow", "false_deny", "wrong_prediction", "wrong_prediction_blocked", "allowed_wrong_action", "legitimate_support"]
    summary = sec.groupby("method")[cols].sum(); summary.to_csv(OUT / "security_summary.csv")
    pd.DataFrame(chal).to_json(OUT / "challenge_results.json", orient="records", indent=2)

    # ---- 2. robustness: grouped 4-fold CV over ALL 64 phrase families + cluster bootstrap ----
    cv_rows, cv_pred = [], []
    kw_pred = data.assign(pred=[core.keyword(t) for t in data.text], conf=np.nan)
    oof = {"Keyword": kw_pred, "TFIDF_LR": grouped_cv(data, "tfidf_lr"), "CHAR_LR_exploratory": grouped_cv(data, "char_lr")}
    corr = {}
    for name, d in oof.items():
        d = d.assign(correct=(d.pred == d.label).astype(int)); corr[name] = d
        lo, hi = ci(cluster_bootstrap(d, ["correct"], rng)[:, 0])
        per_fold = d.groupby(d.family.map(core.fold_of)).correct.mean().round(4).tolist()
        cv_rows.append(dict(method=name, **metrics(d.label, d.pred), acc_ci95_low=lo, acc_ci95_high=hi,
                            families=d.family.nunique(), per_fold_accuracy=str(per_fold)))
        cv_pred.append(d.assign(method=name)[["method", "id", "family", "label", "pred", "conf", "correct"]])
    pd.DataFrame(cv_rows).to_csv(OUT / "cv_metrics.csv", index=False)
    pd.concat(cv_pred).to_csv(OUT / "cv_predictions.csv", index=False)

    # paired difference TF-IDF minus Keyword, resampling the same families (CV and fixed split)
    paired = []
    for scope, sel in [("grouped_cv_64_families", lambda d: d), ("fixed_split_16_families", lambda d: d[d.split == "test"])]:
        a, b = sel(corr["TFIDF_LR"]), sel(corr["Keyword"])
        both = a[["id", "family", "correct"]].merge(b[["id", "correct"]], on="id", suffixes=("_tf", "_kw"))
        both["diff"] = both.correct_tf - both.correct_kw
        diffs = cluster_bootstrap(both, ["diff"], np.random.default_rng(SEED))[:, 0]
        lo, hi = ci(diffs)
        paired.append(dict(scope=scope, tfidf_minus_keyword=both["diff"].mean(), ci95_low=lo, ci95_high=hi))
    pd.DataFrame(paired).to_csv(OUT / "paired_difference.csv", index=False)

    # label-permutation test: is grouped-CV accuracy better than chance-by-construction?
    obs = float(corr["TFIDF_LR"].correct.mean()); perm_acc = []
    for _ in range(N_PERM):
        shuffled = data.assign(label=rng.permutation(data.label.to_numpy()))
        d = grouped_cv(shuffled, "tfidf_lr"); perm_acc.append(float((d.pred == d.label).mean()))
    (OUT / "permutation_test.json").write_text(json.dumps(dict(
        observed_cv_accuracy=obs, permutations=N_PERM, permuted_mean=float(np.mean(perm_acc)),
        permuted_max=float(np.max(perm_acc)), p_value=float((1 + sum(a >= obs for a in perm_acc)) / (N_PERM + 1))), indent=2))

    # ---- 3. why does the keyword baseline fail? lexicon coverage (out-of-vocabulary) ----
    cov = []
    for scope, d in [("train", data[data.split == "train"]), ("test", test), ("all_256_cv", data)]:
        hit = d.text.map(lambda t: any(core.keyword_scores(t)))
        acc_all = np.mean([core.keyword(t) == l for t, l in zip(d.text, d.label)])
        acc_cov = np.mean([core.keyword(t) == l for t, l in zip(d.text[hit], d.label[hit])]) if hit.any() else np.nan
        cov.append(dict(scope=scope, n=len(d), lexicon_coverage=hit.mean(), forced_accuracy=acc_all, accuracy_when_covered=acc_cov))
    pd.DataFrame(cov).to_csv(OUT / "keyword_coverage.csv", index=False)

    # ---- 4. extension (needs supervisor approval): abstention on low-confidence predictions ----
    ab, ab_sec = [], []
    for t in THRESHOLDS:
        d = corr["TFIDF_LR"]; kept = d.conf >= t
        ab.append(dict(threshold=t, coverage=kept.mean(), accuracy_on_answered=d.correct[kept].mean() if kept.any() else np.nan))
        p = core.predict_or_reject(model, test.text, t)
        s = pd.DataFrame([o for (_, r), pr in zip(test.iterrows(), p) for o in core.security_outcomes(r.label, pr)])
        ab_sec.append(dict(threshold=t, rejected=sum(x == REJECT for x in p), **s[cols].sum().to_dict()))
    pd.DataFrame(ab).to_csv(OUT / "abstention_curve.csv", index=False)
    pd.DataFrame(ab_sec).to_csv(OUT / "abstention_security.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); a = pd.DataFrame(ab)
    ax.plot(a.coverage, a.accuracy_on_answered, "o-"); ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Accuracy on answered"); ax.set_title("TF-IDF LR abstention curve (grouped CV)"); fig.tight_layout()
    fig.savefig(OUT / "abstention_curve.png", dpi=140); plt.close(fig)

    versions = {k: importlib.metadata.version(k) for k in ["scikit-learn", "pandas", "numpy", "matplotlib", "pytest"]}
    try: versions["streamlit"] = importlib.metadata.version("streamlit")
    except importlib.metadata.PackageNotFoundError: versions["streamlit"] = "not installed in this run"
    (OUT / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), packages=versions, seed=SEED, bootstrap=N_BOOT, permutations=N_PERM,
        dataset_sha256=hashlib.sha256((ROOT / "data/requests.csv").read_bytes()).hexdigest()), indent=2))
    for f in ["classification_metrics", "cv_metrics", "paired_difference", "keyword_coverage", "abstention_curve", "abstention_security"]:
        print("\n==", f); print(pd.read_csv(OUT / (f + ".csv")).round(4).to_string(index=False))
    print("\n==", "security_summary"); print(summary.to_string()); print((OUT / "permutation_test.json").read_text())


if __name__ == "__main__":
    main()
