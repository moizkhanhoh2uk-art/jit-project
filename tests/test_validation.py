"""Tests designed to FAIL when something is wrong (viva: 'when would your tests fail?').

1. Mutation tests      - deliberately broken checkers must be caught by the contract.
2. Hand-calculated oracles - expected values written down by hand, not produced by the code under test.
3. Leakage / sanity    - label-shuffled training must score near chance; vocabulary must come from train only.
4. Known limitations   - xfail(strict=True): documents failures; the test turns RED if someone 'fixes' them silently.
5. Stale-results guard - committed CSVs must equal a fresh recomputation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (documented dependency)

import core
import evaluate
from core import ACTIONS, REJECT, Decision, Grant, check, resource_for

STOP, START = "ec2:StopInstances", "ec2:StartInstances"
R = resource_for(STOP)


# --------------------------------------------------------------- 1. mutation tests
def make_checker(**flaw):
    """Re-implementation of core.check with switchable defects (the 'mutants')."""
    def fn(action, resource, grant, now):
        if action not in ACTIONS: return Decision(False, "unsupported action")
        if not flaw.get("skip_approval") and not grant.approved: return Decision(False, "approval missing")
        lo = (grant.not_before < now) if flaw.get("exclusive_start") else (grant.not_before <= now)
        hi = (now <= grant.expires_at) if flaw.get("inclusive_expiry") else (now < grant.expires_at)
        if not flaw.get("skip_window") and not (lo and hi): return Decision(False, "outside grant window")
        if not flaw.get("skip_action") and action not in grant.actions: return Decision(False, "action not permitted")
        if not flaw.get("skip_resource") and resource not in grant.resources: return Decision(False, "resource not permitted")
        service = action.split(":")[0]
        if not flaw.get("skip_service") and not resource.startswith("arn:aws:" + service + ":"): return Decision(False, "service mismatch")
        tail = resource.split(":::")[-1]
        if not flaw.get("skip_shape"):
            if action == "s3:ListBucket" and "/" in tail: return Decision(False, "bucket resource required")
            if action.startswith("s3:") and action != "s3:ListBucket" and "/" not in tail: return Decision(False, "object resource required")
        return Decision(True, "ok")
    return fn


def contract_failures(fn):
    """Independent expectations. Returns the names of violated expectations."""
    g = Grant(frozenset([STOP]), frozenset([R]), True, 10, 20)
    bad = []
    def expect(name, got, want):
        if got != want: bad.append(name)
    expect("allow_inside_window", fn(STOP, R, g, 15).allowed, True)
    expect("deny_before_start", fn(STOP, R, g, 9.99).allowed, False)
    expect("allow_at_start", fn(STOP, R, g, 10).allowed, True)            # start inclusive
    expect("deny_at_expiry", fn(STOP, R, g, 20).allowed, False)           # expiry exclusive
    expect("deny_unapproved", fn(STOP, R, Grant(g.actions, g.resources, False, 10, 20), 15).allowed, False)
    expect("deny_other_action", fn(START, R, g, 15).allowed, False)
    expect("deny_other_resource", fn(STOP, R + "-x", g, 15).allowed, False)
    wide = Grant(frozenset(ACTIONS), frozenset(["arn:aws:s3:::demo-bucket/report.txt"]), True, 0, 100)
    expect("deny_service_mismatch", fn(STOP, "arn:aws:s3:::demo-bucket/report.txt", wide, 15).allowed, False)
    bucket = Grant(frozenset(ACTIONS), frozenset([resource_for("s3:ListBucket")]), True, 0, 100)
    expect("deny_object_action_on_bucket", fn("s3:GetObject", resource_for("s3:ListBucket"), bucket, 15).allowed, False)
    return bad


MUTANTS = {"inclusive_expiry": "deny_at_expiry", "exclusive_start": "allow_at_start", "skip_approval": "deny_unapproved",
           "skip_action": "deny_other_action", "skip_resource": "deny_other_resource", "skip_window": "deny_before_start",
           "skip_service": "deny_service_mismatch", "skip_shape": "deny_object_action_on_bucket"}


def test_contract_passes_on_real_checker():
    assert contract_failures(check) == []


def test_reimplementation_equals_real_checker():
    """Differential test: the mutant factory with no flaw must agree with core.check everywhere."""
    ok = make_checker(); rng = np.random.default_rng(0)
    for _ in range(500):
        a = rng.choice(ACTIONS + ["bogus"]); res = resource_for(rng.choice(ACTIONS)) + rng.choice(["", "-x"])
        g = Grant(frozenset(rng.choice(ACTIONS, size=rng.integers(0, 9), replace=False)), frozenset([resource_for(rng.choice(ACTIONS))]),
                  bool(rng.integers(0, 2)), float(rng.integers(0, 60)), float(rng.integers(40, 100)))
        now = float(rng.integers(0, 100))
        assert ok(a, res, g, now).allowed == check(a, res, g, now).allowed


@pytest.mark.parametrize("flaw,expected_catch", MUTANTS.items())
def test_every_mutant_is_caught(flaw, expected_catch):
    """If this fails, the test-suite has a blind spot: a broken checker would go unnoticed."""
    assert expected_catch in contract_failures(make_checker(**{flaw: True}))


# ------------------------------------------------- 2. hand-calculated oracles
def test_metrics_match_hand_calculation():
    y = ["s3:GetObject", "s3:GetObject", "s3:PutObject", "s3:PutObject"]
    p = ["s3:GetObject", "s3:PutObject", "s3:PutObject", "s3:PutObject"]
    m = evaluate.metrics(y, p)
    # Get: P=1, R=1/2, F1=2/3.  Put: P=2/3, R=1, F1=4/5.  Six unused classes score 0 -> divide by 8.
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_precision"] == pytest.approx((1 + 2 / 3) / 8)
    assert m["macro_recall"] == pytest.approx(1.5 / 8)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 8)


def rows_by_profile(true, pred):
    return {r["profile"]: r for r in core.security_outcomes(true, pred)}


def test_security_accounting_wrong_prediction_stop_as_start():
    r = rows_by_profile(STOP, START)
    assert r["narrow"]["false_deny"] and r["narrow"]["wrong_prediction_blocked"] and not r["narrow"]["false_allow"]
    assert r["broad"]["allowed_wrong_action"] and not r["broad"]["false_allow"]     # both permitted: error hidden, not a false allow
    assert r["other_actions_only"]["false_allow"] and r["other_actions_only"]["allowed_wrong_action"]
    for p in ["unapproved", "expired", "future", "wrong_resource"]:
        assert r[p]["wrong_prediction_blocked"] and not r[p]["false_allow"] and not r[p]["false_deny"]


def test_security_accounting_correct_prediction_and_reject():
    r = rows_by_profile(STOP, STOP)
    assert r["narrow"]["legitimate_support"] and r["broad"]["legitimate_support"]
    assert not any(x["wrong_prediction"] for x in r.values())
    rej = rows_by_profile(STOP, REJECT)
    assert not any(x["actual_allow"] for x in rej.values())                 # abstention can never authorise
    assert rej["narrow"]["false_deny"] and rej["broad"]["false_deny"]       # ...but it costs legitimate requests


def test_only_the_intended_action_is_used_to_build_grants():
    g, res = core.make_grant("narrow", STOP)
    assert g.actions == frozenset([STOP]) and res == R


# ---------------------------------------------------- 3. leakage / sanity checks
def test_tfidf_vocabulary_comes_from_training_text_only():
    data, model = core.train()
    vec = model.named_steps["tfidf"]; analyzer = vec.build_analyzer()
    train_terms = {t for text in data[data.split == "train"].text for t in analyzer(text)}
    assert set(vec.vocabulary_) == train_terms
    test_only = {t for text in data[data.split == "test"].text for t in analyzer(text)} - train_terms
    assert test_only and not (test_only & set(vec.vocabulary_))        # e.g. 'wipe' must be unknown to the model


def test_shuffled_labels_score_near_chance():
    """If the pipeline leaked labels, shuffled-label accuracy would stay high. Chance is 0.125."""
    data = core.load_data()
    shuffled = data.assign(label=np.random.default_rng(1).permutation(data.label.to_numpy()))
    d = evaluate.grouped_cv(shuffled, "tfidf_lr")
    assert (d.pred == d.label).mean() < 0.25


def test_cv_folds_never_split_a_family_and_fold3_is_the_fixed_test_split():
    data = core.load_data(); folds = data.family.map(core.fold_of)
    assert data.groupby("family").apply(lambda g: folds[g.index].nunique(), include_groups=False).eq(1).all()
    assert set(data[folds == 3].id) == set(data[data.split == "test"].id)


def test_dataset_is_balanced_and_wrapped_four_times():
    d = core.load_data()
    assert d.label.value_counts().eq(32).all() and d.groupby("family").size().eq(4).all()
    assert d.groupby("family").split.nunique().eq(1).all()


def test_training_is_deterministic():
    _, m1 = core.train(); _, m2 = core.train(); t = core.load_data().text[:50]
    assert list(m1.predict(t)) == list(m2.predict(t))


# --------------------------------------------------------- 4. known limitations
@pytest.mark.xfail(strict=True, reason="LIMITATION: negation is not understood by the keyword baseline")
def test_keyword_understands_negation():
    assert core.keyword("Do not stop the instance") != STOP


@pytest.mark.xfail(strict=True, reason="LIMITATION: TF-IDF bag-of-words ignores 'do not'")
def test_tfidf_understands_negation():
    _, m = core.train(); assert m.predict(["Do not stop the instance"])[0] != STOP


@pytest.mark.xfail(strict=True, reason="LIMITATION: forced single label cannot represent two actions")
def test_single_label_represents_multi_intent():
    assert {core.keyword("Download the object and delete the object")} == {"s3:GetObject", "s3:DeleteObject"}


@pytest.mark.xfail(strict=True, reason="LIMITATION: synonym outside the lexicon ('wipe') is missed by the forced keyword baseline")
def test_keyword_handles_unseen_synonym():
    assert core.keyword("Please wipe the stored document") == "s3:DeleteObject"


def test_documented_behaviour_unseen_synonym_is_rejected_when_abstention_is_enabled():
    assert core.keyword_or_reject("Please wipe the stored document") == REJECT


def test_documented_limitation_checker_has_no_wildcards():
    g = Grant(frozenset(ACTIONS), frozenset(["arn:aws:s3:::demo-bucket/*"]), True, 0, 100)
    assert not check("s3:GetObject", resource_for("s3:GetObject"), g, 50).allowed


def test_documented_limitation_broad_grant_cannot_catch_misclassification():
    g = Grant(frozenset(ACTIONS), frozenset([R]), True, 0, 100)
    assert check(START, R, g, 50).allowed        # intended Stop, proposed Start: permitted although wrong


# ------------------------------------------------------ 5. stale-results guard
def test_committed_results_match_recomputation():
    data, model = core.train(); test = data[data.split == "test"]
    saved = pd.read_csv(core.ROOT / "results/classification_metrics.csv").set_index("method")
    for name, yp in [("Keyword", [core.keyword(t) for t in test.text]), ("TFIDF_LR", model.predict(test.text))]:
        assert saved.loc[name, "accuracy"] == pytest.approx((np.array(yp) == test.label.to_numpy()).mean())


def test_security_counts_are_internally_consistent():
    s = pd.read_csv(core.ROOT / "results/security_summary.csv").set_index("method")
    assert (s.wrong_prediction == s.wrong_prediction_blocked + s.allowed_wrong_action).all()
    sec = pd.read_csv(core.ROOT / "results/security_cases.csv")
    assert sec.groupby("method").size().eq(448).all()
