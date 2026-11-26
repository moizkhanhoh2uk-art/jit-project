"""Offline intent classifiers and an independent exact-match policy simulation.

Nothing here contacts AWS. The policy checker (`check`) never receives the
request sentence, a model probability or the intended label.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SEED = 42
REJECT = "REJECT"  # abstention label; it is never a member of ACTIONS

ACTIONS = [
    "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    "ec2:DescribeInstances", "ec2:StartInstances", "ec2:RebootInstances", "ec2:StopInstances",
]
# One keyword list per action, in the same order as ACTIONS.
RULES = [
    ["download", "read", "fetch", "retrieve", "get", "open"],
    ["upload", "write", "store", "save", "put", "send"],
    ["delete", "remove", "erase", "discard", "destroy", "purge"],
    ["list", "enumerate", "browse"],
    ["describe", "inspect", "check", "status", "details"],
    ["start", "launch", "power on", "boot", "activate", "turn on"],
    ["reboot", "restart", "power cycle", "relaunch", "reset"],
    ["stop", "shutdown", "power off", "halt", "deactivate", "turn off"],
]


# ---------------------------------------------------------------- classifiers
def keyword_scores(text):
    """Number of distinct keywords of each action found as whole words/phrases."""
    lowered = text.lower()
    return [
        sum(bool(re.search(r"\b" + re.escape(w) + r"\b", lowered)) for w in words)
        for words in RULES
    ]


def keyword(text):
    """Forced-choice keyword baseline. Ties and all-zero scores fall to ACTIONS order."""
    scores = keyword_scores(text)
    return ACTIONS[max(range(len(ACTIONS)), key=lambda i: scores[i])]


def keyword_or_reject(text):
    """Keyword baseline that abstains when no keyword matches at all."""
    return REJECT if not any(keyword_scores(text)) else keyword(text)


def make_model(kind="tfidf_lr"):
    """`tfidf_lr` is the approved model. `char_lr` is an exploratory sensitivity check."""
    if kind == "tfidf_lr":
        vec = TfidfVectorizer(ngram_range=(1, 2))
    elif kind == "char_lr":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    else:
        raise ValueError(kind)
    return Pipeline([("tfidf", vec), ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])


def load_data():
    return pd.read_csv(ROOT / "data/requests.csv")


def train(kind="tfidf_lr"):
    """Fit on the training partition only and return (all data, fitted model)."""
    data = load_data()
    tr = data[data.split == "train"]
    model = make_model(kind)
    model.fit(tr.text, tr.label)
    return data, model


def predict_or_reject(model, texts, threshold):
    """Predict, but return REJECT when the top class probability is below threshold."""
    proba = model.predict_proba(list(texts))
    labels = model.classes_[proba.argmax(axis=1)]
    return [REJECT if p.max() < threshold else lab for p, lab in zip(proba, labels)]


def family_index(family):
    """'s3:GetObject-3' -> 3"""
    return int(family.rsplit("-", 1)[1])


def fold_of(family):
    """Grouped 4-fold CV: families (0,1), (2,3), (4,5), (6,7) of every class form one fold.
    Fold 3 is exactly the original fixed test split."""
    return family_index(family) // 2


# ------------------------------------------------------------- policy checker
@dataclass(frozen=True)
class Grant:
    actions: frozenset
    resources: frozenset
    approved: bool
    not_before: float
    expires_at: float


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def check(action, resource, grant, now):
    """Allow only if every condition holds. Window is [not_before, expires_at).
    No wildcard, deny-precedence or full-ARN semantics are modelled."""
    if action not in ACTIONS:
        return Decision(False, "unsupported action")
    if not grant.approved:
        return Decision(False, "approval missing")
    if not grant.not_before <= now < grant.expires_at:
        return Decision(False, "outside grant window")
    if action not in grant.actions:
        return Decision(False, "action not permitted")
    if resource not in grant.resources:
        return Decision(False, "resource not permitted")
    service = action.split(":")[0]
    if not resource.startswith("arn:aws:" + service + ":"):
        return Decision(False, "service mismatch")
    tail = resource.split(":::")[-1]
    if action == "s3:ListBucket" and "/" in tail:
        return Decision(False, "bucket resource required")
    if action.startswith("s3:") and action != "s3:ListBucket" and "/" not in tail:
        return Decision(False, "object resource required")
    return Decision(True, "trusted grant permits action and resource")


def resource_for(action):
    if action == "s3:ListBucket":
        return "arn:aws:s3:::demo-bucket"
    if action.startswith("s3:"):
        return "arn:aws:s3:::demo-bucket/report.txt"
    return "arn:aws:ec2:eu-west-2:000000000000:instance/i-demo"


PROFILES = ["narrow", "broad", "other_actions_only", "unapproved", "expired", "future", "wrong_resource"]


def make_grant(profile, intended):
    """Seven fixed trusted-policy profiles built from the INTENDED action only."""
    r = resource_for(intended)
    everything, others = frozenset(ACTIONS), frozenset(set(ACTIONS) - {intended})
    table = {
        "narrow": (frozenset([intended]), True, 0, 100, r),
        "broad": (everything, True, 0, 100, r),
        "other_actions_only": (others, True, 0, 100, r),
        "unapproved": (everything, False, 0, 100, r),
        "expired": (everything, True, 0, 50, r),
        "future": (everything, True, 60, 100, r),
        "wrong_resource": (everything, True, 0, 100, r + "-other"),
    }
    acts, approved, start, end, granted_resource = table[profile]
    return Grant(acts, frozenset([granted_resource]), approved, start, end), r


def security_outcomes(true_label, predicted, now=50):
    """Per-request policy outcomes over all profiles (REJECT is always denied)."""
    rows = []
    for profile in PROFILES:
        grant, resource = make_grant(profile, true_label)
        expected = check(true_label, resource, grant, now).allowed
        actual = check(predicted, resource, grant, now)
        wrong = predicted != true_label
        rows.append(dict(
            profile=profile, expected_allow=expected, actual_allow=actual.allowed,
            false_allow=actual.allowed and not expected,
            false_deny=expected and not actual.allowed,
            wrong_prediction=wrong,
            wrong_prediction_blocked=wrong and not actual.allowed,
            allowed_wrong_action=wrong and actual.allowed,
            legitimate_support=expected and actual.allowed and not wrong,
            reason=actual.reason))
    return rows
