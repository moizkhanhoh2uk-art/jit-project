# Evaluating ML-Based Task Classification for JIT AWS S3 and EC2 Access Control
Moiz Ahmad B01841891 - UWS MSc IT with Cloud Computing. Supervisor: Michael Lin.

A local, synthetic research simulation. **No AWS call is ever made and no credentials are needed.**

| File | Purpose |
|---|---|
| `core.py` | keyword baseline, TF-IDF + Logistic Regression, abstention helper, independent policy checker, 7 policy profiles |
| `evaluate.py` | regenerates every file in `results/` (fixed split, grouped CV, bootstrap CIs, permutation test, lexicon coverage, abstention) |
| `app.py` | Streamlit demonstration |
| `tests/test_project.py` | unit tests for checker, dataset, examples |
| `tests/test_validation.py` | tests designed to fail: mutation tests, hand-calculated oracles, leakage checks, strict-xfail limitations, stale-results guard |
| `data/` | 256 synthetic requests (32 per action, 64 phrase families x 4 wrappers) and 12 challenge cases |
| `docs/` | report, viva preparation, university forms, approved specification |

## Key findings (see report section 6)
* Fixed split (train families 0-5, test families 6-7): keyword 12.5 %, TF-IDF LR 26.6 %, majority baseline 12.5 %.
* Grouped 4-fold CV over all 64 families: keyword 75.0 %, TF-IDF LR 45.7 %. **The ranking reverses**: it depends on vocabulary overlap, so no method is shown to be generally better.
* A permissive policy cannot detect a wrong-but-permitted action; abstention removes false allows only by rejecting almost everything.

## Honest limits
Synthetic, author-written data; single author; templates; 64 families only. Results are not evidence of real-world security. The abstention analysis is an extension beyond the approved specification and needs supervisor agreement.

## Academic integrity
Understand every line before submission. Declare any AI assistance exactly as UWS rules require and complete the declaration yourself.
