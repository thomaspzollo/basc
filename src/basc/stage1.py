"""Stage 1: binomial-logistic score model on binary agreement observations.

Each training question i has m_i probes, c_i of which agree with its anchor.
Expanding every question into a positive pseudo-row with weight c_i and a
negative pseudo-row with weight m_i - c_i makes sklearn's weighted logistic
regression maximize the binomial log-likelihood exactly; its Bayes-optimal
prediction is E[S | H], the conditional latent self-consistency.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict_binomial_logistic(
    X_train: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 5000,
) -> np.ndarray:
    """Fit on (X_train, m probes, c agreements) and return P(agree) for X_test."""
    m = np.asarray(m, dtype=float)
    c = np.asarray(c, dtype=float)
    if m.shape != c.shape or m.ndim != 1 or X_train.shape[0] != m.shape[0]:
        raise ValueError(f"shape mismatch: X {X_train.shape}, m {m.shape}, c {c.shape}")
    if np.any(m <= 0) or np.any((c < 0) | (c > m)):
        raise ValueError("require m > 0 and 0 <= c <= m")

    w_pos, w_neg = c, m - c
    keep_pos, keep_neg = w_pos > 0, w_neg > 0
    X_expand = np.concatenate([X_train[keep_pos], X_train[keep_neg]], axis=0)
    y_expand = np.concatenate([np.ones(int(keep_pos.sum()), dtype=int),
                               np.zeros(int(keep_neg.sum()), dtype=int)])
    sw = np.concatenate([w_pos[keep_pos], w_neg[keep_neg]])

    if len(np.unique(y_expand)) < 2:
        # Every probe agreed (or none did): predict that constant rate.
        rate = float(y_expand[0]) if len(y_expand) else 0.5
        return np.full(X_test.shape[0], rate, dtype=float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs")),
    ])
    pipe.fit(X_expand, y_expand, clf__sample_weight=sw)
    return np.clip(pipe.predict_proba(X_test)[:, 1], 0.0, 1.0)
