"""Shared fit/predict adapters for the model comparison suite."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import SEED

class SkModel:
    def __init__(self, est, max_train_rows=None):
        self.est, self.max_train_rows, self.history = est, max_train_rows, None

    def fit(self, X, y, X_val=None, y_val=None):
        if self.max_train_rows and len(X) > self.max_train_rows:
            idx = np.random.default_rng(SEED).choice(len(X), self.max_train_rows, replace=False)
            X, y = X[idx], y[idx]
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)[:, 1]

class HGBModel:
    """Choose boosting rounds on held-out trips, then refit at that round count."""
    def __init__(self, **params):
        self.params = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
                           max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
                           early_stopping=False, random_state=SEED)
        self.params.update(params)
        self.m, self.history = None, None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.metrics import log_loss
        full = HistGradientBoostingClassifier(**self.params).fit(X, y)
        if X_val is None:
            self.m = full
            return self
        sub = np.random.default_rng(SEED).choice(len(X), min(len(X), 60_000), replace=False)
        train_loss = [log_loss(y[sub], p[:, 1], labels=[0, 1]) for p in full.staged_predict_proba(X[sub])]
        val_loss = [log_loss(y_val, p[:, 1], labels=[0, 1]) for p in full.staged_predict_proba(X_val)]
        best = int(np.argmin(val_loss)) + 1
        self.history = {"train_loss": train_loss, "val_loss": val_loss, "best_iteration": best}
        self.m = HistGradientBoostingClassifier(**{**self.params, "max_iter": best}).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]
