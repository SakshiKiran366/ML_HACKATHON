"""The five experimental models (one per algorithm family named in the Models & Training Guidelines).

Every wrapper exposes  fit(X, y, X_val=None, y_val=None)  and  predict_proba(X) -> P(label=1)
and keeps `.history` (loss per boosting round) only where the algorithm really produces one.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
    """HistGradientBoosting. The number of rounds is chosen on held-out TRIPS (not random rows):
    pass 1 fits max_iter rounds and records train/val log loss per round via staged_predict_proba,
    pass 2 refits with the best round count. History is real (computed, not simulated)."""

    def __init__(self, **params):
        self.params = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31, max_depth=None,
                           min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                           random_state=SEED)
        self.params.update(params)
        self.m, self.history = None, None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.metrics import log_loss
        full = HistGradientBoostingClassifier(**self.params).fit(X, y)
        if X_val is None:
            self.m = full
            return self
        sub = np.random.default_rng(SEED).choice(len(X), min(len(X), 60_000), replace=False)
        tr = [log_loss(y[sub], p[:, 1], labels=[0, 1]) for p in full.staged_predict_proba(X[sub])]
        va = [log_loss(y_val, p[:, 1], labels=[0, 1]) for p in full.staged_predict_proba(X_val)]
        best = int(np.argmin(va)) + 1
        self.history = {"train_loss": tr, "val_loss": va, "best_iteration": best}
        self.m = HistGradientBoostingClassifier(**{**self.params, "max_iter": best}).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


# name -> (family, factory, human-readable config)
def registry():
    return {
        "logreg": ("Linear / probabilistic",
                   lambda: SkModel(make_pipeline(StandardScaler(), LogisticRegression(
                       C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED))),
                   "StandardScaler + LogisticRegression(L2, C=1.0, class_weight=balanced)"),
        "knn": ("Instance-based",
                lambda: SkModel(make_pipeline(StandardScaler(), KNeighborsClassifier(
                    n_neighbors=25, weights="distance", n_jobs=-1)), max_train_rows=100_000),
                "StandardScaler + KNN(k=25, distance-weighted), train subsample cap 100k rows"),
        "decision_tree": ("Tree-based",
                          lambda: SkModel(DecisionTreeClassifier(
                              max_depth=14, min_samples_leaf=30, class_weight="balanced", random_state=SEED)),
                          "DecisionTree(max_depth=14, min_samples_leaf=30, class_weight=balanced, gini)"),
        "random_forest": ("Ensemble (bagging)",
                          lambda: SkModel(RandomForestClassifier(
                              n_estimators=200, min_samples_leaf=5, max_samples=0.5,
                              class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
                          "RandomForest(200 trees, min_leaf=5, max_samples=0.5, balanced_subsample)"),
        "hist_gb": ("Boosting", lambda: HGBModel(),
                    "HistGradientBoosting(lr=0.1, 31 leaves, min_leaf=40, l2=1.0, rounds chosen on held-out trips, <=400)"),
    }
