"""The five experimental models (one per algorithm family).

Every wrapper exposes:  fit(X, y, X_val=None, y_val=None)  and  predict_proba(X) -> P(label=1)
and keeps `.history` (per-round / per-epoch losses) only when the algorithm really produces one.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


class XGBModel:
    """Gradient boosting on GPU. Early stopping on held-out trips; keeps per-round logloss."""

    def __init__(self, **params):
        import xgboost as xgb
        self.params = dict(n_estimators=800, learning_rate=0.08, max_depth=6, min_child_weight=1,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                           tree_method="hist", device="cuda", random_state=SEED,
                           eval_metric="logloss", early_stopping_rounds=40)
        self.params.update(params)
        self.m = xgb.XGBClassifier(**self.params)
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None:
            self.m.fit(X, y, eval_set=[(X, y), (X_val, y_val)], verbose=False)
            r = self.m.evals_result()
            self.history = {"train_loss": r["validation_0"]["logloss"],
                            "val_loss": r["validation_1"]["logloss"],
                            "best_iteration": int(self.m.best_iteration)}
        else:
            p = dict(self.params); p.pop("early_stopping_rounds", None)
            import xgboost as xgb
            self.m = xgb.XGBClassifier(**p).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class TorchMLP:
    """Shallow MLP, BCE loss, Adam, early stopping on held-out trips; keeps per-epoch loss."""

    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-3, batch=2048, epochs=40,
                 patience=6, weight_decay=1e-5):
        self.cfg = dict(hidden=list(hidden), dropout=dropout, lr=lr, batch=batch, epochs=epochs,
                        patience=patience, weight_decay=weight_decay)
        self.history, self.net = None, None

    def _build(self, d):
        import torch.nn as nn
        layers, last = [], d
        for h in self.cfg["hidden"]:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(self.cfg["dropout"])]
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        torch.manual_seed(SEED)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        if X_val is not None:
            Xv = torch.tensor((X_val - self.mu) / self.sd, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32)
        self.net = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        lossf = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(SEED)
        hist = {"train_loss": [], "val_loss": []}
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.cfg["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(Xt), self.cfg["batch"]):
                b = perm[i:i + self.cfg["batch"]]
                opt.zero_grad()
                loss = lossf(self.net(Xt[b]).squeeze(1), yt[b])
                loss.backward(); opt.step()
                tot += loss.item() * len(b)
            hist["train_loss"].append(tot / len(Xt))
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = lossf(self.net(Xv).squeeze(1), yv).item()
                hist["val_loss"].append(vl)
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.cfg["patience"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = hist
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return torch.sigmoid(self.net(Xt).squeeze(1)).numpy()


# name -> (family, factory, human-readable config)
def registry():
    return {
        "logreg": ("Linear / probabilistic",
                   lambda: SkModel(make_pipeline(StandardScaler(), LogisticRegression(
                       C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED))),
                   "StandardScaler + LogisticRegression(L2, C=1.0, class_weight=balanced)"),
        "knn": ("Instance-based",
                lambda: SkModel(make_pipeline(StandardScaler(), KNeighborsClassifier(
                    n_neighbors=25, weights="distance", n_jobs=-1)), max_train_rows=300_000),
                "StandardScaler + KNN(k=25, distance-weighted), train subsample cap 300k rows"),
        "random_forest": ("Ensemble (bagging)",
                          lambda: SkModel(RandomForestClassifier(
                              n_estimators=200, min_samples_leaf=5, max_samples=0.5,
                              class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
                          "RandomForest(200 trees, min_leaf=5, max_samples=0.5, balanced_subsample)"),
        "xgboost": ("Boosting", lambda: XGBModel(),
                    "XGBoost(GPU hist, lr=0.08, depth=6, <=800 rounds, early stop 40, subsample/colsample 0.8)"),
        "mlp": ("Neural network", lambda: TorchMLP(),
                "MLP(128-64, ReLU, dropout 0.1, Adam lr=1e-3, batch 2048, early stop 6, BCE loss)"),
    }
