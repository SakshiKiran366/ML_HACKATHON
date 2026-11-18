"""Compatibility registry for the five individually defined model implementations."""
from algorithms import decision_tree, hist_gradient_boosting, knn, logistic_regression, random_forest
from wrappers import HGBModel  # kept public for existing experiment/tuning imports

def registry():
    return {
        "logreg": ("Linear / probabilistic", logistic_regression.build, logistic_regression.CONFIG),
        "knn": ("Instance-based", knn.build, knn.CONFIG),
        "decision_tree": ("Tree-based", decision_tree.build, decision_tree.CONFIG),
        "random_forest": ("Ensemble (bagging)", random_forest.build, random_forest.CONFIG),
        "hist_gb": ("Boosting", hist_gradient_boosting.build, hist_gradient_boosting.CONFIG),
    }
