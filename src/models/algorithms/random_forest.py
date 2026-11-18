from sklearn.ensemble import RandomForestClassifier
from common import SEED
from wrappers import SkModel

CONFIG = "RandomForest(200 trees, min_leaf=5, max_samples=0.5, balanced_subsample)"

def build():
    return SkModel(RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
        max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
