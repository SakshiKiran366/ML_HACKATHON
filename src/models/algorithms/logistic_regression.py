from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import SEED
from wrappers import SkModel

CONFIG = "StandardScaler + LogisticRegression(L2, C=1.0, class_weight=balanced)"

def build():
    return SkModel(make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)))
