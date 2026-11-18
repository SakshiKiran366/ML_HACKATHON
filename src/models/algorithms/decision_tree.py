from sklearn.tree import DecisionTreeClassifier
from common import SEED
from wrappers import SkModel

CONFIG = "DecisionTree(max_depth=14, min_samples_leaf=30, class_weight=balanced, gini)"

def build():
    return SkModel(DecisionTreeClassifier(max_depth=14, min_samples_leaf=30,
        class_weight="balanced", random_state=SEED))
