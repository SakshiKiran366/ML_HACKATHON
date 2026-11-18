from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import SEED
from wrappers import SkModel

CONFIG = "StandardScaler + KNN(k=25, distance-weighted), train subsample cap 100k rows"

def build():
    return SkModel(make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=25, weights="distance", n_jobs=-1)), max_train_rows=100_000)
