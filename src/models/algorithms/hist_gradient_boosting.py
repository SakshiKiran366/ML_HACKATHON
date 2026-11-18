from wrappers import HGBModel

CONFIG = "HistGradientBoosting(lr=0.1, 31 leaves, min_leaf=40, l2=1.0, rounds chosen on held-out trips, <=400)"

def build(): return HGBModel()
