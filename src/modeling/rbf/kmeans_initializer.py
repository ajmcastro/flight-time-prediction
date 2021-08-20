import sys

from keras.initializers import Initializer
import tensorflow as tf
from sklearn.cluster import KMeans


class InitCentersKMeans(Initializer):
    """Initializer for RBFNet of centers
    using clustering.
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(init='k-means++', n_clusters=n_centers,
                    max_iter=self.max_iter, n_jobs=-1, verbose=2)
        km.fit(self.X)
        return km.cluster_centers_
