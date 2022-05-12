import numpy as np
import sklearn
import sys
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
np.set_printoptions(threshold=sys.maxsize)
def fun1():

    digits = load_digits()
    print(digits)
    data = scale(digits.data)
    print("rgb: " ,data[0][0] )
    print("rgb: ", data[1].shape)
    y = digits.target
    print(y)
    print("target size: " , len(y))
    k = 10
    # samples give u the rows and feature gives u the column in our dataset i.e load_digit dataset.
    samples, features = data.shape
    print("shape:: " , samples , features)
    print(digits.feature_names)
    def bench_k_means(estimator, name, data):
        estimator.fit(data)
        print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, estimator.inertia_,
                 metrics.homogeneity_score(y, estimator.labels_),
                 metrics.completeness_score(y, estimator.labels_),
                 metrics.v_measure_score(y, estimator.labels_),
                 metrics.adjusted_rand_score(y, estimator.labels_),
                 metrics.adjusted_mutual_info_score(y, estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean')))

    clf = KMeans(n_clusters=k, init="random", n_init=10)
    bench_k_means(clf, "1", data)