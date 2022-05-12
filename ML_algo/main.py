# import tensorflow
# import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import linear_regression_pickle_matplotlib
import KNN_1
import KNN_2
import SVM
from unSupervised import k_means_cluttering_visual_representation
from unSupervised import k_means_tim_example



def linear_regression():
    data = pd.read_csv("dataset/student-mat.csv", sep=";")

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"

    X = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    print("X \n" , X)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)

    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

print("__name__ ::" , __name__  )
if __name__ == '__main__':
    # linear_regression()
    # linear_regression_pickle_matplotlib.linear_regression_pickle_matplotlib()
    # KNN_1.knn()
    # KNN_2.knn_2()
    # SVM.svm1()
    # k_means_cluttering_visual_representation.fun1()
    k_means_tim_example.fun1()