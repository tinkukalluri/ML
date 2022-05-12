#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle



def linear_regression_pickle_matplotlib():
    print('from linear_regressior_pickle_matplotlib')
    data = pd.read_csv("dataset/student-mat.csv", sep=";")

    predict = "G3"

    data = data[["G1", "G2", "absences", "failures", "studytime", "G3"]]
    data = shuffle(data)  # Optional - shuffle the data

    x = np.array(data.drop(columns=[predict]))
    y = np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
    best = 0
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print("Accuracy: " + str(acc))

        if acc > best:
            best = acc
            with open("studentgrades.pickle", "wb") as f:
                pickle.dump(linear, f)

    # LOAD MODEL
    pickle_in = open("studentgrades.pickle", "rb")
    linear = pickle.load(pickle_in)

    print("-------------------------")
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    print("-------------------------")

    predicted = linear.predict(x_test)
    for x in range(len(predicted)):
        print(predicted[x], x_test[x], y_test[x])

    # Drawing and plotting model
    # this is not the trained model graph it is just using regular data from csv file.
    style.use("ggplot")
    plot = "failures"
    plt.scatter(data[plot], data["G3"])
    plt.xlabel(plot)
    plt.ylabel("Final Grade")
    plt.show()

if __name__ == '__main__':
    linear_regression_pickle_matplotlib()