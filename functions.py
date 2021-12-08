from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def choose_classifier():
    print('''Classifiers:
          1 - LinearRegression
          2 - Decision Tree Classifier
          3 - Random Forest
          4 - SVM''')

    number = int(input('Choose by number: '))

    if number == 1:
        return LinearRegression()
    elif number == 2:
        return DecisionTreeClassifier()
    elif number == 3:
        return RandomForestClassifier()
    elif number == 4:
        return svm.SVC()


def show_information(data):
    # Print information about data

    print(data)
    print(data.DESCR)
    print(data.target)


def show_example(data, number=0, image=False):
    # Show data and target of specified index
    # If data is image show it

    print(data.target[number])
    if image:
        show_image(data.images[number])


def classifier_fit(clf, data, number=10):
    # Train model on specified number of digits

    clf.fit((data.data[0:number]),
            data.target[0:number])
    return clf


def simple_predict(clf, data, number=-1):
    # Use model to simple prediction of one digit
    # Then show target of that digit

    result = clf.predict([data.data[number]])
    real_value = data.target[number]

    print(f'Result of prediction: {result[0]}')
    print(f'Real value: {real_value}')


def prediction(clf, X, y, test_size=0.25, shuffle=True):
    # Use train_test_split to split data
    # Train model and predict
    # Use metrics to calculate mean squared error

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)

    clf.fit(X_train, y_train)
    result = clf.predict(X_test)

    print(f'Result: {result}')
    print(f'Real Value: {y_test}')

    print(f'RMSE :{mean_squared_error(y_test, result)}')


def show_image(image, title, cmap='gray'):
    # Show image from data

    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()


def show_all_images(images, targets):
    # Show all images from data

    for i, image in enumerate(images):
        show_image(image, targets)


def cross_validation(clf, X, y, scoring="neg_mean_squared_error", cv=10):
    scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
    neg_scores = np.sqrt(-scores)
    print(f'Scores: {neg_scores}')
    print(f'Mean: {neg_scores.mean()}')
    print(f'Std: {neg_scores.std()}')
