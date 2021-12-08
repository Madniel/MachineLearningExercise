from sklearn import datasets
import functions as fn


if __name__ == '__main__':
    data = datasets.make_classification(n_samples=100, n_features=20)
    clf = fn.choose_classifier()

    # Prediction
    X = data[0]
    y = data[1]
    fn.prediction(clf, X, y)

    # Cross validation
    fn.cross_validation(clf, X, y)