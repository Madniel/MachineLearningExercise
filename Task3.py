from sklearn import datasets
import functions as fn

if __name__ == '__main__':
    iris = datasets.load_iris()
    clf = fn.choose_classifier()

    # Information about data
    fn.show_information(iris)
    fn.show_example(iris, 0)

    # Prediction
    fn.prediction(clf, iris, iris.target, test_size=0.2, shuffle=True)

