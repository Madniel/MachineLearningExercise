from sklearn import datasets
import pickle
import functions as fn

if __name__ == '__main__':
    digits = datasets.load_digits()
    clf = fn.choose_classifier()

    # Information about data
    fn.show_information(digits)
    fn.show_example(digits, 0)

    # Prediction
    # Testing classifier on fragment of data and pickle
    clf = fn.classifier_fit(clf, digits)
    pickle.dump(clf, open('clf.p', 'wb'))
    pickle.load(open('clf.p', 'rb'))
    fn.simple_predict(clf, digits)

    # Use train_test_split
    X = digits.data
    y = digits.target
    fn.prediction(clf, X, y)
    fn.cross_validation(clf, X, y)



