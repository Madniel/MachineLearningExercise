from sklearn import datasets
from sklearn.preprocessing import OrdinalEncoder
import functions as fn

if __name__ == '__main__':
    tic_tac = datasets.fetch_openml('tic-tac-toe')
    clf = fn.choose_classifier()

    # Information about data
    fn.show_information(tic_tac)
    fn.show_example(tic_tac, 0)

    # Transforming data
    # Using encoder
    # Values encoded as o:0 b:1 x:1
    columns = tic_tac.data.shape[1]
    categories = [['o', 'b', 'x']] * columns
    ord_encoder = OrdinalEncoder(categories=categories)
    X = tic_tac.data
    X_encoded = ord_encoder.fit_transform(X)

    # Transfering to o:-1 b:0 x:1
    # Treat 'o' as penalty
    for i in range(len(X_encoded)):
        X_encoded[i] -= 1

    # Targets
    y = tic_tac.target.array.codes

    # Prediction
    fn.prediction(clf, X_encoded, y)

    # Cross validation
    fn.cross_validation(clf, X_encoded, y)
