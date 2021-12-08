from sklearn import datasets
from sklearn.model_selection import train_test_split
import functions as fn

if __name__ == '__main__':
    faces = datasets.fetch_olivetti_faces()
    clf = fn.choose_classifier()

    # Information about data
    fn.show_information(faces)
    fn.show_example(faces, 0)
    fn.show_all_images(faces.images, faces.target)

    # Prediction
    # Use classifier with shuffle
    X = faces.data
    y = faces.target
    images = faces.images
    fn.prediction(clf, X, y,
                  test_size=0.2, shuffle=True)

    # Photos are mixed
    X_train, X_test, y_train, y_test = train_test_split(
        faces.images, y, test_size=0.2, shuffle=True)
    fn.show_all_images(X_train, y_train)
    fn.show_all_images(X_test, y_test)

    # Use classifier without shuffle
    fn.prediction(clf, X, y,
                  test_size=0.2, shuffle=False)

    # Photos are in turn
    X_train, X_test, y_train, y_test = train_test_split(
        faces.images, y, test_size=0.2, shuffle=False)
    fn.show_all_images(X_train, y_train)
    fn.show_all_images(X_test, y_test)


