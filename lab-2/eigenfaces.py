"""Eigenfaces face-recognition example and shared LFW dataset loader."""

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_dataset():
    """Load the face images and labels used by the lab examples."""
    return fetch_lfw_people(min_faces_per_person=70, resize=0.4)


def main():
    lfw_people = load_dataset()

    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % X.shape[1])
    print("n_classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    n_components = 150
    mean = np.mean(X_train, axis=0)
    X_train = X_train - mean
    X_test = X_test - mean
    _, S, V = np.linalg.svd(X_train, full_matrices=False)
    components = V[:n_components]
    eigenfaces = components.reshape((n_components, h, w))
    X_transformed = np.dot(X_train, components.T)
    X_test_transformed = np.dot(X_test, components.T)

    import matplotlib.pyplot as plt

    def plot_gallery(images, titles, height, width, n_row=3, n_col=4):
        """Plot a gallery of portraits."""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(
            bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35
        )
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((height, width)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.show()

    explained_variance = (S**2) / (n_samples - 1)
    ratio_cumsum = np.cumsum(explained_variance / explained_variance.sum())
    plt.plot(np.arange(n_components), ratio_cumsum[:n_components])
    plt.title("Compactness")
    plt.show()

    estimator = RandomForestClassifier(
        n_estimators=150, max_depth=15, max_features=150
    )
    estimator.fit(X_transformed, y_train)
    predictions = estimator.predict(X_test_transformed)
    print("Accuracy:", np.mean(predictions == y_test))
    print(classification_report(y_test, predictions, target_names=target_names))


if __name__ == "__main__":
    main()
