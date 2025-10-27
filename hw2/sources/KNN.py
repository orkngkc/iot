import numpy as np
from .metrics import Metrics
from sklearn.metrics import confusion_matrix, classification_report

class KNN:
    def __init__(self, k=3, number_of_classes=None):
        self.k = k
        self.number_of_classes = number_of_classes
        self.X_train = None
        self.y_train = None

    def fit(self, X : np.ndarray, y: np.ndarray):
        """Store the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        """
        self.X_train = X
        self.y_train = y

    def _eucledian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the Euclidean distance between two points.

        Parameters:
        x1 : np.ndarray, shape (n_features,)
            First point.
        x2 : np.ndarray, shape (n_features,)
            Second point.

        Returns:
        distance : float
            Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the given data.

        Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict.

        Returns:
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels.
        """

        y_pred = []
        for x in X:
            # Compute distances for all training points
            distances = [self._eucledian_distance(x, x_train) for x_train in self.X_train]
            # Get the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get the most common class label among the neighbors
            most_common = np.bincount(self.y_train[k_indices]).argmax()
            y_pred.append(most_common)

        
        return np.array(y_pred)
    
    def evaluate(self, X : np.ndarray, y : np.ndarray) -> float:
        """Evaluate the accuracy of the model.

        Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Test data.
        y : np.ndarray, shape (n_samples,)
            True labels.

        Returns:
        accuracy : float
            Accuracy of the model on the test data.
        recall : float
            Recall of the model on the test data.
        precision : float
            Precision of the model on the test data.
        f1_score : float
            F1 score of the model on the test data.
        """
        y_pred = self._predict(X)

        confusion_mat = Metrics.confusion_matrix(y, y_pred, self.number_of_classes)
        scikit_conf_mat = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred)
        print("Confusion Matrix:\n", scikit_conf_mat)
        print("Classification Report:\n", class_report)

        print("--------------- Custom Metrics Calculation ---------------")
        if self.number_of_classes is None:
            self.number_of_classes = confusion_mat.shape[0]

        metric_results, accuracy = Metrics.calculate_metrics(confusion_mat, self.number_of_classes)

        return metric_results, accuracy, confusion_mat
        