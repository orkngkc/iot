import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
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
            most_common = max(k_indices, key=k_indices.count)
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

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for true, pred in zip(y, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 1 and pred == 0:
                fn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 0 and pred == 0:
                tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, recall, precision, f1_score
    
    def __repr__(self):
        """
        K-Nearest Neighbors (KNN) classifier.

        This object represents a KNN classifier configured with a fixed number of neighbors.
        Its repr appears as "KNN(k=<value>)", where <value> is the integer number of neighbors.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider when predicting labels.

        Attributes
        ----------
        k : int
            The number of neighbors configured for this instance.

        Methods (typical)
        -----------------
        fit(X, y)
            Store training samples and their labels for later neighbor queries.
        predict(X)
            Predict labels for provided samples by majority vote among the k nearest neighbors.
        score(X, y)
            Compute the accuracy of predictions on (X, y).

        Notes
        -----
        This is a simple instance-based classifier. Implementation details such as distance
        metric, tie-breaking behavior, weighting of neighbors, and acceleration structures
        (e.g., KD-tree) affect performance and should be documented in the implementation.
        """
        # write like this is a knn object k= this value and explain this class
        return (
            f"This is a KNN object, k={self.k}. "
            "Simple k-nearest neighbors classifier using Euclidean distance. "
            "Call fit(X, y) to store training data, predict(X) to get labels, and evaluate(X, y) for accuracy."
        )
        