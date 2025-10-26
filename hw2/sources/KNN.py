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
        x1 : array-like, shape (n_features,)
            First point.
        x2 : array-like, shape (n_features,)
            Second point.

        Returns:
        distance : float
            Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the given data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted labels.
        """
        import numpy as np

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
        