import numpy as np
class Metrics:
    @staticmethod
    def confusion_matrix(y_true : np.ndarray, y_pred : np.ndarray, num_classes : int = None) -> np.ndarray:
        if num_classes is None:
            num_classes = max(np.max(y_true), np.max(y_pred))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[true - 1][pred - 1] += 1
        return cm
    
    @staticmethod
    def calculate_metrics(cm : np.ndarray, num_classes : int) -> tuple:
        accuracy = np.trace(cm) / np.sum(cm)

        metric_results = {}
        precision = int()
        recall = int()
        f1_score = int()
        tp = int()
        fp = int()
        fn = int()

        
        for i in range(num_classes):
            tp = cm[i][i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metric_results[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }

        return metric_results, accuracy
