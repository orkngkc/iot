from sources.KNN import KNN
import numpy as np
import os

def read_data(file_path: str, x = True) -> np.ndarray:
    # Placeholder for data reading logic
    file = open(file_path, 'r')
    data = file.readlines()

    results = []
    if x:
        for i in range(len(data)):
            data[i] = data[i].strip().split()
            result = []
            for j in range(len(data[i])):
                result.append(float(data[i][j]))
            result = np.array(result)
            results.append(result)

        return np.array(results)

    for i in range(len(data)):
        data[i] = data[i].strip()
        results.append(int(data[i]))
    return np.array(results)

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # dataset folder = "UCI HAR Dataset" 
    DATASET_DIR = os.path.join(BASE_DIR, "UCI HAR Dataset")

    # individual file paths
    X_train_path = os.path.join(DATASET_DIR, "train", "X_train.txt")
    y_train_path = os.path.join(DATASET_DIR, "train", "y_train.txt")
    X_test_path  = os.path.join(DATASET_DIR, "test",  "X_test.txt")
    y_test_path  = os.path.join(DATASET_DIR, "test",  "y_test.txt")

    
    print("X_train_path =", X_train_path)
    print("exists?      =", os.path.exists(X_train_path))

    # read training data
    X_train = read_data(X_train_path, x=True)
    y_train = read_data(y_train_path, x=False)

    # read test data
    X_test = read_data(X_test_path, x=True)
    y_test = read_data(y_test_path, x=False)

    # create KNN model
    model = KNN(k=5)

    # fit the model
    model.fit(X_train, y_train)

    # evaluate the model
    metric_results, accuracy, cm = model.evaluate(X_test, y_test)
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy}")
    print("Detailed Metrics per Class:")
    for class_label, metrics in metric_results.items():
        print(f"Class {class_label}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")



if __name__ == "__main__":
    main()