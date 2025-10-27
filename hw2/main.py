from sources.KNN import KNN
import numpy as np

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

    # read training data
    X_train = read_data('UCI HAR Dataset/train/X_train.txt', x=True)
    y_train = read_data('UCI HAR Dataset/train/y_train.txt', x=False)

    # read test data
    X_test = read_data('UCI HAR Dataset/test/X_test.txt', x=True)
    y_test = read_data('UCI HAR Dataset/test/y_test.txt', x=False)

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