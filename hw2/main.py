from sources.KNN import KNN
import numpy as np
import os
from sources.CNN import (build_har_model, build_har_model_light, compile_model, run_user_specific_models_with_internal_split
                         ,train_model, predict_model,
                         evaluate_classification_metrics)
from sources.perceptrons import part3_gate_models


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

   
    num_classes = len(np.unique(y_train))  # genelde 6
    timesteps   = X_train.shape[1]         # D boyutu (örn 561)
    channels    = 1                        # tek kanal gibi davranıyoruz şimdilik

    # Conv1D modelimiz (128,9) bekliyordu, ama şu an elimizde (D,) vektör var.
    # Bunu (D,1) yapacağız:
    X_train_cnn = np.expand_dims(X_train, axis=-1)  # (N, D) -> (N, D, 1)
    X_test_cnn  = np.expand_dims(X_test,  axis=-1)
    y_train = y_train.astype(int) - 1
    y_test  = y_test.astype(int) - 1
    model = build_har_model_light(
    input_timesteps=timesteps,
    num_channels=channels,
    num_classes=num_classes,
    dropout_conv=0.3,   # 0.3 conv bloklarda
    dropout_fc=0.4,     # 0.4 dense blokta
    model_name="UCI_HAR_CNN_LIGHT"
    )

    model = compile_model(
        model,
        learning_rate=3e-4
    )

    model.summary()

    EPOCHS = 30
    BATCH = 16
    
    print("[INFO] training model")
    model, history = train_model(
        model,
        X_train_cnn,
        y_train,
        X_val=None,
        y_val=None,
        use_validation_split=True,       
        validation_split_ratio=0.1,       
        epochs=EPOCHS,
        batch_size=BATCH,
        patience=5,
        verbose=1
    )

    # 6. test set üzerinde tahmin al
    print("[INFO] predicting on test set...")
    proba_test, y_pred_test = predict_model(model, X_test_cnn)

    # 7. metrikleri yazdır (precision / recall / F1 dahil)
    print("[INFO] evaluating test performance...")
    metrics_all = evaluate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        target_names=[
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ],
        print_report=True
    )
    
    subject_train_path = os.path.join(DATASET_DIR, "train", "subject_train.txt")
    subject_test_path  = os.path.join(DATASET_DIR, "test",  "subject_test.txt")

    subject_train = read_data(subject_train_path, x=False)
    subject_test  = read_data(subject_test_path,  x=False)
    X_all_cnn = np.concatenate([X_train_cnn, X_test_cnn], axis=0)
    y_all     = np.concatenate([y_train,     y_test    ], axis=0)
    subject_all = np.concatenate([subject_train, subject_test], axis=0)

    user_specific_reports = run_user_specific_models_with_internal_split(
        X_all_cnn=X_all_cnn,
        y_all=y_all,
        subject_all=subject_all,
        num_classes=len(np.unique(y_all)),
        input_timesteps=X_all_cnn.shape[1],
        num_channels=X_all_cnn.shape[2],
        batch_size=16
    )

    print("\n====== USER SPECIFIC SUMMARY ======")
    for uid, rep in user_specific_reports.items():
        print(f"\n--- Subject {uid} ---")
        print(rep)

    
    part3_gate_models() #PART 3

if __name__ == "__main__":
    main()