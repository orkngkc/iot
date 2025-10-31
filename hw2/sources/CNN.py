import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def build_har_model(
    input_timesteps=128,
    num_channels=9,
    num_classes=6,
    dropout_conv=0.3,
    dropout_fc=0.5,
    model_name="UCI_HAR_CNN"
):
    """
    Build the 1D Conv HAR model.
    Input shape: (timesteps, channels) = (128, 9) by default.
    Output: softmax over num_classes (default 6 classes).
    """

    inputs = layers.Input(shape=(input_timesteps, num_channels))

    # --- Conv Block 1 ---
    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # --- Conv Block 2 ---
    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # --- Conv Block 3 (bigger receptive field) ---
    x = layers.Conv1D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # Temporal aggregation
    x = layers.GlobalAveragePooling1D()(x)

    # Dense head
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_fc)(x)

    # Classifier
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=model_name)
    return model
def build_har_model_light(
    input_timesteps=128,
    num_channels=9,
    num_classes=6,
    dropout_conv=0.3,
    dropout_fc=0.4,
    model_name="UCI_HAR_CNN_LIGHT"
):
    """
    Lighter/stabler CNN for HAR.
    Uses fewer filters and a smaller dense layer to reduce capacity
    (~1e5 params instead of ~3.7e5 in the bigger model).
    """

    inputs = layers.Input(shape=(input_timesteps, num_channels))

    # --- Conv Block 1 ---
    x = layers.Conv1D(
        filters=32,          # was 64
        kernel_size=5,
        strides=1,
        padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # --- Conv Block 2 ---
    x = layers.Conv1D(
        filters=64,          # was 128
        kernel_size=5,
        strides=1,
        padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # --- Conv Block 3 ---
    x = layers.Conv1D(
        filters=128,         # was 256
        kernel_size=9,
        strides=1,
        padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_conv)(x)

    # Global temporal pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Smaller dense head
    x = layers.Dense(64)(x)  # was 128
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_fc)(x)

    # Classifier
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=model_name)
    return model

def compile_model(model, learning_rate=1e-3):
    """
    Compile with Adam + SparseCategoricalCrossentropy.
    Tracks accuracy during training.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_size=64,
    epochs=30,
    patience=5,
    use_validation_split=False,
    validation_split_ratio=0.1,
    verbose=1,
):
    """
    Train the model.

    Two modes:
    1) If use_validation_split=True:
        we ignore X_val/y_val and let Keras carve out validation_split_ratio from X_train.
    2) Else:
        we require X_val and y_val explicitly.

    Returns:
        trained model,
        history (the History object from model.fit)
    """

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ]

    if use_validation_split:
        history = model.fit(
            X_train,
            y_train,
            validation_split=validation_split_ratio,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        if X_val is None or y_val is None:
            raise ValueError(
                "use_validation_split=False but X_val/y_val not provided."
            )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

    return model, history


def predict_model(model, X):
    """
    Get predictions (softmax probs and argmax class ids).
    Returns:
        proba: [N, num_classes]
        pred:  [N]
    """
    proba = model.predict(X)
    pred = np.argmax(proba, axis=1)
    return proba, pred


def evaluate_classification_metrics(y_true, y_pred, target_names=None, print_report=True):
    """
    Compute confusion matrix + classification report.
    classification_report includes precision / recall / F1 per class,
    plus macro avg and weighted avg.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    if print_report:
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report_str)

    # You can also return them if you want to log/save
    return {
        "confusion_matrix": cm,
        "report_text": report_str,
    }


def run_user_specific_models_with_internal_split(
    X_all_cnn,
    y_all,
    subject_all,
    num_classes,
    input_timesteps,
    num_channels,
    batch_size=16
):
    unique_users = np.unique(subject_all)
    user_results = {}

    for user_id in unique_users:
        # o kullanıcıya ait TÜM örnekleri çek
        idx = np.where(subject_all == user_id)[0]

        # çok az sample varsa model eğitmek saçma olur, skip
        if len(idx) < 20:
            continue

        X_user = X_all_cnn[idx]
        y_user = y_all[idx]

        # kendi içinden 80/20 split yapıyoruz
        X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(
            X_user,
            y_user,
            test_size=0.2,
            random_state=42,
            stratify=y_user  # her sınıf dengeli kalsın diye
        )

        # model yarat (light versiyon – stabil olan)
        user_model = build_har_model_light(
            input_timesteps=input_timesteps,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_conv=0.3,
            dropout_fc=0.4,
            model_name=f"UCI_HAR_CNN_LIGHT_USER_{user_id}"
        )

        user_model = compile_model(
            user_model,
            learning_rate=3e-4  # batch=16 ile iyi çalışan lr
        )

        # kişiye özel fit – kendi train setinin %10’unu val olarak ayırıyoruz
        user_model, _ = train_model(
            user_model,
            X_user_train,
            y_user_train,
            X_val=None,
            y_val=None,
            use_validation_split=True,
            validation_split_ratio=0.1,
            epochs=30,
            batch_size=batch_size,
            patience=5,
            verbose=0   # sessize aldık çünkü loop'ta çok user var
        )

        # kişiye özel test performansı
        _, y_user_pred = predict_model(user_model, X_user_test)

        print(f"\n=== Subject {int(user_id)} ===")
        metrics = evaluate_classification_metrics(
            y_true=y_user_test,
            y_pred=y_user_pred,
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

        # rapor için sakla (mesela rapora tablo halinde koymak istersen)
        user_results[int(user_id)] = metrics["report_text"]

    return user_results
