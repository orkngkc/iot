import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix


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


def evaluate_on_single_user(model, X_user, y_user, user_label="user", target_names=None):
    """
    Helper for per-user scores.
    Runs predict on that user's samples only,
    then prints precision / recall / F1 for that user.
    """

    _, y_pred_user = predict_model(model, X_user)

    print(f"===== {user_label} =====")
    return evaluate_classification_metrics(
        y_true=y_user,
        y_pred=y_pred_user,
        target_names=target_names,
        print_report=True
    )
