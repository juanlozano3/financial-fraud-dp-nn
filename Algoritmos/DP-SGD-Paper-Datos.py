import sklearn
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def define_model(features):
    """
    Define the model architecture.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(features["x"].shape[1])),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def model(features, labels, mode, params):
    """
    Model Function for tf.Estimator.
    Define el modelo, la función de pérdida, el optimizador y las salidas según el modo.
    """
    # Construir el modelo
    model = define_model(features)

    # Forward pass: logits = salida sin activar
    logits = model(features["x"])

    # Definir predicciones (para modo PREDICT)
    predictions = {
        "class_ids": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits),
        "logits": logits,
    }

    # --- MODO PREDICT ---
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)
        # Expandir labels para que tengan forma (batch_size, 1)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))

    # --- FUNCIÓN DE PÉRDIDA ---
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    vector_loss = loss_fn(y_true=labels, y_pred=logits)

    scalar_loss = tf.reduce_mean(vector_loss)

    # --- MÉTRICAS ---
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions["class_ids"]
    )

    # --- OPTIMIZADOR DP ---
    optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=params["l2_norm_clip"],
        noise_multiplier=params["noise_multiplier"],
        num_microbatches=params["num_microbatches"],
        learning_rate=params["learning_rate"],
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss=vector_loss, global_step=global_step)

    # --- MODO TRAIN ---
    if mode == tf_estimator.ModeKeys.TRAIN:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op
        )

    # --- MODO EVAL ---
    elif mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, eval_metric_ops={"accuracy": accuracy}
        )


def make_input_fn(X, y, batch_size, shuffle=True, repeat=True):
    """
    Create input function for the Estimator.
    """

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"x": X}, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def main():
    """
    Main function to run DP-SGD training with tf.Estimator.
    """
    # Load and prepare dataset
    data = pd.read_csv("./Datos/2/Base.csv")
    # Split features and labels
    X = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"]

    # Dividir por clases
    X_majority = X[y == 0]
    X_minority = X[y == 1]

    y_majority = y[y == 0]
    y_minority = y[y == 1]

    # Downsample majority class
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority,
        y_majority,
        replace=False,
        n_samples=len(y_minority),
        random_state=42,
    )
    
    #orden para balancearrrrrrr

    # Concatenate majority and minority class
    X_balanced = pd.concat([X_majority_downsampled, X_minority])
    y_balanced = pd.concat([y_majority_downsampled, y_minority])

    X, y = sklearn.utils.shuffle(X_balanced, y_balanced, random_state=42)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Codificamos las categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical_cols])
    # transformar las numericas
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    # Combinamos con las numéricas
    X_num = X[numeric_cols].values
    X = np.hstack([X_num, X_cat])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    # Training parameters
    batch_size = 256
    total_epochs = 10
    steps_per_epoch = X_train.shape[0] // batch_size

    # DP-SGD parameters
    params = {
        "l2_norm_clip": 1.2,
        "noise_multiplier": 1.1,
        "num_microbatches": 32,
        "learning_rate": 0.15,
    }

    # Estimator
    fraud_classifier = tf_estimator.Estimator(model_fn=model, params=params)

    # Training loop
    for epoch in range(1, total_epochs + 1):
        start_time = time.time()

        # Train
        fraud_classifier.train(
            input_fn=make_input_fn(X_train, y_train, batch_size), steps=steps_per_epoch
        )

        end_time = time.time()
        print(f"Epoch {epoch}/{total_epochs} - Time: {end_time - start_time:.2f}s")

        # Evaluate
        eval_results = fraud_classifier.evaluate(
            input_fn=make_input_fn(
                X_test, y_test, batch_size, shuffle=False, repeat=False
            ),
            steps=X_test.shape[0] // batch_size,
        )
        print(f"Evaluation: {eval_results}")

        # Classification report
        # Classification report
        predictions = list(
            fraud_classifier.predict(
                input_fn=make_input_fn(
                    X_test, y_test, batch_size, shuffle=False, repeat=False
                )
            )
        )

        y_pred = [1 if p["logits"][0] > 0.3 else 0 for p in predictions]

        print("\nClassification Report:")
        print(classification_report(y_test[: len(y_pred)], y_pred, digits=4))
        # Matriz
        cm = confusion_matrix(y_test[: len(y_pred)], y_pred)

        print("\nConfusion Matrix:")
        print(cm)
        #  privacy
        if params["noise_multiplier"] > 0:
            epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
                n=X_train.shape[0],
                batch_size=batch_size,
                noise_multiplier=params["noise_multiplier"],
                epochs=epoch,
                delta=1e-5,
            )

            print(f"DP-SGD Privacy after {epoch} epochs: ε = {epsilon:.2f}, δ = 1e-5")


if __name__ == "__main__":
    main()
