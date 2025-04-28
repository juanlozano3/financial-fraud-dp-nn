import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
)
from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import time

# Cargar y balancear datos
data = pd.read_csv("./Datos/2/Base.csv")
X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

# Balanceo por submuestreo
majority = data[data["fraud_bool"] == 0]
minority = data[data["fraud_bool"] == 1]
majority_downsampled = resample(
    majority, replace=False, n_samples=len(minority), random_state=42
)
balanced_df = pd.concat([majority_downsampled, minority]).sample(
    frac=1, random_state=42
)
X = balanced_df.drop(columns=["fraud_bool"])
y = balanced_df["fraud_bool"].astype(np.int32)

# Codificación
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = (
    encoder.fit_transform(X[categorical_cols])
    if len(categorical_cols) > 0
    else np.empty((len(X), 0))
)
X_num = X[numeric_cols].values
X = np.hstack([X_num, X_cat])

# Escalado
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

# Hiperparámetros
batch_size = 256
epochs = 30
learning_rate = 0.15
l2_norm_clip = 1.0
noise_multiplier = 1.1
delta = 1e-5

# Crear el modelo Keras
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Optimizer DP
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=batch_size,
    learning_rate=learning_rate,
)

# Compilar modelo
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    ),
    metrics=["accuracy"],
)

# Entrenamiento
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
)

# Evaluación final
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final test accuracy: {accuracy:.4f}")

# Cálculo de privacidad
epsilon, _ = compute_dp_sgd_privacy(
    n=len(X_train),
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=delta,
)
print(f"Privacidad diferencial lograda: ε = {epsilon:.2f}, δ = {delta}")

# Reporte
preds = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification report:")
print(classification_report(y_test, preds))
