import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
)
import time
from sklearn.metrics import classification_report

# Configuración reproducible
tf.random.set_seed(42)
np.random.seed(42)


# ========== 1. Cargar y balancear datos ==========
df = pd.read_csv("./Datos/2/Base.csv")

# Balanceo por submuestreo
majority = df[df["fraud_bool"] == 0]
minority = df[df["fraud_bool"] == 1]
majority_downsampled = resample(
    majority, replace=False, n_samples=len(minority), random_state=42
)
balanced_df = pd.concat([majority_downsampled, minority]).sample(
    frac=1, random_state=42
)
# Imprimir cantidad de datos de cada categoría:
print(
    f"Datos balanceados: {len(balanced_df[balanced_df['fraud_bool'] == 0])} no fraude, {len(balanced_df[balanced_df['fraud_bool'] == 1])} fraude"
)
# Separar features y target
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


# ========== 2. Hiperparámetros ==========
epochs = 30
batch_size = 256
learning_rate = 0.15
l2_norm_clip = 1.0
noise_multiplier = 1.1
delta = 1e-5
num_classes = 2
steps_per_epoch = len(X_train) // batch_size


# ========== 3. Modelo ==========
def create_model(input_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )


model = create_model(X_train.shape[1])
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)


# ========== 4. Entrenamiento manual con DP ==========
def train_step(x_batch, y_batch):
    with tf.GradientTape(persistent=True) as tape:
        logits = model(x_batch, training=True)
        per_example_losses = loss_fn(y_batch, logits)
        mean_loss = tf.reduce_mean(per_example_losses)

    # Gradientes por dato
    grads_list = tape.jacobian(
        per_example_losses,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    # Clipping
    clipped_grads = []
    for g in grads_list:
        norms = tf.norm(tf.reshape(g, [g.shape[0], -1]), axis=1)
        scaling_factors = tf.minimum(1.0, l2_norm_clip / (norms + 1e-6))
        clipped = tf.stack([scaling_factors[i] * g[i] for i in range(batch_size)])
        clipped_mean = tf.reduce_mean(clipped, axis=0)
        clipped_grads.append(clipped_mean)

    # Añadir ruido gaussiano
    noised_grads = [
        g
        + tf.random.normal(tf.shape(g), stddev=l2_norm_clip * noise_multiplier)
        / batch_size
        for g in clipped_grads
    ]

    # Aplicar gradientes
    optimizer.apply_gradients(zip(noised_grads, model.trainable_variables))

    return mean_loss


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
# ========== 5. Entrenar ==========
for epoch in range(1, epochs + 1):
    start_time = time.time()
    # Barajar los datos
    idx = np.random.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train[idx], y_train[idx]

    # Mini-batch loop
    for step in range(steps_per_epoch):
        start = step * batch_size
        end = start + batch_size
        x_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        loss = train_step(x_batch, y_batch)

    # Evaluar
    logits_test = model(X_test)
    preds_test = tf.argmax(logits_test, axis=1)
    acc = np.mean(preds_test.numpy() == y_test)
    print(
        f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f} - Time: {time.time() - start_time:.2f}s"
    )

    # Calcular epsilon
    epsilon, _ = compute_dp_sgd_privacy(
        n=len(X_train),
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epoch,
        delta=delta,
    )
    print(f"→ ε after {epoch} epochs: {epsilon:.2f} (δ={delta})")


# ========== 6. Reporte ==========
print("\nClassification report:")
print(classification_report(y_test, preds_test.numpy(), digits=4))
