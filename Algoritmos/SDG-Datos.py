import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import resample
import tensorflow as tf
import tensorflow_privacy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
)
from sklearn.metrics import classification_report

data = pd.read_csv("./Datos/2/Base.csv")

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

# Dividir por clases
X_majority = X[y == 0]
X_minority = X[y == 1]

y_majority = y[y == 0]
y_minority = y[y == 1]

# Submuestreo de la clase mayoritaria para igualar la minoritaria
X_majority_downsampled, y_majority_downsampled = resample(
    X_majority,
    y_majority,
    replace=False,
    n_samples=len(y_minority),
    random_state=42,
)

# Combinar datos balanceados
X_balanced = pd.concat([X_majority_downsampled, X_minority])
y_balanced = pd.concat([y_majority_downsampled, y_minority])

# Mezclar aleatoriamente
# Mezclar aleatoriamente
X, y = sklearn.utils.shuffle(X_balanced, y_balanced, random_state=42)


# Convert the categorical columns to dummies
X = pd.get_dummies(X, drop_first=True)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# convert the labels
y = tf.keras.utils.to_categorical(y, num_classes=2)

# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
# TODO solve the imbalance problem

# Define hyperparameters
epochs = 10
batch_size = 256
learning_rate = 0.25

# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"),
    ]
)

# Define the loss function and the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Train the model
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# ==== NUEVO BLOQUE: Classification report ====

# Obtener predicciones
y_pred = model.predict(X_test)

# Convertir de one-hot a etiquetas
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Imprimir classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, digits=4))


"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
