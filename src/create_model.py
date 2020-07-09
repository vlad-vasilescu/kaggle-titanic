import tensorflow as tf
import pandas as pd
from util import process_data


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(6, )),
        tf.keras.layers.Dense(36, activation="relu"),
        tf.keras.layers.Dense(36, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

train_data = pd.read_csv("../data/train.csv")

survived = train_data.pop("Survived")

train_data = process_data(train_data)
train_data = tf.data.Dataset.from_tensor_slices((train_data.values, survived.values))
train_data = train_data.shuffle(len(survived)).batch(1)

model = create_model()
model.fit(train_data, epochs=50)
model.save("../models/1")