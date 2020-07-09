import tensorflow as tf
import pandas as pd
import numpy as np
from util import process_data


test_data = pd.read_csv("../data/test.csv")
passenger_ids = test_data["PassengerId"]
test_data = process_data(test_data)

model = tf.keras.models.load_model("../models/1")
predictions = model.predict(test_data.values)

outcome = []
for pid, prediction in zip(passenger_ids, predictions):
    outcome.append([pid, int(round(prediction[0]))])

outcome = pd.DataFrame(outcome, columns=["PassengerId", "Survived"])
outcome.to_csv("../data/prediction.csv", index=False)
