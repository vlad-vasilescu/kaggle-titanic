import pandas as pd


def process_data(data):
    data = data.drop(columns=["PassengerId", "Name", "Ticket", "Fare", "Embarked"])
    data["Sex"] = data["Sex"].apply(lambda x: 0 if x == "male" else 1)
    data["Cabin"] = pd.Categorical(data["Cabin"])
    data["Cabin"] = data.Cabin.cat.codes

    data["Age"] = pd.Categorical(data["Age"])
    data["Age"] = data.Age.cat.codes

    data = data.replace([None, -1], [0.0, 0.0])

    data = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Cabin"]].astype(float)

    return data