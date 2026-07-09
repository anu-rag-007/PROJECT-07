import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class SleepStageModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()

    def train(self, csv_path):
        data = pd.read_csv(csv_path)

        X = data.drop("true_stage", axis=1)
        y = data["true_stage"]

        y_encoded = self.label_encoder.fit_transform(y)

        self.model.fit(X, y_encoded)

    def predict(self, features_df):
        predictions = self.model.predict(features_df)
        return self.label_encoder.inverse_transform(predictions)