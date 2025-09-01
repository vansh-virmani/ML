import joblib
model = joblib.load("soil_model.pkl")
import pandas as pd

sample = pd.DataFrame([[213, 9.8, 338, 7.62]], columns=["N", "P", "K", "pH"])
print(model.predict(sample))

