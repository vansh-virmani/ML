import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("data/dataset1.csv")


df = df.dropna()


X = df[["N", "P", "K", "pH"]]       # You can expand this later
y = df["Output"]                   # Target column

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Quick evaluation
print("Accuracy:", model.score(X_test, y_test))

# 7. Save model for deployment
joblib.dump(model, "soil_model.pkl")