# 🌲 Forest Fire Prediction System (Regression + Classification)

## 📌 Overview

This project focuses on predicting forest fire behavior using Machine Learning. It consists of **two major tasks**:

1. **Classification Task** → Predict whether a fire will occur (**Fire / Not Fire**)
2. **Regression Task** → Predict the **Fire Weather Index (FWI)**, which indicates fire intensity

The project demonstrates a complete **end-to-end Machine Learning pipeline**, including data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 🎯 Problem Statement

Forest fires are influenced by environmental and meteorological factors such as temperature, humidity, wind speed, and rainfall.

This project aims to:

* Classify whether conditions will lead to a fire
* Predict fire intensity (FWI) for better risk assessment

---

## 📊 Dataset

* **Dataset Used:** Algerian Forest Fires Dataset
* Contains:

  * Temperature
  * Relative Humidity (RH)
  * Wind Speed (Ws)
  * Rain
  * Fine Fuel Moisture Code (FFMC)
  * Duff Moisture Code (DMC)
  * Drought Code (DC)
  * Initial Spread Index (ISI)
  * Fire Weather Index (FWI)
  * Classes (Fire / Not Fire)

---

## ⚙️ Machine Learning Pipeline

### 🔹 1. Data Preprocessing

* Removed unnecessary columns
* Checked dataset structure and unique values
* Converted categorical labels into numerical format
* Performed feature scaling using **StandardScaler**

> Note: Dataset had minimal missing values, so no major imputation was required.

---

### 🔹 2. Exploratory Data Analysis (EDA)

* Basic data inspection performed:

  * Head of dataset
  * Unique values
* Feature relationships were analyzed using correlation

> (Further visualizations like heatmaps and boxplots can be added for deeper insights.)

---

### 🔹 3. Feature Engineering

* Feature selection based on correlation
* Scaling applied for models sensitive to feature magnitude
* Target variables separated:

  * Classification → `Classes`
  * Regression → `FWI`

---

## 🤖 Models Used

### 🔸 Classification Models

* Logistic Regression ✅ (Best Performing)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

### 🔸 Regression Models

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet

---

## 🏆 Model Selection

### ✅ Best Classification Model: Logistic Regression

**Reason:**

* Achieved highest accuracy
* Generalized well on unseen data
* Simple and interpretable
* Efficient for linearly separable data

---

### ✅ Best Regression Model: Regularized Linear Models (Ridge/Lasso)

**Reason:**

* Reduced overfitting
* Handled multicollinearity
* Provided stable predictions

---

## 🔧 Hyperparameter Tuning

* Used **GridSearchCV** for optimization
* Tuned parameters such as:

  * Regularization strength
  * Tree depth
  * Kernel parameters (SVM)

---

## 📈 Evaluation Metrics

### 🔹 Classification

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

**Why?**

> These metrics help evaluate both overall performance and class-wise prediction quality.

---

### 🔹 Regression

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

**Why?**

> These metrics measure prediction error and model fit.

---

## 🔄 Pipeline Workflow

```
Data Loading
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Scaling
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Evaluation
```

---

## ⚠️ Limitations

* Limited EDA visualizations
* No deployment (web app/API)
* No automated pipeline (e.g., sklearn Pipeline)
* Dataset size is relatively small

---

## 🚀 Future Improvements

* Add interactive dashboard (Streamlit/Flask)
* Deploy model as an API
* Perform advanced feature engineering
* Use ensemble stacking techniques
* Add real-time weather data integration

---

## 🧠 Key Learnings

* Importance of model comparison
* Simpler models can outperform complex ones
* Feature scaling significantly impacts performance
* Hyperparameter tuning improves generalization

---

## 📌 Conclusion

This project successfully demonstrates a complete machine learning workflow for forest fire prediction. By combining both classification and regression approaches, it provides a more comprehensive understanding of fire occurrence and intensity.

---

## 💻 How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the notebooks:

   ```
   jupyter notebook
   ```

---

## 🙌 Acknowledgment

* Dataset: UCI / Kaggle (Algerian Forest Fires Dataset)

---

## 📎 Author

**Vansh Virmani**

---
