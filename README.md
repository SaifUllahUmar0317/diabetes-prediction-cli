# 🩺 Diabetes Prediction CLI App (Machine Learning Project)

This project is a **command-line machine learning application** that predicts whether a person is diabetic based on medical inputs. It uses multiple classification models and selects the best-performing one (Random Forest) for final deployment.

---

## 📊 Dataset

The model uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target variable: 1 for diabetic, 0 for non-diabetic)

---

## 🚀 Features

- Data Cleaning (missing values, duplicates)
- Feature Selection using:
  - Correlation
  - Mutual Information
- Data Scaling using `MinMaxScaler`
- Data Balancing using `SMOTE`
- Models trained:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
- Performance evaluation using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (visualized)
- Best model (Random Forest) saved using `joblib`
- CLI-based interface for user input and prediction

---

## 🛠️ Installation

1. Clone the repository or download the project files.
2. Make sure you have Python 3.7+ installed.
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
