
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
   ```
   Or install manually:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib
   ```

---

## 📁 Files Description

| File Name              | Description                                      |
|------------------------|--------------------------------------------------|
| `main.ipynb`  | Main notebook for training and evaluating models |
| `user interface.ipynb` | CLI-based interface for predicting new input     |
| `Diabetes predictor model.pkl` | Saved trained Random Forest model     |
| `Scaler.pkl`           | Saved MinMaxScaler for input normalization       |

---

## 🧪 How to Use (Command Line)

1. Run the prediction script (from terminal or Jupyter cell):
   ```python
   import joblib
   import numpy as np

   scaler = joblib.load("Scaler.pkl")
   model = joblib.load("Diabetes predictor model.pkl")

   features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

   print("\n--- Enter Patient Details ---")
   data = []
   for f in features:
       val = float(input(f"{f}: "))
       data.append(val)

   input_scaled = scaler.transform([data])
   prediction = model.predict(input_scaled)

   print("\n🔍 Prediction Result:")
   if prediction[0] == 1:
       print("⚠️ The person is likely to be Diabetic.")
   else:
       print("✅ The person is NOT Diabetic.")
   ```

---

## 📈 Example Output

```
--- Enter Patient Details ---
Pregnancies: 2
Glucose: 120
BloodPressure: 70
SkinThickness: 20
Insulin: 79
BMI: 25.3
DiabetesPedigreeFunction: 0.5
Age: 29

🔍 Prediction Result:
✅ The person is NOT Diabetic.
```

---

## 🏆 Evaluation Summary

Random Forest performed best with high accuracy, precision, recall, and F1-score. Confusion matrices for each model were visualized using Seaborn heatmaps.

---

## 🤖 Author

**Saif Ullah Umar**  
Aspiring Machine Learning Engineer  
Currently following an 8-week ML roadmap

---

## 📌 Future Enhancements

- Convert CLI to GUI using Tkinter or PyQt
- Add model explainability (e.g., SHAP)
- Deploy as a web app using Flask or Streamlit

---
