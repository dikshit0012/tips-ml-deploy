# 🍽️ Restaurant Tip Predictor

A machine learning regression model that predicts restaurant tip amounts based on dining features such as total bill, party size, time of day, and more — built with scikit-learn and deployed via a serialized pipeline.

---

## 📊 Dataset

Uses the classic **Tips dataset** from the Seaborn library, containing **244 records** of restaurant transactions.

| Feature | Type | Description |
|---|---|---|
| `total_bill` | float | Total bill amount (USD) |
| `tip` | float | Tip amount (USD) — **target variable** |
| `sex` | categorical | Gender of the bill payer |
| `smoker` | categorical | Whether the party included smokers |
| `day` | categorical | Day of the week |
| `time` | categorical | Meal time (Lunch / Dinner) |
| `size` | int | Number of people in the party |

---

## 🧠 Model Overview

A **Linear Regression** model wrapped inside a **scikit-learn Pipeline** that handles preprocessing automatically.

```
Pipeline
├── ColumnTransformer (Preprocessor)
│   ├── OneHotEncoder  →  sex, smoker, day, time
│   └── StandardScaler →  total_bill, size
└── LinearRegression
```

---

## 📈 Results

| Metric | Value |
|---|---|
| **R² Score** | `0.4373` |
| **RMSE** | `0.8387` |

> The model explains ~44% of variance in tip amounts. Total bill amount shows the strongest correlation with tip (r = 0.68).

---

## 🗂️ Project Structure

```
├── Deploy.ipynb        # Main Colab notebook
├── model.pkl           # Serialized pipeline (pickle)
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### Run the Notebook

1. Open `Deploy.ipynb` in [Google Colab](https://colab.research.google.com/) or JupyterLab
2. Run all cells sequentially
3. The trained pipeline will be saved as `model.pkl`

### Load & Use the Model

```python
import pickle
import pandas as pd

# Load the saved pipeline
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Example prediction
sample = pd.DataFrame([{
    "total_bill": 25.00,
    "size": 3,
    "sex": "Male",
    "smoker": "No",
    "day": "Sat",
    "time": "Dinner"
}])

predicted_tip = pipeline.predict(sample)
print(f"Predicted Tip: ${predicted_tip[0]:.2f}")
```

---

## 🔍 Exploratory Data Analysis

Key findings from the EDA phase:

- **Total bill** is the strongest predictor of tip amount (correlation: **0.68**)
- **Party size** has a moderate positive correlation with tip (correlation: **0.49**)
- No missing values found across all 244 records

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0)
![Colab](https://img.shields.io/badge/Google%20Colab-notebook-F9AB00?logo=googlecolab&logoColor=white)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
