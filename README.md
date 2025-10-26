# 📡 Telecom Customer Churn Prediction using XGBoost

A complete end-to-end **Machine Learning project** that predicts whether a telecom customer is likely to **churn (leave the service)** based on their usage patterns and plan details.  
The project includes **data preprocessing, feature engineering, model optimization, and Streamlit app deployment**.

---

## 🚀 Live Demo
🔗 **Streamlit App:** [Click to Open App](https://YOUR-STREAMLIT-APP-URL)

---

## 📊 Project Overview
Churn prediction is one of the most critical business problems in the telecom industry.  
By identifying customers likely to churn, telecom companies can **take proactive retention measures** such as personalized offers or customer engagement.

This project uses **XGBoost**, an advanced ensemble algorithm, combined with **SMOTETomek resampling** to handle data imbalance and optimize recall for the minority class (churners).

---

## 🧠 Features

- **Data Cleaning & Feature Engineering**
  - Created derived metrics like `total_charge`, `mins_per_call`, `charge_per_min`.
  - Added behavior-based features such as `service_call_bin` and `pay_as_you_go_intl`.
- **Imputation & Encoding**
  - Handled missing values and categorical encoding using `SimpleImputer` and `OneHotEncoder`.
- **Scaling**
  - Standardized all numerical features for optimal model performance.
- **Resampling**
  - Used **SMOTETomek** to balance the churn vs non-churn classes.
- **Model Training**
  - Tuned **XGBoost** hyperparameters using `RandomizedSearchCV`.
- **Model Performance**
  - ✅ Accuracy: ~90%  
  - ✅ Recall (Churn class): ~82%  
  - ✅ Precision (Churn class): ~62%
- **Deployment**
  - Built and deployed an interactive Streamlit app.

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.13 |
| **Framework** | Streamlit |
| **ML / Data Science** | XGBoost, scikit-learn, imblearn, pandas, numpy |
| **Visualization** | Streamlit Components |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ Setup Instructions (Local)

1. **Clone this repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/telecom-churn-prediction.git
   cd telecom-churn-prediction
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run XGB_app.py
   ```

4. The app will open in your browser at `http://localhost:8501`

---

## 📁 Repository Structure

```
├── XGB_app.py                 # Streamlit app
├── P585 Churn.xlsx            # Dataset
├── num_imputer.pkl            # Numerical imputer
├── cat_imputer.pkl            # Categorical imputer
├── encoder.pkl                # OneHotEncoder
├── scaler.pkl                 # StandardScaler
├── xgb_model.pkl              # Trained XGBoost model
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
```

---

## 💡 Business Impact

By deploying this churn prediction system:

* Customer retention teams can **target high-risk customers** with offers before they leave.
* Marketing can **optimize retention budgets** using data-driven insights.
* Reduces churn rate, directly impacting **revenue growth and brand loyalty**.

---

## 🧑‍💻 Author

**👨‍💻 Chintu**
M.Sc. Computational Data Science — Acharya Nagarjuna University
📧 *Add your email or LinkedIn link here if you’d like (optional)*

---

## 🏁 Acknowledgments

* Dataset inspired by telecom churn case studies.
* Libraries: scikit-learn, imblearn, xgboost, streamlit.
* Project guided by real-world business objectives in customer retention.

---

⭐ **If you found this project helpful, consider giving it a star on GitHub!**

```
