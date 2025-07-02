# 🌍 Environmental Prediction Models using Machine Learning

This project explores predictive modeling across three key environmental datasets—**Air Quality**, **Weather Data**, and **CO₂ Emissions**—using various machine learning regression techniques. It forms part of the MSc in Data Analytics at the National College of Ireland.

---

## 👩‍💻 Author

**Jayashree Rajkumar**  


---

## 📘 Project Title

**Prediction of Air Quality, Weather Data, and CO₂ in Air using Machine Learning Models**

---

## 🎯 Objective

To evaluate the performance of multiple machine learning regression models—Random Forest, Gradient Boosting, Support Vector Regression (SVR), Ridge Regression, and Neural Networks—on environmental datasets and determine the most accurate model based on MSE, RMSE, and R² metrics.

---

## 📊 Datasets Used

1. **Air Quality Dataset** – 16,219 rows, 11 columns  
2. **Weather Dataset** – 86,504 rows, 12 columns  
3. **CO₂ Emissions Dataset** – 25,205 rows, 35 columns

---

## 🧠 Models Implemented

- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression (SVR)
- Ridge Regression
- Neural Network Regression (for Weather data)

---

## 🧪 Methodology

The **KDD (Knowledge Discovery in Databases)** process was followed:

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Correlation Analysis
5. Model Training using train/test split
6. Evaluation using:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score

---

## 📈 Results Summary

| Dataset     | Best Model          | R² Score | Notes |
|-------------|---------------------|----------|-------|
| Air Quality | Random Forest       | ~0.99    | High accuracy and low RMSE |
| Weather     | Random Forest       | ~0.98    | Best overall performer |
| CO₂ Emissions | Random Forest    | ~0.95    | Most reliable prediction |

> Gradient Boosting also performed competitively across all datasets, while Ridge and SVR performed less consistently.

---

## 🔮 Future Work

- Advanced feature engineering to improve model performance
- Hyperparameter tuning for all models
- Application of ensemble techniques (e.g., stacking)
- Interpretability using SHAP/LIME
- Expansion with additional environmental datasets

---

## 🛠️ Technologies & Libraries

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook

---



