# 🌍 Environmental Prediction Models using Machine Learning

This project explores predictive modeling across three key environmental datasets—**Air Quality**, **Weather Data**, and **CO₂ Emissions**—using various machine learning regression techniques. It forms part of the MSc in Data Analytics at the National College of Ireland.

---

## 👩‍💻 Author

**Jayashree Rajkumar**  
MSc in Data Analytics  
National College of Ireland, Dublin  
📧 x23199491@student.ncirl.ie

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
- Neural Network Regression (Weather data only)

---

## 📊 Visualizations

Visualizations were critical for EDA (Exploratory Data Analysis), model interpretation, and evaluation.

### 📌 Common Techniques Used
- **Correlation Heatmaps** – Show inter-variable relationships
- **Histograms** – Used for understanding CO₂ distributions
- **Boxplots** – Identified outliers in feature columns
- **Line Charts** – Represented feature trends over time
- **Evaluation Graphs** – RMSE, MSE, and R² plotted to compare model performance

### 📂 Highlights by Dataset

#### 🔴 Air Quality
- Correlation matrix revealed strong relationships with particulate matter
![correlation matrix](https://github.com/user-attachments/assets/226a8745-73fb-4cf6-93bf-c4deccc042e6)
- Feature distribution graphs showed skewness and need for scaling
- ![Distirbution of Temperature](https://github.com/user-attachments/assets/70c04277-cd6e-4080-a11d-c8aa4c9a21f7)
- ![Temperature vs Apparent temperature](https://github.com/user-attachments/assets/a4d6cc02-6ed2-411f-b73a-0a71f844b625)

#### 🌦 Weather
- Heatmaps and scatter plots helped reveal humidity–temperature patterns
- ![Correlation heatmap](https://github.com/user-attachments/assets/09c1cf2d-20cd-4f39-ae4e-6e5a54df367f)
- scatter plots captured temperature distribution across time
- ![Sactter plot for data value vs Start_date](https://github.com/user-attachments/assets/7032a7fa-bf2e-4f06-bac0-59fb3fd7f239)

#### 🟢 CO₂ Emissions
- Histograms tracked CO₂ emissions over decades
  ![Histogram of df$CO2](https://github.com/user-attachments/assets/99e6cd95-cc01-44f4-b977-5fd3a315e7d5)
- Heatmaps assisted in identifying highly correlated emissions features
  ![correlation matrix](https://github.com/user-attachments/assets/f7aa3d73-32f2-4d74-894e-3e2eb39a762e)
- Scatter Plot
  ![scatter plot df$co2](https://github.com/user-attachments/assets/b86991da-17ee-4127-9682-372fdad02610)


---

## 🧪 Methodology

The **KDD (Knowledge Discovery in Databases)** process was used:

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

| Dataset       | Best Model      | R² Score | Notes                          |
|---------------|------------------|----------|--------------------------------|
| Air Quality   | Random Forest    | ~0.99    | Best performance, high accuracy |
| Weather       | Random Forest    | ~0.98    | Superior performance overall   |
| CO₂ Emissions | Random Forest    | ~0.95    | Robust across time series data |

> Gradient Boosting also performed competitively across all datasets.

---

## 🔮 Future Work

- Advanced feature engineering
- Hyperparameter optimization
- Ensemble model stacking
- Explainability using SHAP or LIME
- Adding real-time data streaming support

---

## 🛠️ Technologies & Libraries

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---
