# House Price Prediction Using Machine Learning

**Mini Project Report Implementation**  
Based on the internship report by **Varun Mishra**  
United College of Engineering & Research, Prayagraj  
Dr. A.P.J. Abdul Kalam Technical University, Lucknow (2024-25)

---

## Project Overview

This project builds an end-to-end supervised machine-learning pipeline that predicts residential house prices from property attributes. It follows the exact workflow described in the report:

1. **Data Collection** – California Housing dataset (publicly available)
2. **Preprocessing** – Missing-value imputation, one-hot encoding, feature scaling
3. **Feature Engineering** – rooms_per_household, bedrooms_per_room, population_per_household
4. **Model Training** – Linear Regression (baseline) + Random Forest Regressor
5. **Evaluation** – MAE, RMSE, R² score + visualizations
6. **Deployment** – Interactive Streamlit web application

---

## Project Structure

```
house_price_prediction/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md
├── data/
│   └── housing.csv        # California Housing dataset
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── metadata.pkl
│   └── evaluation_plots.png
└── src/
    └── train_models.py    # Training & evaluation script
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain the models

```bash
python src/train_models.py
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## Model Performance (Test Set)

| Model              | MAE ($)   | RMSE ($)  | R² Score |
|--------------------|-----------|-----------|----------|
| Linear Regression  | ~49,645   | ~69,127   | 0.635    |
| Random Forest      | ~31,980   | ~49,759   | 0.811    |

Random Forest captures non-linear relationships better and is recommended for predictions.

---

## Features Used

**Numeric:** longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income + engineered features

**Categorical:** ocean_proximity (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`)

**Target:** median_house_value

---

## Streamlit App Tabs

- **Prediction** – Enter property details → get instant price forecasts
- **Model Performance** – Metrics table + evaluation plots
- **Data Explorer** – Dataset sample, distributions, correlations
- **About** – Project summary matching the original report

---

## References (from the report)

1. Kaggle – House Prices datasets  
2. Pedregosa et al. – Scikit-learn: Machine Learning in Python  
3. Streamlit Documentation  
4. Breiman, L. (2001). Random Forests  
5. James et al. – An Introduction to Statistical Learning  
6. Harrison & Rubinfeld (1978) – Hedonic housing prices  

---

*This repository is a complete, runnable implementation of the mini-project described in the internship report.*
