# Inventory Intelligence & Demand Forecasting Model

## Overview

The Retail Intelligence & Demand Forecasting System is an end-to-end Data Analytics and Product Management project designed to analyze retail sales data, identify business insights, and forecast product demand using Machine Learning.

The project integrates SQL, Data Analytics, Machine Learning, and Streamlit Deployment to provide an interactive business intelligence dashboard for retailers and analysts.

---

## Problem Statement

Retail businesses often struggle with:

* Inaccurate demand forecasting
* Overstocking and understocking
* Inefficient inventory planning
* Lack of actionable business insights

This project addresses these challenges by analyzing historical retail data and predicting future demand using machine learning models.

---

## Tech Stack

### Programming & Analytics

* Python
* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-Learn
* Random Forest Regressor
* Gradient Boosting Regressor

---

## Dataset Information

The dataset contains retail product sales information including:

* Product Category
* Quantity Sold
* Unit Price
* Freight Cost
* Customer Count
* Product Ratings
* Product Weight
* Competitor Pricing
* Seasonal Factors
* Holiday Indicators

* Features: 30+

---

## SQL Integration

The dataset was imported into MySQL for structured data storage and business analysis.

### Key SQL Operations

* Category-wise Sales Analysis
* Revenue Analysis
* Monthly Demand Analysis
* Data Validation

---

## Exploratory Data Analysis (EDA)

Performed extensive EDA to understand business patterns.

### Key Analyses

* Data Quality Assessment
* Missing Value Analysis
* Demand Distribution Analysis
* Outlier Detection
* Correlation Analysis
* Category-wise Demand Analysis
* Revenue Analysis
* Price vs Demand Analysis
* Time Series Demand Trends

### Key Insights

* Health & Beauty generated the highest revenue.
* Garden Tools recorded the highest demand.
* Customer count strongly influenced demand.
* Freight cost significantly impacted sales.
* Product presentation (images) positively affected demand.

---

## Feature Engineering

Performed preprocessing and feature engineering:

* Removed data leakage features
* One-Hot Encoding of product categories
* Feature selection using correlation and importance analysis
* Dataset preparation for machine learning models

---

## Machine Learning Models

### 1. Random Forest Regressor

Performance:

* MAE: 6.08
* RMSE: 9.97
* R² Score: 0.714

### 2. Gradient Boosting Regressor (Final Model)

Performance:

* MAE: 5.99
* RMSE: 9.93
* R² Score: 0.736

The Gradient Boosting model was selected as the final model due to its superior predictive performance.

---

## Feature Importance

Top demand drivers identified by the model:

1. Seasonal Factor (s)
2. Customer Count
3. Freight Price
4. Product Weight
5. Product Photos
6. Competitor Pricing
7. Unit Price
8. Product Specifications

These insights help businesses make informed inventory and pricing decisions.

---

## Streamlit Dashboard

An interactive Streamlit dashboard was developed to visualize insights and predictions.

### Dashboard Features

* Project Overview
* Business Analytics Dashboard
* Demand Trend Visualization
* Category Analysis
* Revenue Insights
* Demand Prediction System
* Inventory Alert System
* Interactive Charts and KPIs

---

## Project Structure

```text
Retail-Intelligence-Demand-Forecasting
│
├── app.py
├── eda.py
├── file.py
├── retail_engineered.csv
├── demand_forecasting_model.pkl
├── retail_price.csv
├── requirements.txt
├── README.md
│
├── screenshots/
```

---

## Future Improvements

* Time-Series Forecasting Models
* Real-Time Data Integration
* Automated Inventory Optimization
* Cloud Database Integration
* Advanced Business Intelligence Reports
