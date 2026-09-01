# ============================================
# RETAIL INTELLIGENCE & DEMAND FORECASTING
# ADVANCED EDA SCRIPT
# ============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ============================================
# CONNECT TO MYSQL
# ============================================

password = quote_plus("Varun@123")

engine = create_engine(
    f"mysql+pymysql://root:{password}@localhost:3306/retail_db"
)

# ============================================
# LOAD DATA
# ============================================

df = pd.read_sql(
    "SELECT * FROM sales_data",
    engine
)

# ============================================
# BASIC DATASET OVERVIEW
# ============================================

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# DATE CONVERSION
# ============================================

df['month_year'] = pd.to_datetime(
    df['month_year'],
    format='%d-%m-%Y'
)

print("\nDate Conversion Successful")
print(df['month_year'].head())

# ============================================
# DESCRIPTIVE STATISTICS
# ============================================

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

print(df.describe().T)

# ============================================
# DEMAND DISTRIBUTION
# ============================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['qty'],
    kde=True,
    bins=25
)

plt.title("Demand Distribution")
plt.xlabel("Quantity Sold")
plt.ylabel("Frequency")
plt.show()

# ============================================
# DEMAND OUTLIERS
# ============================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df['qty']
)

plt.title("Demand Outliers")
plt.show()

# ============================================
# CORRELATION HEATMAP
# ============================================

numeric_df = df.select_dtypes(
    include=['int64','float64']
)

plt.figure(figsize=(16,10))

sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ============================================
# CORRELATION WITH TARGET VARIABLE
# ============================================

print("\n" + "="*50)
print("CORRELATION WITH DEMAND (QTY)")
print("="*50)

corr = numeric_df.corr()['qty']

print(
    corr.sort_values(
        ascending=False
    )
)

# ============================================
# CATEGORY-WISE DEMAND ANALYSIS
# ============================================

top_cat = df.groupby(
    'product_category_name'
)['qty'].sum()

print("\n" + "="*50)
print("CATEGORY-WISE DEMAND")
print("="*50)

print(
    top_cat.sort_values(
        ascending=False
    )
)

top_cat.sort_values(
    ascending=False
).plot(
    kind='bar',
    figsize=(12,5)
)

plt.title("Demand by Product Category")
plt.xlabel("Category")
plt.ylabel("Quantity Sold")
plt.show()

# ============================================
# REVENUE ANALYSIS
# ============================================

df['revenue'] = (
    df['qty']
    *
    df['unit_price']
)

cat_rev = df.groupby(
    'product_category_name'
)['revenue'].sum()

print("\n" + "="*50)
print("CATEGORY-WISE REVENUE")
print("="*50)

print(
    cat_rev.sort_values(
        ascending=False
    )
)

cat_rev.sort_values(
    ascending=False
).plot(
    kind='bar',
    figsize=(12,5),
    color='green'
)

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()

# ============================================
# DEMAND TREND OVER TIME
# ============================================

monthly = df.groupby(
    'month_year'
)['qty'].sum()

plt.figure(figsize=(12,5))

monthly.plot()

plt.title("Demand Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.grid(True)

plt.show()

print("\n" + "="*50)
print("MONTHLY DEMAND")
print("="*50)

print(monthly.tail(20))

# ============================================
# PRICE ELASTICITY ANALYSIS
# ============================================

plt.figure(figsize=(10,5))

sns.scatterplot(
    data=df,
    x='unit_price',
    y='qty'
)

plt.title("Price vs Demand")
plt.xlabel("Unit Price")
plt.ylabel("Quantity Sold")

plt.show()

print("\n" + "="*50)
print("PRICE-DEMAND CORRELATION")
print("="*50)

print(
    df[
        ['unit_price','qty']
    ].corr()
)

# ============================================
# ABSOLUTE CORRELATION ANALYSIS
# ============================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE PREVIEW")
print("="*50)

corr_abs = abs(
    df.select_dtypes(
        include='number'
    ).corr()['qty']
)

print(
    corr_abs.sort_values(
        ascending=False
    )
)

# ============================================
# SUMMARY STATISTICS OF TARGET VARIABLE
# ============================================

print("\n" + "="*50)
print("QTY STATISTICS")
print("="*50)

print(df['qty'].describe())

# ============================================
# EDA COMPLETED
# ============================================

print("\n" + "="*50)
print("ADVANCED EDA COMPLETED SUCCESSFULLY")
print("="*50)

# ============================================================
# FEATURE ENGINEERING + MODEL TRAINING SCRIPT
# ============================================================
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ============================================================
# 1. CREATE COPY OF DATASET
# ============================================================

df_ml = df.copy()

# ============================================================
# 2. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ============================================================
# product_id   -> Identifier, not useful for prediction
# month_year   -> Already represented by month and year
# total_price  -> Data leakage because total_price = qty * unit_price
# revenue      -> Data leakage because revenue = qty * unit_price

df_ml.drop(
    columns=[
        'product_id',
        'month_year',
        'total_price',
        'revenue'
    ],
    inplace=True,
    errors='ignore'
)

print("Columns after dropping leakage columns:")
print(df_ml.columns.tolist())

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLE
# ============================================================

df_ml = pd.get_dummies(
    df_ml,
    columns=['product_category_name'],
    drop_first=True
)

# Convert boolean dummy columns into integers
bool_cols = df_ml.select_dtypes(include='bool').columns
df_ml[bool_cols] = df_ml[bool_cols].astype(int)

print("\nData types after encoding:")
print(df_ml.dtypes)

# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X = df_ml.drop(
    'qty',
    axis=1
)

y = df_ml['qty']

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================================
# 6. SAVE ENGINEERED DATASET
# ============================================================

df_ml.to_csv(
    "retail_engineered.csv",
    index=False
)

print("\nEngineered dataset saved as retail_engineered.csv")

# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, pred_rf)
rf_rmse = mean_squared_error(y_test, pred_rf) ** 0.5
rf_r2 = r2_score(y_test, pred_rf)

print("\n" + "="*50)
print("RANDOM FOREST MODEL PERFORMANCE")
print("="*50)
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance.head(15))

top15 = importance.head(15)

plt.figure(figsize=(10,6))

plt.barh(
    top15['Feature'],
    top15['Importance']
)

plt.title("Top 15 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.gca().invert_yaxis()
plt.show()

# ============================================================
# 9. GRADIENT BOOSTING MODEL
# ============================================================

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

gbr.fit(X_train, y_train)

pred_gbr = gbr.predict(X_test)

gbr_mae = mean_absolute_error(y_test, pred_gbr)
gbr_rmse = mean_squared_error(y_test, pred_gbr) ** 0.5
gbr_r2 = r2_score(y_test, pred_gbr)
gbr_mape = mean_absolute_percentage_error(y_test, pred_gbr) * 100

print("\n" + "="*50)
print("GRADIENT BOOSTING MODEL PERFORMANCE")
print("="*50)
print("MAE :", gbr_mae)
print("RMSE:", gbr_rmse)
print("R²  :", gbr_r2)
print("MAPE:", gbr_mape)

# ============================================================
# 10. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'MAE': [rf_mae, gbr_mae],
    'RMSE': [rf_rmse, gbr_rmse],
    'R2 Score': [rf_r2, gbr_r2]
})

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(model_comparison)

# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    gbr,
    "demand_forecasting_model.pkl"
)

print("\nFinal Gradient Boosting model saved as demand_forecasting_model.pkl")

# ============================================================
# 12. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    pred_gbr,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand")

plt.show()

# ============================================================
# 13. ACTUAL VS PREDICTED LINE PLOT
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(
    y_test.values,
    label='Actual'
)

plt.plot(
    pred_gbr,
    label='Predicted'
)

plt.legend()
plt.title("Actual vs Predicted Demand")
plt.xlabel("Test Samples")
plt.ylabel("Demand")

plt.show()

# ============================================================
# SCRIPT COMPLETED
# ============================================================

print("\nFeature Engineering and Model Training Completed Successfully!")