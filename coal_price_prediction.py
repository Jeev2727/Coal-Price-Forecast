import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load all CSV and Excel files
external_files = {
    "Crude Oil": "/Users/mandalajeevan/Downloads/gas_data/Crude Oil WTI Futures Historical Data.csv",
    "Brent Oil": "/Users/mandalajeevan/Downloads/gas_data/Brent Oil Futures Historical Data.csv",
    "Dubai Crude": "/Users/mandalajeevan/Downloads/gas_data/Dubai Crude Oil (Platts) Financial Futures Historical Data.csv",
    "Dutch TTF": "/Users/mandalajeevan/Downloads/gas_data/Dutch TTF Natural Gas Futures Historical Data.csv",
    "Natural Gas": "/Users/mandalajeevan/Downloads/gas_data/Natural Gas Futures Historical Data.csv"
}

coal_file = "/Users/mandalajeevan/Downloads/gas_data/merged_file.xlsx"

# Load coal price data
coal_df = pd.read_excel(coal_file, sheet_name=None)
coal_data = pd.concat(coal_df.values(), ignore_index=True)
coal_data['Date'] = pd.to_datetime(coal_data['Date'])
coal_data.sort_values(by='Date', inplace=True)

# Load and merge external factor data
external_data = []
for key, file in external_files.items():
    df = pd.read_csv(file, usecols=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Price': key + '_Price'}, inplace=True)
    df.sort_values(by='Date', inplace=True)
    external_data.append(df)

merged_df = coal_data
for df in external_data:
    merged_df = pd.merge(merged_df, df, on='Date', how='left')

# Fill missing values
merged_df.fillna(method='ffill', inplace=True)
merged_df.interpolate(method='linear', inplace=True)

# Correlation matrix
correlation_matrix = merged_df.drop(columns=['Date']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of External Factors and Coal Prices")
plt.show()

# Define features and targets
feature_columns = ["Crude Oil_Price", "Brent Oil_Price", "Dubai Crude_Price", "Dutch TTF_Price", "Natural Gas_Price"]
target_columns = [col for col in merged_df.columns if col not in ['Date'] + feature_columns]
X = merged_df[feature_columns]
y = merged_df[target_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

# Multi-output training
model = MultiOutputRegressor(search.best_estimator_)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
results = []
for idx, col in enumerate(target_columns):
    results.append({
        "Coal Type": col,
        "MAE": mean_absolute_error(y_test[col], y_pred[:, idx]),
        "RMSE": np.sqrt(mean_squared_error(y_test[col], y_pred[:, idx])),
        "R2": r2_score(y_test[col], y_pred[:, idx])
    })
results_df = pd.DataFrame(results)
print(results_df)

# Compute average R2 and save model + score
avg_r2_score = results_df["R2"].mean()
print(f"\nAverage R² Score: {avg_r2_score:.4f}")

# Save both model and accuracy
joblib.dump({"model": model, "r2_score": avg_r2_score}, "coal_price_prediction.pkl")

# Forecast future values (next 30 days)
future_dates = pd.date_range(start=merged_df['Date'].max() + pd.Timedelta(days=1), periods=30)
latest_external_factors = X.iloc[-1:].copy()
future_external_factors = []

for factor in feature_columns:
    model_fit = ExponentialSmoothing(X[factor], trend='add', seasonal=None).fit()
    forecast_values = model_fit.forecast(30)
    future_external_factors.append(forecast_values.values)

future_external_factors = pd.DataFrame(np.array(future_external_factors).T, columns=feature_columns)
future_predictions = model.predict(future_external_factors)
future_df = pd.DataFrame(future_predictions, columns=target_columns)
future_df['Date'] = future_dates

# Save predictions
future_df.to_csv("Coal_Price_Predictions.csv", index=False)

# Plot actual vs predicted
for col in target_columns:
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['Date'], merged_df[col], label='Actual', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title(f"Actual Coal Prices: {col}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(future_df['Date'], future_df[col], label='Predicted (Next 30 Days)', color='red', linestyle='dashed', marker='o', markersize=6, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title(f"Predicted Coal Prices: {col}")
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nPredicted Coal Prices for the Next 30 Days:")
print(future_df)
