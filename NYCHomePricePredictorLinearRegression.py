import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Reading CSVs with borough sales data as Pandas dataframes
manhattan_df = pd.read_csv('/content/rollingsales_manhattan_final(Manhattan).csv', header=0)
bronx_df = pd.read_csv('/content/rollingsales_bronx_final(Bronx).csv', header=0)
brooklyn_df = pd.read_csv('/content/rollingsales_brooklyn_final(Brooklyn).csv', header=0)
queens_df = pd.read_csv('/content/rollingsales_queens_final(Queens).csv', header=0)
statenisland_df = pd.read_csv('/content/rollingsales_statenisland_final(Staten Island).csv', header=0)

# Combine all borough data
nyc_df = pd.concat([manhattan_df, bronx_df, brooklyn_df, queens_df, statenisland_df], ignore_index=True)

# Drop irrelevant features
nyc_df = nyc_df.drop(['EASEMENT', 'ADDRESS', 'APARTMENT NUMBER', 'BLOCK', 'LOT', 'SALE DATE'], axis=1)

# Filter target variable outliers (restrict to reasonable price range)
nyc_df[' SALE PRICE '] = nyc_df[' SALE PRICE '].str.replace(',', '').astype(float)
nyc_df = nyc_df[nyc_df[' SALE PRICE '].between(10000, 1e7)]

# Drop rows with missing critical values
nyc_df = nyc_df.dropna(subset=['ZIP CODE', 'RESIDENTIAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT'])

# Convert columns to numeric
nyc_df.loc[:, 'GROSS SQUARE FEET'] = (
    nyc_df['GROSS SQUARE FEET']
    .str.replace(',', '')
    .astype(float)
)
nyc_df.loc[:, 'LAND SQUARE FEET'] = (
    nyc_df['LAND SQUARE FEET']
    .str.replace(',', '')
    .astype(float)
)

# One-hot encode categorical variables
categorical_columns = [
    'BOROUGH', 'BUILDING CLASS AT PRESENT',
    'BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY'
]

for column in nyc_df.columns:
    num_distinct_values = nyc_df[column].nunique()
    print(f"Column '{column}' has {num_distinct_values} distinct values.")

nyc_df = pd.get_dummies(nyc_df, columns=categorical_columns, drop_first=True)

# Create derived features
nyc_df['BUILDING_AGE'] = 2024 - nyc_df['YEAR BUILT']

# Define which columns will be target-encoded
encode_cols = ['NEIGHBORHOOD', 'ZIP CODE', 'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE']

def target_encode(data, col_to_encode, target_col=' SALE PRICE '):
    # Compute the global mean of the target
    overall_mean = data[target_col].mean()
    # Compute the mean sale price for each category in col_to_encode
    means = data.groupby(col_to_encode)[target_col].mean()
    # Map those means back onto the dataframe
    data[col_to_encode + '_encoded'] = data[col_to_encode].map(means)
    # Replace any missing with overall mean (assign back without inplace= to avoid warnings)
    data[col_to_encode + '_encoded'] = data[col_to_encode + '_encoded'].fillna(overall_mean)
    return data

# Perform target encoding on the entire dataframe
for col in encode_cols:
    nyc_df = target_encode(nyc_df, col)

# Drop original categorical columns (now that we've encoded them)
nyc_df.drop(columns=encode_cols, inplace=True)

# Define features (X) and target (y)
X = nyc_df.drop([' SALE PRICE ', 'PRICE_PER_SQFT'], axis=1, errors='ignore')
y = nyc_df[' SALE PRICE ']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # e.g. 20% test size
    random_state=42     # for reproducibility
)

# Apply log-transformation to the target variable
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression model on log-transformed target
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_log)

# Make predictions and revert log-transformation
y_train_pred_log = lr_model.predict(X_train_scaled)
y_test_pred_log = lr_model.predict(X_test_scaled)

y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.2f}, Train R²: {train_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, Test R²: {test_r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(feature_importance)

# Visualize predictions
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Prices")
plt.show()

# Residual distribution
residuals = y_test - y_test_pred
plt.hist(residuals, bins=30, edgecolor='k')
plt.title("Residual Distribution")
plt.show()

# 5-Fold Cross-Validation MSE (on the training set)
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train_log, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {np.mean(-cv_scores):.2f}")

target_corr = nyc_df.corr()[' SALE PRICE '].sort_values(ascending=False)
print(target_corr)

from sklearn.metrics import explained_variance_score
train_ev = explained_variance_score(y_train, y_train_pred)
test_ev = explained_variance_score(y_test, y_test_pred)
print(f"Train Explained Variance: {train_ev:.2f}, Test Explained Variance: {test_ev:.2f}")

train_r2_adj = 1 - ((1 - train_r2) * (len(y_train) - 1)) / (len(y_train) - X_train_scaled.shape[1] - 1)
test_r2_adj = 1 - ((1 - test_r2) * (len(y_test) - 1)) / (len(y_test) - X_test_scaled.shape[1] - 1)
print(f"Train Adjusted R²: {train_r2_adj:.2f}, Test Adjusted R²: {test_r2_adj:.2f}")


train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
print(f"Train MAPE: {train_mape:.2f}%, Test MAPE: {test_mape:.2f}%")
