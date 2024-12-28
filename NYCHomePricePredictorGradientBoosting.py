import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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



# Initialize the model
gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Train the model
gbm.fit(X_train, y_train)

# Predict on the test set
y_pred = gbm.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
