import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from matplotlib import cm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import plotly.graph_objs as go
from ipywidgets import interact, FloatSlider

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"
sns.set(font_scale=0.6)
# Read in the processed data
data = pd.read_csv(input_directory + "Wabiskaw.csv", header=0)
nan_values = data[data.isna().any(axis=1)]
print(nan_values)

input_directory = "../Data/ProcessedData/"

# List all files in the input directory
all_files = os.listdir(input_directory)

# Filter the list to include only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]

# Initialize an empty list to store individual DataFrames
dataframes = []

# Loop through each CSV file and read it into a DataFrame, then append it to the list
for file in csv_files:
    file_path = os.path.join(input_directory, file)
    df = pd.read_csv(file_path, header=0)
    dataframes.append(df)

# Concatenate all the DataFrames into a single DataFrame
data = pd.concat(dataframes, ignore_index=True)
data = pd.read_csv(input_directory + 'Wabiskaw.csv', header=0)

# Calculate correlations
corr_matrix = data.corr()

# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
########################################

ncg_threshold = 0  # Example threshold for distinguishing between NCG and non-NCG sites
ncg_data = data[data['MonInjGas(m3)'] > ncg_threshold]
non_ncg_data = data[data['MonInjGas(m3)'] == ncg_threshold]

features = ['OilRate', 'cSOR', 'CumulativeMonth', 'MonInjGas(m3)', 'CumInjGas(E3m3)', 'OCT(%)', 'NbrofWells']

target = 'SteamRate'

# NCG sites
X_ncg_train, X_ncg_test, y_ncg_train, y_ncg_test = train_test_split(ncg_data[features], ncg_data[target], test_size=0.2, random_state=42)
ncg_model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
ncg_model.fit(X_ncg_train, y_ncg_train)

# Non-NCG sites

X_non_ncg_train, X_non_ncg_test, y_non_ncg_train, y_non_ncg_test = train_test_split(non_ncg_data[features], non_ncg_data[target], test_size=0.2, random_state=42)
best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}

optimal_rf_model = RandomForestRegressor(**best_params, random_state=42)
optimal_rf_model.fit(X_non_ncg_train, y_non_ncg_train)

train_score = optimal_rf_model.score(X_non_ncg_train, y_non_ncg_train)
test_score = optimal_rf_model.score(X_non_ncg_test, y_non_ncg_test)

print("Train score: ", train_score)
print("Test score: ", test_score)
non_ncg_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
non_ncg_model.fit(X_non_ncg_train, y_non_ncg_train)
cv_scores = cross_val_score(non_ncg_model, X_non_ncg_test, y_non_ncg_test, cv=5)
print("Cross-validation scores: ", cv_scores)

file_path = os.path.join(input_directory, 'combined_data.csv')
non_ncg_data = pd.read_csv(file_path, header=0)

X_non_ncg = non_ncg_data[features]
non_ncg_data_ncg_model_pred = non_ncg_model.predict(X_non_ncg)

non_ncg_comparison_df = pd.DataFrame({
    'ProdDate': non_ncg_data['ProdDate'],
    'Original_Steam_Inj': non_ncg_data['MonInjSteam(m3)'],
    'Predicted_Steam_Inj': non_ncg_data_ncg_model_pred
})

plt.figure(figsize=(12, 6))
plt.plot(non_ncg_comparison_df['ProdDate'], non_ncg_comparison_df['Original_Steam_Inj'], label='Original Steam Injection', marker='o', linestyle='-')
plt.plot(non_ncg_comparison_df['ProdDate'], non_ncg_comparison_df['Predicted_Steam_Inj'], label='Predicted Steam Injection', marker='o', linestyle='-')

plt.xlabel('ProdDate')
plt.ylabel('Steam Injection')
plt.title('Comparison of Original and Predicted Steam Injection for Non-NCG Sites')
plt.legend()
plt.grid()

plt.show()
# Predictions for NCG sites
y_ncg_pred = ncg_model.predict(X_ncg_test)

# Predictions for Non-NCG sites
y_non_ncg_pred = non_ncg_model.predict(X_non_ncg_test)

# Scatter plot for NCG sites
plt.figure(figsize=(12, 6))
plt.scatter(y_ncg_test, y_ncg_pred, alpha=0.7, label='NCG Sites')
plt.xlabel('True Steam Injection')
plt.ylabel('Predicted Steam Injection')
plt.title('Scatter Plot of True vs. Predicted Steam Injection for NCG Sites')
plt.legend()
plt.grid()
plt.show()

# Scatter plot for Non-NCG sites
plt.figure(figsize=(12, 6))
plt.scatter(y_non_ncg_test, y_non_ncg_pred, alpha=0.7, label='Non-NCG Sites')
plt.xlabel('True Steam Injection')
plt.ylabel('Predicted Steam Injection')
plt.title('Scatter Plot of True vs. Predicted Steam Injection for Non-NCG Sites')
plt.legend()
plt.grid()
plt.show()
##########################################

# Define input features and target
# feature_list = ['MonInjSteam(m3)', 'MonthlyOil(m3)', 'MonInjGas(E3m3)', 'PrdHours(hr)', 'InjHours(hr)', 'CumInjSteam(m3)', 'CumInjGas(E3m3)', 'CumPrdOil(m3)']
data['CumInjGas(E3m3)'] = data['CumInjGas(E3m3)'] * 1000
data['MonInjGas(E3m3)'] = data['MonInjGas(E3m3)'] * 1000
print('features:' + str(features))
target = 'MonInjSteam(m3)'
print('target:' + str(target))

plt.figure(figsize=(12, 6))
plt.plot(data['CumPrdOil(m3)'], data['MonInjSteam(m3)'], label='Steam Injection', marker='o', linestyle='-')
plt.plot(data['CumPrdOil(m3)'], data['MonInjGas(m3)'], label='NCG Injection', marker='o', linestyle='-')

plt.xlabel('Cumulative Produced Oil')
plt.ylabel('Injection Volume')
plt.title('Steam and NCG Injection Volumes')
plt.legend()
plt.grid()

plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Create and train the combined_model
combined_model = RandomForestRegressor(n_estimators=100, random_state=42)
combined_model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = combined_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Create a function to modify the NCG values
def inflate_ncg_values(data, inflation_factor):
    data_copy = data.copy()
    data_copy['MonInjGas(m3)'] *= inflation_factor
    data_copy['CumInjGas(E3m3)'] = data_copy['MonInjGas(m3)'].cumsum()
    return data_copy


# Inflate NCG values by a factor (e.g., 2 for doubling the NCG values)
inflation_factor = 0
inflated_data = inflate_ncg_values(data, inflation_factor)

# Scale both original and inflated data
X_orig = data[features]
X_inflated = inflated_data[features]

# Predict the required steam injection for the original data and inflated data
steam_inj_orig = combined_model.predict(X_orig)
steam_inj_inflated = combined_model.predict(X_inflated)

# Inverse transform the predicted values
steam_inj_orig = steam_inj_orig.reshape(-1, 1).ravel()
steam_inj_inflated = steam_inj_inflated.reshape(-1, 1).ravel()
data = pd.concat(dataframes, ignore_index=True)
data = pd.read_csv(input_directory + 'combined_data.csv', header=0)
# Create a new dataframe with the original and inflated NCG values and corresponding steam injection predictions
comparison_df = pd.DataFrame({
    'CumPrdOil(m3)': data['CumPrdOil(m3)'],
    'Original_NCG': data['MonInjGas(m3)'],
    'Inflated_NCG': inflated_data['MonInjGas(m3)'],
    'Steam_Inj_Original': steam_inj_orig,
    'Steam_Inj_Inflated': steam_inj_inflated
})

# Plot the predicted steam injection for both the original data and the new dataset with inflated NCG values
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['CumPrdOil(m3)'], comparison_df['Steam_Inj_Original'], label='Original NCG Injection', marker='o',
         linestyle='-')
plt.plot(comparison_df['CumPrdOil(m3)'], comparison_df['Steam_Inj_Inflated'], label='Inflated NCG Injection', marker='o',
         linestyle='-')

plt.xlabel('CumPrdOil(m3)')
plt.ylabel('Predicted Steam Injection')
plt.title('Comparison of Predicted Steam Injection for Original and Inflated NCG Injection')
plt.legend()
plt.grid()

plt.show()

# Filter the rows where original NCG injection is not 0
non_zero_ncg = comparison_df[comparison_df['Original_NCG'] != 0]

# Calculate the differences between original and inflated steam injections
non_zero_ncg['Steam_Inj_Difference'] = non_zero_ncg['Steam_Inj_Original'] - non_zero_ncg['Steam_Inj_Inflated']

# Create a plot to visualize the difference
plt.figure(figsize=(12, 6))
plt.plot(non_zero_ncg['CumPrdOil(m3)'], non_zero_ncg['Steam_Inj_Difference'], marker='o', linestyle='-')
plt.xlabel('CumPrdOil(m3)')
plt.ylabel('Difference in Steam Injection')
plt.title('Difference in Steam Injection between Original and Inflated NCG (During Times with NCG)')
plt.grid()
plt.show()

# Inverse transform the y_test_scaled and y_pred arrays
actual_injected_steam = y_test
predicted_injected_steam = y_pred

# Create a scatter plot of actual vs. predicted injected steam
plt.scatter(actual_injected_steam, predicted_injected_steam)
plt.xlabel('Actual Injected Steam')
plt.ylabel('Predicted Injected Steam')
plt.title('Actual vs. Predicted Injected Steam')

# Set the axis limits to have the same scale
min_value = min(min(actual_injected_steam), min(predicted_injected_steam))
max_value = max(max(actual_injected_steam), max(predicted_injected_steam))
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.grid()
plt.show()

# Calculate the total amount of steam to be replaced with NCG
replacement_percentage = 0.1  # Replace 10% of steam with NCG
total_steam_to_replace = data['MonInjSteam(m3)'].sum() * replacement_percentage

# Replace the specified percentage of steam with NCG
data_replaced = data.copy()
data_replaced['MonInjGas(m3)'] += total_steam_to_replace / len(data_replaced)

# Scale the modified dataset
X_replaced = data_replaced[features]

# Predict the new steam injection requirements using the trained combined_model
steam_inj_replaced_scaled = combined_model.predict(X_replaced)

# Inverse transform the predicted values
steam_inj_replaced = steam_inj_replaced_scaled

# Create a new dataframe with the original and replaced NCG values and corresponding steam injection predictions
comparison_replaced_df = pd.DataFrame({
    'CumPrdOil(m3)': data['CumPrdOil(m3)'],
    'Original_NCG': data['MonInjGas(m3)'],
    'Replaced_NCG': data_replaced['MonInjGas(m3)'],
    'Steam_Inj_Original': steam_inj_orig,
    'Steam_Inj_Replaced': steam_inj_replaced
})

# Plot the predicted steam injection for both the original data and the new dataset with replaced NCG values
plt.figure(figsize=(12, 6))
plt.plot(comparison_replaced_df['CumPrdOil(m3)'], comparison_replaced_df['Steam_Inj_Original'], label='Original NCG Injection',
         marker='o', linestyle='-')
plt.plot(comparison_replaced_df['CumPrdOil(m3)'], comparison_replaced_df['Steam_Inj_Replaced'], label='Replaced NCG Injection',
         marker='o', linestyle='-')

plt.xlabel('CumPrdOil(m3)')
plt.ylabel('Predicted Steam Injection')
plt.title('Comparison of Predicted Steam Injection for Original and Replaced NCG Injection')
plt.legend()
plt.grid()

plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have a DataFrame called 'data' with your independent variables
# 'features' is a list of column names for the independent variables, and 'target' is the target column name
X = pd.concat([data[target], data[features]], axis=1)

# Add a constant term to the independent variables
X['Intercept'] = 1

# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


def evaluate_total_steam_injection(data, features, model, inflation_factor):
    inflated_data = inflate_ncg_values(data, inflation_factor)
    X_inflated = inflated_data[features]
    steam_inj_inflated = model.predict(X_inflated)
    return steam_inj_inflated.sum()


ncg_inflation_factors = np.linspace(0, 2, 100)  # Test NCG inflation factors from 0 to 2 with 100 steps
steam_injection_results = []

for factor in ncg_inflation_factors:
    total_steam_injection = evaluate_total_steam_injection(data, features, combined_model, factor)
    steam_injection_results.append(total_steam_injection)

min_steam_injection = min(steam_injection_results)
best_ncg_factor = ncg_inflation_factors[steam_injection_results.index(min_steam_injection)]

print(f"Minimum Steam Injection: {min_steam_injection}")
print(f"Best NCG Inflation Factor: {best_ncg_factor}")

plt.plot(ncg_inflation_factors, steam_injection_results)
plt.xlabel('NCG Inflation Factor')
plt.ylabel('Total Steam Injection')
plt.title('Total Steam Injection vs NCG Inflation Factor')
plt.grid()
plt.show()
