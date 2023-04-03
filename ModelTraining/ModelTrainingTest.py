import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)
nan_values = data[data.isna().any(axis=1)]
print(nan_values)
# Calculate the steam-oil ratio (SOR)
#data['MonInjSteam(m3)'] = data['MonInjSteam(m3)'] / data['InjHours(hr)']
#data['MonthlyOil(m3)'] = data['MonthlyOil(m3)'] / data['PrdHours(hr)']
data['SOR'] = data['MonInjSteam(m3)'] / data['MonthlyOil(m3)']
data['Delta_SOR'] = data['SOR'].diff()
data['Delta_percent_SOR'] = data['SOR'].pct_change() * 100
data = data.iloc[1:]

# Calculate correlations
corr_matrix = data.corr()

# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Define input features and target
# feature_list = ['MonInjSteam(m3)', 'MonthlyOil(m3)', 'MonInjGas(E3m3)', 'PrdHours(hr)', 'InjHours(hr)', 'CumInjSteam(m3)', 'CumInjGas(E3m3)', 'CumPrdOil(m3)']
features = ['MonthlyOil(m3)', 'PrdHours(hr)', 'MonInjGas(E3m3)']
print('features:' + str(features))
target = 'MonInjSteam(m3)'
print('target:' + str(target))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Create and train the combined_model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# combined_model = LinearRegression().fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Generate a range of NCG injection values
ncg_injection_range = np.linspace(data['MonInjGas(E3m3)'].min(), data['MonInjGas(E3m3)'].max(), num=100)

# Initialize an empty list to store the predicted SOR
predicted_sor = []

# Iterate through the NCG injection values
for ncg_value in ncg_injection_range:
    # Create a new dataset with the same features as the test set,
    # but with the 'MonInjGas(E3m3)' feature set to the current NCG injection value
    modified_test_set = X_test.copy()
    modified_test_set['MonInjGas(E3m3)'] = ncg_value

    # Predict the SOR for the modified test set and average the predictions
    sor_pred = model.predict(modified_test_set)
    predicted_sor.append(np.mean(sor_pred))

# Plot the predicted SOR against the NCG injection values
plt.plot(ncg_injection_range, predicted_sor)
plt.xlabel('Monthly NCG Injected')
plt.ylabel('Predicted SOR')
plt.title('Effect of Monthly NCG Injection on Predicted SOR **Monthly effect**')
plt.grid()
plt.show()

# Calculate the SOR predictions for the test set
y_test_pred = model.predict(X_test)

# Scatter plot of actual vs. predicted SOR
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual SOR')
plt.ylabel('Predicted SOR')
plt.title('Actual vs. Predicted SOR')
plt.grid()
plt.show()

# Calculate feature importances
importances = model.feature_importances_

# Print feature importances
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.2f}")

# Plot feature importance
plt.bar(features, importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# Create interaction features
data['NCG_Steam_interaction'] = data['MonInjGas(E3m3)'] * data['MonInjSteam(m3)']
data['NCG_PrdHours_interaction'] = data['MonInjGas(E3m3)'] * data['PrdHours(hr)']

# Add interaction features to the feature list
features_with_interaction = features + ['NCG_Steam_interaction', 'NCG_PrdHours_interaction']

# Split data into training and testing sets with interaction features
X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(data[features_with_interaction], data[target],
                                                                    test_size=0.2, random_state=42)

# Create and train the combined_model with interaction features
model_int = RandomForestRegressor(n_estimators=100, random_state=42)
model_int.fit(X_train_int, y_train_int)

# Predict the target variable on the test set with interaction features
y_pred_int = model_int.predict(X_test_int)

# Calculate evaluation metrics for the combined_model with interaction features
mse_int = mean_squared_error(y_test_int, y_pred_int)
r2_int = r2_score(y_test_int, y_pred_int)

print(f'Mean Squared Error (with interaction): {mse_int:.2f}')
print(f'R^2 Score (with interaction): {r2_int:.2f}')

# Define the feature you want to analyze
feature_to_plot = 'MonInjGas(E3m3)'

# Create the PartialDependenceDisplay object
display = PartialDependenceDisplay.from_estimator(model, X_train, [feature_to_plot], feature_names=features)

# Customize the plot title
display.axes_[0][0].set_title('Partial Dependence of SOR on Monthly NCG Injection')

# Show the plot
plt.show()

cum_ncg_injection_range = np.linspace(data['CumInjGas(E3m3)'].min(), data['CumInjGas(E3m3)'].max(), num=100)
predicted_sor = []

for cum_ncg_value in cum_ncg_injection_range:
    modified_test_set = X_test.copy()
    modified_test_set['CumInjGas(E3m3)'] = cum_ncg_value

    sor_pred = model.predict(modified_test_set)
    predicted_sor.append(np.mean(sor_pred))

plt.plot(cum_ncg_injection_range, predicted_sor)
plt.xlabel('Cumulative NCG Injected')
plt.ylabel('Predicted SOR')
plt.title('Effect of Cumulative NCG Injection on Predicted SOR')
plt.grid()
plt.show()

data['Delta_SOR'] = data['SOR'].diff()
data = data.iloc[1:, :]

plt.scatter(data['CumInjGas(E3m3)'], data['Delta_SOR'])
plt.xlabel('Cumulative NCG Injected')
plt.ylabel('Change in SOR')
plt.title('Relationship between Cumulative NCG and Change in SOR')
plt.grid()
plt.show()

plt.scatter(data['MonInjGas(E3m3)'], data['Delta_SOR'])
plt.xlabel('Monthly NCG Injected')
plt.ylabel('Change in SOR')
plt.title('Relationship between Monthly NCG and Change in SOR')
plt.grid()
plt.show()

# Convert the 'ProdDate' column to datetime
data['ProdDate'] = pd.to_datetime(data['ProdDate'])

# Split data into two subsets based on whether NCG was injected or not
data_no_ncg = data[data['CumInjGas(E3m3)'] == 0]
data_with_ncg = data[data['CumInjGas(E3m3)'] > 0]

# Calculate the average SOR for each time period
avg_sor_no_ncg = data_no_ncg['SOR'].mean()
avg_sor_with_ncg = data_with_ncg['SOR'].mean()

print(f"Average SOR without NCG injection: {avg_sor_no_ncg:.2f}")
print(f"Average SOR with NCG injection: {avg_sor_with_ncg:.2f}")

# Plot the SOR values over time for both subsets
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_no_ncg['ProdDate'], data_no_ncg['SOR'], label='No NCG Injection', marker='o', linestyle='-')
ax.plot(data_with_ncg['ProdDate'], data_with_ncg['SOR'], label='With NCG Injection', marker='o', linestyle='-')

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('SOR')
ax.set_title('SOR values over time (With and Without NCG Injection)')
ax.legend()
ax.grid()

# Display only each year on the x label
years = mdates.YearLocator()  # Locate years on the x-axis
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

fig.autofmt_xdate()
plt.show()

# Split data into two subsets based on whether NCG was injected or not
data_no_ncg = data[data['CumInjGas(E3m3)'] < 4000]
data_with_ncg = data[data['CumInjGas(E3m3)'] > 4000]

# Calculate the average Delta SOR for each time period
avg_delta_sor_no_ncg = data_no_ncg['Delta_SOR'].mean()
avg_delta_sor_with_ncg = data_with_ncg['Delta_SOR'].mean()

# Calculate the percentage improvement in Delta SOR
percentage_improvement = (avg_delta_sor_no_ncg - avg_delta_sor_with_ncg) / avg_delta_sor_no_ncg * 100

print(f"Percentage improvement in Delta SOR due to NCG injection: {percentage_improvement:.2f}%")

# Create a bar plot to visualize the results
labels = ['No NCG Injection', 'With NCG Injection']
values = [avg_delta_sor_no_ncg, avg_delta_sor_with_ncg]

plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.ylabel('Average Delta SOR')
plt.title('Average Delta SOR Comparison (With and Without NCG Injection)')
plt.grid(axis='y')

# Annotate the plot with percentage improvement
plt.text(1, avg_delta_sor_with_ncg / 2, f"{percentage_improvement:.2f}% improvement", fontsize=12, color='white',
         ha='center')

plt.show()

from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(data_no_ncg['SOR'], data_with_ncg['SOR'])
print(f"T-test statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.5f}")

