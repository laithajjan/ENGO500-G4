import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Load the dataset
data = pd.read_csv(input_directory + 'Wabiskaw.csv')
data = data[data['CumulativeMonth'] > 0]

# Split the dataset into two subsets: one where MonInjGas(E3m3) is zero, and one where it is not
data_zero_ncg = data[data['CumulativeMonth'] > 450]
data_nonzero_ncg = data[data['MonInjGas(E3m3)'] != 0]

# Define the features and target variable for each model
features_zero_ncg = ['MonInjSteam(m3)', 'MonthlyOil(m3)']
target_zero_ncg = 'SOR'

features_nonzero_ncg = ['NCG/S', 'MonthlyOil(m3)']
target_nonzero_ncg = 'SOR'

# Split each subset into train and test sets
X_train_zero_ncg, X_test_zero_ncg, y_train_zero_ncg, y_test_zero_ncg = train_test_split(
    data_zero_ncg[features_zero_ncg], data_zero_ncg[target_zero_ncg], test_size=0.2, random_state=42)

X_train_nonzero_ncg, X_test_nonzero_ncg, y_train_nonzero_ncg, y_test_nonzero_ncg = train_test_split(
    data_nonzero_ncg[features_nonzero_ncg], data_nonzero_ncg[target_nonzero_ncg], test_size=0.2, random_state=42)

# Train the models
model_zero_ncg = LinearRegression()
model_zero_ncg.fit(X_train_zero_ncg, y_train_zero_ncg)

model_nonzero_ncg = LinearRegression()
model_nonzero_ncg.fit(X_train_nonzero_ncg, y_train_nonzero_ncg)

# Evaluate the models on the test set
print('Model trained without NCG:')
print('R-squared:', model_zero_ncg.score(X_test_zero_ncg, y_test_zero_ncg))
print('Cross-validation scores:', cross_val_score(model_zero_ncg, X_test_zero_ncg, y_test_zero_ncg, cv=5))

print('Model trained with NCG:')
print('R-squared:', model_nonzero_ncg.score(X_test_nonzero_ncg, y_test_nonzero_ncg))
print('Cross-validation scores:', cross_val_score(model_nonzero_ncg, X_test_nonzero_ncg, y_test_nonzero_ncg, cv=5))

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate the VIF for each feature in the model trained without NCG
vif_zero_ncg = pd.DataFrame()
vif_zero_ncg["Feature"] = features_nonzero_ncg
vif_zero_ncg["VIF"] = [variance_inflation_factor(X_train_nonzero_ncg.values, i) for i in range(X_train_nonzero_ncg.shape[1])]

print('VIF for the model trained without NCG:')
print(vif_zero_ncg)