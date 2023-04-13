import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from math import sqrt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

# [The rest of your data preprocessing code here]

features = ['delta_OCT(%)', 'delta_oil', 'delta_wells', 'delta_lnncg', 'delta_Inj/PrdHours', 'delta_cSOR']
target = 'delta_steam'

ncg_data = data[data['CumInjGas(E3m3)'] != 0]

# Create the NCG model
ncg_X_train, ncg_X_test, ncg_y_train, ncg_y_test = train_test_split(ncg_data[features], ncg_data[target], test_size=0.2,
                                                                    random_state=42)

ncg_model = XGBRegressor(n_estimators=25, random_state=42)
ncg_model.fit(ncg_X_train, ncg_y_train)
ncg_train_score = ncg_model.score(ncg_X_train, ncg_y_train)
ncg_test_score = ncg_model.score(ncg_X_test, ncg_y_test)
ncg_cv_scores = cross_val_score(ncg_model, ncg_X_train, ncg_y_train, cv=5)

print("NCG Model Scores:")
print("Train score: ", ncg_train_score)
print("Test score: ", ncg_test_score)
print("Cross-validation scores: ", ncg_cv_scores)
print("Root Mean Squared Error: ", str(sqrt(mean_squared_error(ncg_y_test, ncg_model.predict(ncg_X_test)))))
print("R^2 Score: ", r2_score(ncg_y_test, ncg_model.predict(ncg_X_test)))

# Calculate VIF scores
X = ncg_data[features]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("VIF Scores:")
print(vif_data)

# Save the model to the output directory as 'OptimalNcgModel'
print('Saving model to: ' + output_directory)
joblib.dump(ncg_model, output_directory + 'OptimalNcgModel.pkl')