import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

features = ['Ncg/steam', 'CalDlyOil(m3/d)', 'PrdHours(hr)', 'NbrofWells', 'InjHours(hr)']
target = 'SOR'

# Create the model
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2,
                                                    random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10, max_features=1,
                              min_samples_leaf=4, min_samples_split=10)
model.fit(X_train, y_train)
ncg_train_score = model.score(X_train, y_train)
ncg_test_score = model.score(X_test, y_test)
ncg_cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Calculate the RMSE
rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))

# Calculate the range of the target variable
target_min = data[target].min()
target_max = data[target].max()
target_range = target_max - target_min

# Calculate the NRMSE
nrmse = rmse / target_range

print("Model Scores:")
print("Train score: ", ncg_train_score)
print("Test score: ", ncg_test_score)
print("Cross-validation scores: ", ncg_cv_scores)
print("Root Mean Squared Error: ", str(rmse))
print("Normalized Root Mean Squared Error: ", nrmse)

# Calculate VIF scores
X = data[features]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("VIF Scores:")
print(vif_data)

# Save the model to the output directory as 'Model'
print('Saving model to: ' + output_directory)
joblib.dump(model, output_directory + 'Model.pkl')

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'feature': features, 'importance': importances})

print("Feature Importance:")
print(importance_df)


# Plot partial dependence
pdp_ncg = partial_dependence(model, X_train, ['Ncg/steam'])

plt.plot(pdp_ncg['values'][0], pdp_ncg['average'][0], lw=2)
plt.xlabel('Ncg/steam')
plt.ylabel('SOR Partial dependence')
plt.title('SOR partial dependence on Ncg/steam')
plt.show()
