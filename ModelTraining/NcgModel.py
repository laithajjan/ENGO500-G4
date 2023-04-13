import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
ncg_data = pd.read_csv(input_directory + "WabiskawEdit.csv", header=0)

features = ['CalDlyOil(m3/d)']
target = 'Ncg/steam'

# Create the NCG model
ncg_X_train, ncg_X_test, ncg_y_train, ncg_y_test = train_test_split(ncg_data[features], ncg_data[target], test_size=0.2,
                                                                    random_state=42)
ncg_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, max_features=1.0,
                                  min_samples_leaf=4, min_samples_split=10)

ncg_model.fit(ncg_X_train, ncg_y_train)
ncg_train_score = ncg_model.score(ncg_X_train, ncg_y_train)
ncg_test_score = ncg_model.score(ncg_X_test, ncg_y_test)
ncg_cv_scores = cross_val_score(ncg_model, ncg_X_train, ncg_y_train, cv=5)

# Calculate the RMSE
rmse = sqrt(mean_squared_error(ncg_y_test, ncg_model.predict(ncg_X_test)))

# Calculate the range of the target variable
target_min = ncg_data[target].min()
target_max = ncg_data[target].max()
target_range = target_max - target_min

# Calculate the NRMSE
nrmse = rmse / target_range

print("NCG Model Scores:")
print("Train score: ", ncg_train_score)
print("Test score: ", ncg_test_score)
print("Cross-validation scores: ", ncg_cv_scores)
print("Root Mean Squared Error: ", str(rmse))
print("Normalized Root Mean Squared Error: ", nrmse)

# Save the model to the output directory as 'OptimalNcgModel'
print('Saving model to: ' + output_directory)
joblib.dump(ncg_model, output_directory + 'WabiskawNcgModel.pkl')

ncg_y_pred = ncg_model.predict(ncg_X_test[features])

# Create a scatter plot of true values vs predicted values
plt.scatter(ncg_y_test, ncg_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')
plt.grid()
plt.show()

