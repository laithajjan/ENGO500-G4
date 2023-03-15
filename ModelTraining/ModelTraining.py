# Script to perform the training of the model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from time import time
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import seaborn as sns


from sklearn.model_selection import train_test_split


# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"


# Read in the processed data
df = pd.read_csv(input_directory + "KitchenSink.csv", header=0)
train_df, test_df = train_test_split(df, test_size=0.9)


# Display the headers in the csv
print(df.head(0))
feature_list = ['MonInjSteam(m3)']
#target = ['MonthlyOil(m3)', 'MonthlyGas(E3m3)', 'MonthlyWater(m3)']
target = ['MonthlyOil(m3)']


X = train_df[feature_list].values
y = train_df[target].values


# Fit the data to the model
start_time = time()
model = LinearRegression().fit(X, y)
training_time = time() - start_time
print('Training time: ' + str(training_time))
dump(model, output_directory + 'KitchenSinkBasicModel.pkl')




# Get the coefficient of determination (R^2) to evaluate the goodness of fit of the model
r_squared = model.score(X, y)
print("Coefficient of determination (R^2):", r_squared)


# Get the estimated coefficients (intercept and slope)
intercept, *coef = model.intercept_, *model.coef_
print("Intercept:", intercept)
print("Coefficients:", coef)


# Use the model to make predictions
X_test = df[feature_list].values
y_test = df[target].values
y_pred = model.predict(X_test)


# Plot the predicted vs actual values over the dates
plt.plot(df["ProdDate"], y_test, label="Actual Oil Produced")
plt.plot(df["ProdDate"], y_pred, label="Predicted Oil Produced")
plt.plot(df["ProdDate"], X_test, label="Injected Steam")
# Rotate x-axis labels by 90 degrees
plt.xticks(df.index[::6], rotation=90)
plt.xlabel("Date")
plt.ylabel("Oil Production")
plt.title("Actual vs Predicted Oil Production over Dates")
plt.legend()
plt.show()


# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)


# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)


# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Explained Variance Score (EVS):", evs)


# Plot the residuals
residuals = y_test - y_pred
sns.residplot(x=y_pred.flatten(), y=residuals.flatten(), lowess=True)
plt.xlabel("Predicted Oil Produced")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Oil Produced")
plt.show()


# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Oil Produced")
plt.ylabel("Predicted Oil Produced")
plt.title("Actual vs Predicted Oil Production")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()


# Use the model to make predictions on the test set
X_test = test_df[feature_list].values
y_test = test_df[target].values
y_pred = model.predict(X_test)


# Calculate the residuals for the test set
test_residuals = y_test - y_pred


# Plot residuals only
plt.plot(test_df["ProdDate"], test_residuals, marker='o', linestyle='')
plt.xticks(test_df.index[::6], rotation=90)
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.title("Residuals over Dates")
plt.show()


