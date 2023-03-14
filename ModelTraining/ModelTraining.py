# Script to perform the training of the model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from time import time
from joblib import dump

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

