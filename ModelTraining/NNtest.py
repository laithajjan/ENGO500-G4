import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

# Define the features and target
features = ['MonthsNCG', 'CumulativeMonth', 'MonInjGas(E3m3)']
target = 'MonInjSteam(m3)'

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Build and train the model
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', batch_size=32, max_iter=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred2 = model.predict(data[features])
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R^2 Score: ", r2_score(y_test, y_pred))
print("R^2 Score: ", r2_score(data[target], y_pred2))

from sklearn.inspection import PartialDependenceDisplay
feature_names = ['cSOR', 'CumulativeMonth', 'MonInjGas(E3m3)', 'CumInjGas(E3m3)', 'OCT(%)']
# Define the feature(s) to calculate partial dependence on
features = ['MonInjGas(E3m3)']

# Instantiate the PartialDependenceDisplay object
display = PartialDependenceDisplay.from_estimator(model, X_test, features=features)

# Calculate the partial dependence of MonInjGas on MonInjSteam
fig, ax = plt.subplots(figsize=(8, 4))
display.plot(ax=ax)
plt.show()

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({"feature": feature_names, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
importance_df.sort_values(by="importance_mean", ascending=False, inplace=True)

sns.barplot(x="importance_mean", y="feature", data=importance_df, xerr=importance_df["importance_std"])
plt.title("Feature Importances")
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.show()

sns.pairplot(data, x_vars=features, y_vars=[target], diag_kind="hist", corner=True)
plt.show()

correlations = data[features + [target]].corr()
sns.heatmap(correlations, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": model.predict(X_test), "NCG": X_test['MonthsNCG']})
sns.scatterplot(x="y_true", y="y_pred", hue="NCG", data=predictions_df, palette="viridis")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Model Predictions vs. True Values")
plt.legend(title="NCG")
plt.show()

# Find the maximum NCG value in the dataset
max_ncg = data['MonthsNCG'].max()

# Create an empty DataFrame to store the results
results_df = pd.DataFrame()


