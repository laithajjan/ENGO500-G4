import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

features = ['Ncg/steam', 'CalDlyOil(m3/d)', 'PrdHours(hr)', 'NbrofWells', 'InjHours(hr)']
target = 'SOR'
features = ['CalDlyOil(m3/d)']
target = 'Ncg/steam'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2,
                                                    random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2,
                           n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters: ", best_params)
