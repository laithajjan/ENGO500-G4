import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

# Black listed features not to use
# feature_blacklist = ['CalInjSteam(m3/d)', 'MonInjSteam(m3)', 'SOR', 'cSOR', 'min_sor', 'daily_sor', 'MonthlyOil(m3)',
#                      'AvgDlyOil(m3/d)', 'CumPrdGas(E3m3)', 'InjHours(hr)', 'WCT(%)', 'WOR(m3/m3)',
#                      'AvgInjSteam(m3/d)', 'CumInjSteam(m3)', 'SteamRateOverOilRate',
#                      'oil/scale', 'SteamRate/PrdHours', 'CumPrdHours(hr)/OilRate', 'CumOil/CumPrdHours',
#                      'cum_produced_volume', 'CumPrdOil(m3)', 'MonthlyGas(E3m3)', 'AvgDlyGas(E3m3/d)',
#                      'CalDlyGas(E3m3/d)', 'CumPrdGas(E3m3)', 'MonthlyWater(m3)', 'AvgDlyWtr(m3/d)', 'CalDlyWtr(m3/d)',
#                      'CumPrdWtr(m3)', 'MonthlyFluid(m3)', 'AvgDlyFluid(m3/d)', 'CalDlyFluid(m3/d)', 'CumPrdFluid(m3)',
#                      'MonInjGas(E3m3)', 'AvgInjGas(E3m3/d)', 'CalInjGas(E3m3/d)', 'CumInjGas(E3m3)', 'MonInjWtr(m3)',
#                      'AvgInjWtr(m3/d)', 'CalInjWtr(m3/d)', 'CumInjWtr(m3)', 'MonInjSlv(E3m3)', 'AvgInjSlv(E3m3/d)',
#                      'CalInjSlv(E3m3/d)', 'CumInjSlv(E3m3)', 'InjHours(hr)', 'PrdHours(hr)', 'Inj/PrdHours(hr)',
#                      'WCT(%)', 'OCT(%)', 'GOR(m3/m3)', 'WGR(m3/E3m3)', 'WOR(m3/m3)', 'NbrofWells', 'MonInjSteam(m3)',
#                      'AvgInjSteam(m3/d)', 'CalInjSteam(m3/d)', 'CumInjSteam(m3)', 'steam/wellhours(m3/d/well/hr)',
#                      'CumulativeMonth']
feature_blacklist = ['delta_SOR']

target = 'delta_SOR'

# Get a list of all the numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Use all numeric column names in the data DataFrame as features, except for the feature_blacklist variable
all_features = [col for col in numeric_columns if col not in feature_blacklist]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[all_features], data[target], test_size=0.2,
                                                    random_state=42)

# Replace infinities with 0
X_train.replace([np.inf, -np.inf], 0, inplace=True)
X_test.replace([np.inf, -np.inf], 0, inplace=True)

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define a cross-validation object with 5 folds
cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Create the RFECV object
selector = RFECV(model, step=1, cv=cv, scoring='neg_mean_squared_error', verbose=2)

# Fit the RFECV object to the train data
selector.fit(X_train, y_train)

# Get the selected features
all_features_np = np.array(all_features)
selected_features = all_features_np[selector.support_].tolist()

# Train the model on the selected features
model.fit(X_train[selected_features], y_train)

# Make predictions and calculate the RMSE
y_pred = model.predict(X_test[selected_features])
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("Selected features:", selected_features)
print("RMSE:", rmse)
