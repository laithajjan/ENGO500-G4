import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "Wabiskaw.csv", header=0)
clearwater_data = pd.read_csv(input_directory + 'Clearwater.csv', header=0)

data = data[data['CumulativeMonth'] > 450]
data = data[1:]
clearwater_data = clearwater_data[clearwater_data['CumulativeMonth'] > 250]
data_clearwater = clearwater_data[clearwater_data['CumulativeMonth'] > 250]

# data = pd.concat([data, data_clearwater])
# data = pd.read_csv(input_directory + 'combined_data.csv', header=0)

scale = pd.DataFrame()
scale['PrdHours(hr)'] = data['PrdHours(hr)']

data['scaled_oil'] = data['CalDlyOil(m3/d)'] / scale['PrdHours(hr)']
data['scaled_steam'] = data['CalInjSteam(m3/d)'] / scale['PrdHours(hr)']
data['scaled_ncg'] = data['CalInjGas(E3m3/d)'] / scale['PrdHours(hr)']
data['scaled_prd_water'] = data['CalDlyWtr(m3/d)'] / scale['PrdHours(hr)']
data['scaled_inj_water'] = data['CalInjWtr(m3/d)'] / scale['PrdHours(hr)']
data['cSOR/Prd'] = data['cSOR']/scale['PrdHours(hr)']

features = ['NewOilRate', 'OCT(%)', 'cSOR']
target = 'SOR'

ncg_data = data[data['CumInjGas(E3m3)'] != 0]

# Create the NCG model
ncg_X_train, ncg_X_test, ncg_y_train, ncg_y_test = train_test_split(ncg_data[features], ncg_data[target], test_size=0.2,
                                                                    random_state=42)

ncg_model = RandomForestRegressor(n_estimators=10, random_state=42)
ncg_model.fit(ncg_X_train, ncg_y_train)
ncg_train_score = ncg_model.score(ncg_X_train, ncg_y_train)
ncg_test_score = ncg_model.score(ncg_X_test, ncg_y_test)
ncg_cv_scores = cross_val_score(ncg_model, ncg_X_train, ncg_y_train, cv=5)

print("NCG Model Scores:")
print("Train score: ", ncg_train_score)
print("Test score: ", ncg_test_score)
print("Cross-validation scores: ", ncg_cv_scores)
print("Mean Squared Error: ", mean_squared_error(ncg_y_test, ncg_model.predict(ncg_X_test)))
print("R^2 Score: ", r2_score(ncg_y_test, ncg_model.predict(ncg_X_test)))

scale = pd.DataFrame()
scale['PrdHours(hr)'] = clearwater_data['PrdHours(hr)']

clearwater_data['scaled_oil'] = clearwater_data['CalDlyOil(m3/d)'] / scale['PrdHours(hr)']
clearwater_data['scaled_steam'] = clearwater_data['CalInjSteam(m3/d)'] / scale['PrdHours(hr)']
clearwater_data['scaled_ncg'] = clearwater_data['CalInjGas(E3m3/d)'] / scale['PrdHours(hr)']
clearwater_data['scaled_prd_water'] = clearwater_data['CalDlyWtr(m3/d)'] / scale['PrdHours(hr)']
clearwater_data['scaled_inj_water'] = clearwater_data['CalInjWtr(m3/d)'] / scale['PrdHours(hr)']
clearwater_data['cSOR/Prd'] = clearwater_data['cSOR']/scale['PrdHours(hr)']

zero_ncg_data = clearwater_data.copy()
zero_ncg_data['scaled_ncg'] = 0
print(zero_ncg_data['scaled_ncg'])

zero_ncg_data['scaled_steam'] = ncg_model.predict(zero_ncg_data[features])
zero_ncg_data = zero_ncg_data[zero_ncg_data['CumulativeMonth'] > 250]

ncg_vs_zero_ncg = pd.DataFrame()
ncg_vs_zero_ncg['steam_difference'] = zero_ncg_data['scaled_steam'] - ncg_data['scaled_steam']
plt.figure(figsize=(10, 6))
plt.scatter(clearwater_data['CumulativeMonth'], clearwater_data['MonInjSteam(m3)'],
            label='Steam Inj (m3/d)', alpha=0.7)
plt.scatter(zero_ncg_data['CumulativeMonth'], zero_ncg_data['scaled_steam'] * zero_ncg_data['MonthlyOil(m3)'],
            label='Predicted Steam Inj With NCG (m3/d)', alpha=0.7)

plt.xlabel('Cumulative Month')
plt.ylabel('Steam (m3/d)')
plt.legend()
plt.title('Steam Injection values for site: Clearwater')
plt.show()

# Calculate VIF scores
X = ncg_data[features]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

#print("VIF Scores:")
#print(vif_data)

steam_saved = (clearwater_data['scaled_steam'] * clearwater_data['PrdHours(hr)'] - zero_ncg_data['scaled_steam'] *
               zero_ncg_data['PrdHours(hr)']).sum()
print('The percent difference in non ncg vs ncg: ' + str(-steam_saved/(clearwater_data['scaled_steam'] * clearwater_data['PrdHours(hr)']).sum()))
