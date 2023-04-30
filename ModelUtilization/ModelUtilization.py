import os
import joblib
import pandas as pd
from matplotlib import pyplot as plt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
model_directory = "../Models/"

model = joblib.load(model_directory + 'Model.pkl')
optimal_ncg_model = joblib.load(model_directory + 'WabiskawNcgModel.pkl')


def process_csv_file(file_path, model, name, optimal_ncg=False):
    data = pd.read_csv(file_path, header=0)

    features = ['Ncg/steam', 'CalDlyOil(m3/d)', 'PrdHours(hr)', 'NbrofWells', 'InjHours(hr)']
    target = 'SOR'

    if optimal_ncg:
        data['Ncg/steam'] = optimal_ncg_model.predict(data['CalDlyOil(m3/d)'].values.reshape(-1, 1))
        data = data[data['MonIn jGas(m3)'] == 0].copy()

    if not data.empty:
        data[target] = model.predict(data[features])
        data['predicted_steam'] = data['SOR'] * data['CalDlyOil(m3/d)']

        ncg_vs_zero_ncg = pd.DataFrame()
        ncg_vs_zero_ncg['steam_difference'] = data['CalInjSteam(m3/d)'] - data['CalInjSteam(m3/d)']

        # Plotting of the model predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(data['CumulativeMonth'], data['CalInjSteam(m3/d)'],
                    label='Original Daily Injected Steam (m3/d)', alpha=0.7)
        plt.scatter(data['CumulativeMonth'], data['predicted_steam'],
                    label='Predicted Daily Injected Steam (m3/d)', alpha=0.7)

        steam_saved = data['predicted_steam'].sum() - data['CalInjSteam(m3/d)'].sum()
        percent_steam_saved = 100 * steam_saved / (data['CalInjSteam(m3/d)']).sum()

        plt.xlabel('Cumulative Month')
        plt.ylabel('Calculated Daily Steam Produced (m3/d)')
        plt.legend()
        if optimal_ncg:
            print('The Steam Required without optimal ncg: ' + str(data['CalInjSteam(m3/d)'].sum()))
            print('The Steam Required with optimal ncg: ' + str(data['predicted_steam'].sum()))
            print('The percent difference in non ncg vs ncg: ' +
                  str(percent_steam_saved))
            plt.title('Optimal NCG Injection Predictions: ' + name + ' ' + str(int(abs(percent_steam_saved))) +
                      '% Steam Savings Possible')
        else:
            print('The Steam Required in the true values: ' + str(data['CalInjSteam(m3/d)'].sum()))
            print('The Steam Required with predicted values: ' + str(data['predicted_steam'].sum()))
            print('The percent difference in predicted and true: ' +
                  str(percent_steam_saved))
            plt.title('Steam Injection values for site: ' + name + ' ' + str(int(abs(percent_steam_saved))) +
                      '% Prediction Error')
        plt.show()


csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

# Process each .csv file under regular conditions
for csv_file in csv_files:
    file_path = os.path.join(input_directory, csv_file)
    print(f"Processing {csv_file}")
    process_csv_file(file_path, model, csv_file)

# Process each .csv file with optimal NCG
for csv_file in csv_files:
    file_path = os.path.join(input_directory, csv_file)
    print(f"Processing {csv_file}")
    process_csv_file(file_path, model, csv_file, optimal_ncg=True)
