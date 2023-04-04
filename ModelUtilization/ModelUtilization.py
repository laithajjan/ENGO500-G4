# Script that utilizes the combined_model
import joblib
import os
import pandas as pd
from matplotlib import pyplot as plt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
model_directory = "../Models/"

ncg_model = joblib.load(model_directory + 'OptimalNcgModel.pkl')


def process_csv_file(file_path, model, name):
    data = pd.read_csv(file_path, header=0)
    data = data[data['CumulativeMonth'] > 0]
    # data = data[data['MonInjGas(m3)'] == 0]
    data = data[data['Prod/Inj'] < 10]

    features = ['NewOilRate', 'OCT(%)', 'cSOR']
    target = ['SOR', 'Ncg/Steam']

    scale = pd.DataFrame()
    scale['PrdHours(hr)'] = data['PrdHours(hr)']

    data['scaled_oil'] = data['CalDlyOil(m3/d)'] / scale['PrdHours(hr)']
    data['scaled_steam'] = data['CalInjSteam(m3/d)'] / scale['PrdHours(hr)']
    data['scaled_ncg'] = data['CalInjGas(E3m3/d)'] / scale['PrdHours(hr)']
    data['scaled_prd_water'] = data['CalDlyWtr(m3/d)'] / scale['PrdHours(hr)']
    data['scaled_inj_water'] = data['CalInjWtr(m3/d)'] / scale['PrdHours(hr)']
    data['cSOR/Prd'] = data['cSOR'] / scale['PrdHours(hr)']

    zero_ncg_data = data.copy()
    zero_ncg_data['scaled_ncg'] = 0

    zero_ncg_data['SOR'] = ncg_model.predict(zero_ncg_data[features])
    zero_ncg_data['scaled_steam'] = zero_ncg_data['SOR'] * zero_ncg_data['scaled_oil']
    zero_ncg_data['steam'] = zero_ncg_data['SOR'] * zero_ncg_data['MonthlyOil(m3)']

    ncg_vs_zero_ncg = pd.DataFrame()
    ncg_vs_zero_ncg['steam_difference'] = zero_ncg_data['scaled_steam'] - data['scaled_steam']

    plt.figure(figsize=(10, 6))
    plt.scatter(data['CumulativeMonth'], data['MonInjSteam(m3)'],
                label='Steam Inj (m3/d)', alpha=0.7)
    plt.scatter(zero_ncg_data['CumulativeMonth'], zero_ncg_data['steam'],
                label='Predicted Steam Inj With NCG (m3/d)', alpha=0.7)

    plt.xlabel('Cumulative Month')
    plt.ylabel('Steam (m3/d)')
    plt.legend()
    plt.title('Steam Injection values for site: ' + name)
    plt.show()

    # Calculate the actual and predicted Ncg/Steam ratios
    data['actual_ncg_steam_ratio'] = data['MonInjGas(m3)'] / data['MonInjSteam(m3)']

    steam_saved = zero_ncg_data['steam'].sum() - data['MonInjSteam(m3)'].sum()
    print('The Steam Required without optimal ncg: ' + str(data['MonInjSteam(m3)'].sum()))
    print('The Steam Required with optimal ncg: ' + str(zero_ncg_data['steam'].sum()))
    print('The percent difference in non ncg vs ncg: ' +
          str(100 * steam_saved / (data['MonInjSteam(m3)']).sum()))


csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

# Process each .csv file
for csv_file in csv_files:
    file_path = os.path.join(input_directory, csv_file)
    print(f"Processing {csv_file}")
    process_csv_file(file_path, ncg_model, csv_file)
