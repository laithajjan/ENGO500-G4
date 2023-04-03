import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

combined_data = pd.read_csv(input_directory + 'Wabiskaw.csv')
combined_data['ncg/steam'] = combined_data['MonInjGas(E3m3)']*1000/combined_data['MonInjSteam(m3)']
combined_data['ln_NCG'] = np.log(combined_data['MonInjGas(E3m3)']*1000)
combined_data = combined_data[combined_data['CumulativeMonth'] > 450]
zeros = (combined_data['NewOilRate'] == 0).sum()
print(zeros)

combined_data_ncg = combined_data[combined_data['CumInjGas(E3m3)'] != 0]
combined_data_non_ncg = combined_data[combined_data['CumInjGas(E3m3)'] == 0]

corr_matrix_ncg = combined_data_ncg.corr()
sns.set(font_scale=0.6)
# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_ncg, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix ncg')
plt.show()

corr_matrix_non_ncg = combined_data_non_ncg.corr()

# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_non_ncg, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix non ncg')
plt.show()
plt.show()

# Split the data based on the condition
gas_not_zero = combined_data[(combined_data['CumInjGas(E3m3)'] > 0)]
gas_zero = combined_data[(combined_data['CumInjGas(E3m3)'] == 0)]


def plot_scatter_and_line(data, title):
    filtered_data = data[data['CumulativeMonth'] >= 0]
    y = filtered_data['MonInjSteam(m3)']
    x = filtered_data['MonthlyOil(m3)']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_of_best_fit = slope * x + intercept
    plt.subplots(figsize=(12, 8))
    plt.scatter(x, y, label="Data")
    plt.plot(x, line_of_best_fit, color="red", label="Line of best fit")
    plt.text(0.1, 0.9, f'R-squared: {r_value ** 2:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Equation: y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes)
    plt.ylabel('Monthly Injected Steam (m3)')
    plt.xlabel('Monthly Oil Produced (m3)')
    plt.title(title)
    plt.legend()
    plt.show()


filename = 'Wabiskaw'
# Plot the first scatter plot when 'CalInjGas(E3m3)' is not equal to 0
plot_scatter_and_line(combined_data, "Plot of Site: " + str(filename))
plot_scatter_and_line(gas_not_zero, "NCG Plot of Site: " + str(filename))

# Plot the second scatter plot when 'CalInjGas(E3m3/d)' is equal to 0
plot_scatter_and_line(gas_zero, "Non NCG Plot of Site: " + str(filename))

def plot_scatter_and_line_both(data_ncg, data_non_ncg, title):
    # Plot NCG data
    y_ncg = data_ncg['OilRate']
    x_ncg = data_ncg['WOR(m3/m3)']
    color_ncg = data_ncg['MonInjGas(E3m3)']
    plt.scatter(x_ncg, y_ncg, c=color_ncg, cmap='Blues', label="NCG", alpha=0.6)

    # Plot non-NCG data
    y_non_ncg = data_non_ncg['SOR']
    x_non_ncg = data_non_ncg['CumulativeMonth']
    color_non_ncg = data_non_ncg['MonInjGas(E3m3)']
    plt.scatter(x_non_ncg, y_non_ncg, c=color_non_ncg, cmap='Blues', label="Non-NCG", alpha=0.6)

    plt.ylabel('OilRate')
    plt.xlabel('WOR')
    plt.title(title)
    plt.legend()
    plt.show()

filename = 'Combined Data'
plot_scatter_and_line_both(combined_data_ncg, combined_data_non_ncg, "Comparison of NCG and Non-NCG: " + str(filename))
