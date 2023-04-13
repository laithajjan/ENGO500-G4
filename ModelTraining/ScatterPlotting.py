import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

data = pd.read_csv(input_directory + 'combined_data.csv')


def plot_scatter_and_line(y_data, x_data, y_feature, x_feature):
    y = y_data[y_feature]
    x = x_data[x_feature]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_of_best_fit = slope * x + intercept
    plt.subplots(figsize=(12, 8))
    plt.scatter(x, y, label="Data")
    plt.plot(x, line_of_best_fit, color="red", label="Line of best fit")
    plt.text(0.1, 0.9, f'R-squared: {r_value ** 2:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Equation: y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes)
    plt.ylabel(y_feature)
    plt.xlabel(x_feature)
    plt.title(x_feature + ' vs: ' + y_feature)
    plt.legend()
    plt.show()


# Add features to test/Feature Engineering
# Filter/Trim the dataset even more
filtered_data = data.copy()

features = ['Ncg/steam', 'SOR']

# Replace any zero values in 'scale' with a small positive value
#data['scale'] = data['SOR']
#data['scale'] = data['scale'].replace(0, 1e-8)

# Divide each feature by 'scale'
#data[features] = data[features].div(data['scale'], axis=0)

#data = data.dropna()


# Create non-NCG and NCG data DataFrames
non_ncg_data = data[data['CumInjGas(E3m3)'] == 0].copy()
ncg_data = data[data['CumInjGas(E3m3)'] != 0].copy()



plot_scatter_and_line(ncg_data, ncg_data, 'SOR', 'CumulativeMonth')
plot_scatter_and_line(non_ncg_data, non_ncg_data, 'SOR', 'CumulativeMonth')

# Define the variable you want to use for color coding
color_variable = filtered_data['CalInjGas(E3m3/d)']
filtered_data['rolling_csor'] = filtered_data['SOR'].rolling(window=3).mean()
filtered_data['trail_rolling_sor'] = filtered_data['SOR'].rolling(window=6).mean()
filtered_data['delta_cSOR'] = filtered_data['CumInjSteam(m3)']/filtered_data['CumPrdOil(m3)']
filtered_data['scale'] = filtered_data['CumPrdHours(hr)']*filtered_data['delta_cSOR']
filtered_data['testx'] = filtered_data['delta_SOR']
filtered_data['testy'] = filtered_data['delta_SOR']

# Plot two features
x_feature = 'testx'
y_feature = 'testy'

# Filter the data based on the presence of NCG
ncg_data = filtered_data[filtered_data['CalInjGas(E3m3/d)'] > 0]
print('Mean ncg: ' + str(ncg_data['delta_delta_SOR'].mean()))
non_ncg_data = filtered_data[filtered_data['CalInjGas(E3m3/d)'] == 0]
non_ncg_data = non_ncg_data.dropna()
print('Mean non ncg: ' + str(non_ncg_data['delta_delta_SOR'].mean()))

# Calculate the mean values for 'testy' in both datasets
ncg_mean = ncg_data['testy'].mean()
non_ncg_mean = non_ncg_data['testy'].mean()

# Filter the data to keep only values within +/- sigma of the mean
sigma = 2

ncg_data_filtered = ncg_data[(ncg_data['testy'] >= ncg_mean * (-sigma)) & (ncg_data['testy'] <= ncg_mean * sigma)]
non_ncg_data_filtered = non_ncg_data[(non_ncg_data['testy'] >= non_ncg_mean * (-sigma)) & (non_ncg_data['testy'] <= non_ncg_mean * sigma)]

# Create the histogram for NCG cases
plt.hist(ncg_data_filtered['testy'].dropna(), bins=30, alpha=0.5, label='NCG')

# Create the histogram for non-NCG cases
plt.hist(non_ncg_data_filtered['testy'].dropna(), bins=30, alpha=0.5, label='Non-NCG')

# Add a legend, labels, and a title
plt.legend()
plt.xlabel('Testy')
plt.ylabel('Frequency')
plt.title('Histogram of Testy for NCG and Non-NCG cases (within +/- 100% of the mean)')

# Show the plot
plt.show()

x_data = filtered_data[x_feature]
y_data = filtered_data[y_feature]
slope, intercept, r_value, p_value, std_err = linregress(y=y_data, x=x_data)
line_of_best_fit = slope * x_data + intercept
plt.subplots(figsize=(12, 8))

# Add the 'c' parameter to color the dots based on the 'color_variable'
plt.scatter(y=y_data, x=x_data, label="Data", c=color_variable, cmap='viridis')

plt.plot(x_data, line_of_best_fit, color="red", label="Line of best fit")
plt.text(0.1, 0.9, f'R-squared: {r_value ** 2:.3f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f'Equation: y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes)
plt.ylabel(y_feature)
plt.xlabel(x_feature)
plt.title(x_feature + ' vs. ' + y_feature)
plt.legend()

# Add a colorbar to show the range of values for the color_variable
cbar = plt.colorbar()
cbar.set_label('Blue')

plt.show()
