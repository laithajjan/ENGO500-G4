import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns


class AccumapCsvProcessor:
    def __init__(self, input_directory, output_directory, header_row=0, start_row=1, debug_printing=False,
                 feature_list=[], steady_state_only=True, max_change_percent=30, outlier_removal_percent=100, sigma=2,
                 show_plots=True, combine_data=True, ncg_only=False):
        # Class constructor

        # Set the directories to be used
        self.input_directory = input_directory
        self.output_directory = output_directory

        # Set the first row of data without headers
        self.start_row = start_row
        self.header_row = header_row
        self.header_row_data = []

        # Set debug_printing printing
        self.debug_printing = debug_printing

        # Counter to see how many files are preprocessed from a directory
        self.preprocessed_file_count = 0

        # Column numbers to extract from the data
        self.feature_list = feature_list

        # Boolean to only use steady state values. Default=True
        self.steady_state_only = steady_state_only

        # Maximum monthly percent change in data to consider steady state
        self.max_change_percent = max_change_percent

        # Minimum percent to consider when calculating the standard deviations of the columns
        self.outlier_removal_percent = outlier_removal_percent

        # Number of standard deviations to keep
        self.sigma = sigma

        # Boolean to show plots
        self.show_plots = show_plots

        # Produce a combined dataset
        self.combine_data = combine_data

        # Only include ncg sites
        self.ncg_only = ncg_only

        return

    def loadCsvAsDataFrame(self, file_name):
        # Read a comma deliminated csv file with file name file_name and return numpy array

        # Using pandas read_csv(), setting the delimiter to ',' and header=None, we get a DataFrame from a csv file
        csv = pd.read_csv(self.input_directory + file_name, sep=',', header=self.header_row, low_memory=False)

        # Change the blank Nan values to 0s
        csv = csv.fillna(0)
        csv.columns = csv.columns.str.replace(' ', '')

        # Returning the DataFrame as a numpy array
        return csv

    def formatAccumapData(self, data):
        formatted_data = data.groupby('ProdDate').aggregate(np.sum)

        return formatted_data

    def saveDataFrameAsCsv(self, filename, data):
        # Convert to pd DataFrame, then convert to csv, save to output directory with no header
        pd.DataFrame(data).to_csv(self.output_directory + filename, index=True)

        return

    def preprocessSetDirectory(self):
        # Grab all files from input directory
        file_list = [f for f in listdir(self.input_directory) if isfile(join(self.input_directory, f))]
        combined_data = []

        for file in file_list:
            # Separate file's name components into name and extension
            file_extension = file.split('.')[1]

            # Check that the extension is a .csv
            if file_extension == 'csv':
                preprocessed_file = self.preprocessFile(file)
                if self.combine_data:
                    combined_data.append(preprocessed_file)
                self.preprocessed_file_count += 1

        self.debugPrint("Number of files preprocessed: " + str(self.preprocessed_file_count))

        if self.combine_data:
            # Concatenate all the DataFrames into a single DataFrame
            combined_data = pd.concat(combined_data)

            # Save combined data to CSV
            combined_data.to_csv(os.path.join(self.output_directory, 'combined_data.csv'))

        return

    def trimFeatures(self, data):
        # data = data[self.feature_list].copy()
        data = data[['MonthlyOil(m3)', 'AvgDlyOil(m3/d)', 'CalDlyOil(m3/d)', 'CumPrdOil(m3)',
                     'MonthlyGas(E3m3)', 'AvgDlyGas(E3m3/d)', 'CalDlyGas(E3m3/d)',
                     'CumPrdGas(E3m3)', 'MonthlyWater(m3)', 'AvgDlyWtr(m3/d)',
                     'CalDlyWtr(m3/d)', 'CumPrdWtr(m3)', 'MonthlyFluid(m3)',
                     'AvgDlyFluid(m3/d)', 'CalDlyFluid(m3/d)', 'CumPrdFluid(m3)',
                     'MonInjGas(E3m3)', 'AvgInjGas(E3m3/d)', 'CalInjGas(E3m3/d)',
                     'CumInjGas(E3m3)', 'MonInjWtr(m3)', 'AvgInjWtr(m3/d)',
                     'CalInjWtr(m3/d)', 'CumInjWtr(m3)', 'MonInjSlv(E3m3)',
                     'AvgInjSlv(E3m3/d)', 'CalInjSlv(E3m3/d)', 'CumInjSlv(E3m3)',
                     'InjHours(hr)', 'PrdHours(hr)', 'Inj/PrdHours(hr)', 'WCT(%)', 'OCT(%)', 'GOR(m3/m3)',
                     'WGR(m3/E3m3)', 'WOR(m3/m3)',
                     'NbrofWells', 'MonInjSteam(m3)', 'AvgInjSteam(m3/d)', 'CalInjSteam(m3/d)',
                     'CumInjSteam(m3)']]

        return data

    def trimSteadyState(self, raw_data):

        self.debugPrint('The number of raw rows of the dataset: ' + str(raw_data.shape[0]))

        trimmed_data = raw_data[raw_data['CumInjSteam(m3)'] != 0].copy()
        trimmed_data['CumulativeMonth'] = np.arange(len(trimmed_data))
        trimmed_data['MonthsNCG'] = (trimmed_data['CumInjGas(E3m3)'] > 0).cumsum()

        # Filter out rows where CalInjSteam(m3/d) is 0
        trimmed_data = trimmed_data[trimmed_data['CalInjSteam(m3/d)'] != 0].copy()
        self.debugPrint('The number of non zero monthly steam injected rows of the dataset: ' +
                        str(trimmed_data.shape[0]))

        # Calculate and populate SOR and percent change SOR
        trimmed_data['SOR'] = trimmed_data['MonInjSteam(m3)'] / trimmed_data['MonthlyOil(m3)']

        # Calculate the percent change of the 12-month rolling window
        trimmed_data['percent_change_SOR'] = trimmed_data['SOR'].rolling(window=12).apply(
            lambda x: x.pct_change().iloc[-1]) * 100

        # Calculate and remove rows with a percent change that is outside the outlier threshold
        trimmed_data = trimmed_data[trimmed_data['percent_change_SOR'] < self.outlier_removal_percent]
        self.debugPrint('The number of rows with SOR percent change less than ' + str(self.outlier_removal_percent) +
                        ' percent of the dataset: ' + str(trimmed_data.shape[0]))

        # Calculate the std dev of the percent change SOR
        std_dev_percent_change_sor = trimmed_data['percent_change_SOR'].std()
        self.debugPrint('The standard deviation of the percent change SOR: ' + str(std_dev_percent_change_sor))

        # Replace rows that are less than the specified standard deviations with interpolated values
        mask = abs(trimmed_data['percent_change_SOR']) <= std_dev_percent_change_sor * self.sigma
        columns_to_interpolate = ['MonInjSteam(m3)', 'MonthlyOil(m3)', 'percent_change_SOR', 'CalInjSteam(m3/d)',
                                  'CalDlyOil(m3/d)']
        trimmed_data.loc[~mask, columns_to_interpolate] = np.nan

        # Create a new column to track which values are interpolated
        trimmed_data['is_interpolated'] = False

        # Set the is_interpolated column to True for interpolated values
        trimmed_data.loc[~mask, 'is_interpolated'] = True

        num_nans = trimmed_data.isna().sum().sum()
        self.debugPrint(f"There are {num_nans} NaN values in the DataFrame.")

        trimmed_data[columns_to_interpolate] = trimmed_data[columns_to_interpolate].interpolate(method='linear')

        num_nans = trimmed_data.isna().sum().sum()
        self.debugPrint(f"There are {num_nans} NaN values in the DataFrame.")

        trimmed_data = trimmed_data.fillna(method='ffill')
        nan_rows = trimmed_data[trimmed_data.isnull().any(axis=1)]

        if not nan_rows.empty:
            self.debugPrint("Rows with NaN values:")
            self.debugPrint(nan_rows.index)
            self.debugPrint("Attempting to drop rows with Nan values...")
            trimmed_data = trimmed_data.dropna()
            nan_rows = trimmed_data[trimmed_data.isnull().any(axis=1)]
            if not nan_rows.empty:
                self.debugPrint("Rows with NaN values:")
                self.debugPrint(nan_rows.index)
            else:
                self.debugPrint("The trimmed_data does not contain NaN values")

        else:
            self.debugPrint("The trimmed_data does not contain NaN values")

        self.debugPrint('The number of rows with SOR percent change less than ' + str(self.sigma) +
                        ' sigma of the dataset: ' + str(trimmed_data.shape[0]))

        #if not (trimmed_data[trimmed_data['is_interpolated'] == True]).empty:
            #self.debugPrint(
                #'Number of interpolated values: ' + str(trimmed_data[trimmed_data['is_interpolated'] == True].count))

        return trimmed_data

    def engineerFeatures(self, data):
        if all(elem in self.feature_list for elem in
               ['MonthlyOil(m3)', 'MonInjSteam(m3)', 'InjHours(hr)', 'PrdHours(hr)']):
            data['MonInjGas(m3)'] = data['MonInjGas(E3m3)'] * 1000
            data['CumInjGas(m3)'] = data['CumInjGas(E3m3)'] * 1000

            # Recalculate the cumulative oil for sites that
            data['CumPrdOilSinceSteamInj(m3)'] = data['MonthlyOil(m3)'].cumsum()

            # Calculate the Oil rate and Steam rate
            data['OilRate'] = data['MonthlyOil(m3)'] / data['PrdHours(hr)']
            data['SteamRate'] = data['MonInjSteam(m3)'] / data['InjHours(hr)']

            # Calculate the Steam rate over Oil rate
            data['SteamRateOverOilRate'] = data['SteamRate'] / data['OilRate']

            # Scaled features
            data['scaled_steam'] = data['MonInjSteam(m3)'] / data['CumInjSteam(m3)'].max()
            data['scaled_oil'] = data['MonthlyOil(m3)'] / data['CumPrdOil(m3)'].max()

            data['DeltaMonInjSteam(m3)'] = data['MonInjSteam(m3)'].diff() * data['CumPrdOilSinceSteamInj(m3)']

            data['delta_SOR'] = data['SOR'].diff()

            data['cSOR'] = data['CumInjSteam(m3)'] / data['CumPrdOil(m3)']
            data['cN/S'] = data['CumInjGas(m3)'] / data['CumInjSteam(m3)']
            data['Ncg/O'] = data['MonInjGas(m3)']/data['MonthlyOil(m3)']
            data['NCG/S'] = data['MonInjGas(m3)']/data['MonInjSteam(m3)']
            data['time'] = data['OilRate']/((data['WOR(m3/m3)']**2)*0.04 + 0.59*data['WOR(m3/m3)'] + 3)
            data['NewOilRate'] = data['OilRate']*data['time']

        return data

    def preprocessFile(self, filename):
        self.debugPrint("Preprocessing " + filename)
        # Load the csv file into a numpy array
        file_data = self.loadCsvAsDataFrame(filename)

        # Format from Accumap to desired shape
        file_data = self.formatAccumapData(file_data)

        # Trim the data to contain only the desired features
        file_data = self.trimFeatures(file_data)

        # Trim the data to only include steady_state numbers
        if self.steady_state_only:
            file_data = self.trimSteadyState(file_data)

        # Engineer extra features
        file_data = self.engineerFeatures(file_data)

        ###################

        # Calculate correlations
        file_data = file_data[file_data['CalInjSteam(m3/d)'] != 0]
        corr_matrix = file_data.corr()

        # Visualize correlations using a heatmap
        if self.show_plots:
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()

        # Calculate summary statistics
        summary_stats = file_data.describe()

        # Calculate skewness and kurtosis
        skewness = file_data.skew()
        kurtosis = file_data.kurtosis()

        # Identify outliers
        outlier_threshold = 3
        outliers = pd.DataFrame()
        for col in file_data.columns:
            col_mean = file_data[col].mean()
            col_std = file_data[col].std()
            col_min = col_mean - outlier_threshold * col_std
            col_max = col_mean + outlier_threshold * col_std
            outliers[col] = (file_data[col] < col_min) | (file_data[col] > col_max)

        # Print statistics
        # print("Start of site: " + str(filename))
        # print("Summary stats:")
        # print(summary_stats)
        # print("Skewness:")
        # print(skewness)
        # print("Kurtosis:")
        # print(kurtosis)
        # print("Outliers:")
        # print(outliers)
        # print("End of site: " + str(filename))

        # Split the data based on the condition
        gas_not_zero = file_data[(file_data['MonInjGas(E3m3)'] > 0)]
        gas_zero = file_data[(file_data['MonInjGas(E3m3)'] == 0)]

        # Plot the first scatter plot when 'CalInjGas(E3m3)' is not equal to 0
        self.plot_scatter_and_line(gas_not_zero, "NCG Plot of Site: " + str(filename))

        # Plot the second scatter plot when 'CalInjGas(E3m3/d)' is equal to 0
        self.plot_scatter_and_line(gas_zero, "Non NCG Plot of Site: " + str(filename))
        #######################
        self.plot_cum_month_vs_mon_inj_steam(file_data)
        # Save the formatted data as a csv in the output directory
        if self.ncg_only and not (file_data['MonInjGas(E3m3)'] != 0).any():
            self.debugPrint('Not an NCG site so it is being excluded.')
            return pd.DataFrame()

        self.saveDataFrameAsCsv(filename, file_data)

        return file_data

    def plot_scatter_and_line(self, data, title):
        if self.show_plots:
            if data.empty:  # check if dataframe is empty
                self.debugPrint(f"No data to plot for {title}")
                return
            y = data['MonInjSteam(m3)']
            x = data['MonthlyOil(m3)']
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line_of_best_fit = slope * x + intercept
            plt.subplots(figsize=(12, 8))
            plt.scatter(x, y, label="Data")
            plt.plot(x, line_of_best_fit, color="red", label="Line of best fit")
            plt.text(0.1, 0.9, f'R-squared: {r_value ** 2:.3f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.8, f'Equation: y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes)
            plt.ylabel('Monthly Steam (m3)')
            plt.xlabel('Monthly Oil (m3)')
            plt.title(title)
            plt.legend()
            plt.show()

    def plot_cum_month_vs_mon_inj_steam(self, data):
        if self.show_plots:
            # Filter the data to include only the NCG and non-NCG values
            ncg_data = data[data['MonInjGas(E3m3)'] > 0]
            non_ncg_data = data[data['MonInjGas(E3m3)'] == 0]

            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot the non-NCG data
            ax.scatter(non_ncg_data['CumulativeMonth'], non_ncg_data['MonInjSteam(m3)'], label='Non-NCG Data')

            # Plot the NCG data
            ax.scatter(ncg_data['CumulativeMonth'], ncg_data['MonInjSteam(m3)'], label='NCG Data')

            # Set the axis labels and title
            ax.set_xlabel('Cumulative Months')
            ax.set_ylabel('Monthly Steam Injection (m3)')
            ax.set_title('NCG vs Non-NCG Monthly Steam Injection')

            # Add the legend
            ax.legend()

            # Show the plot
            plt.show()

    def debugPrint(self, string):
        if self.debug_printing:
            print(string)

        return
