from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


class AccumapCsvProcessor:
    def __init__(self, input_directory, output_directory, header_row=0, start_row=1, debug_printing=False,
                 feature_list=[], steady_state_only=True, max_change_percent=30):
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

        for file in file_list:
            # Separate file's name components into name and extension
            file_extension = file.split('.')[1]

            # Check that the extension is a .csv
            if file_extension == 'csv':
                self.preprocessFile(file)
                self.preprocessed_file_count += 1

        self.debugPrint("Number of files preprocessed: " + str(self.preprocessed_file_count))

        return

    def trimFeatures(self, data):
        data = data[self.feature_list]
        return data

    def trimSteadyState(self, data):
        # Needs to be changed so that the MonSteamInj column is used instead
        steam_injected_column = data[:, 0]
        last_value = float(steam_injected_column[0])
        percent_list = [999]

        for value in steam_injected_column[1:]:
            value = float(value)
            last_value = float(last_value)

            if value == 0:
                percent_change = 999
            else:
                percent_change = abs((value - last_value) / value * 100)

            percent_list.append(percent_change)
            last_value = value

        trimmed_data = []
        trimmed_data_index = 0

        for index, percent in enumerate(percent_list):
            if percent <= self.max_change_percent:
                trimmed_data.append(data[index, :])
                trimmed_data_index += 1

        return trimmed_data

    def preprocessFile(self, filename):
        self.debugPrint("Preprocessing " + filename)
        # Load the csv file into a numpy array
        file_data = self.loadCsvAsDataFrame(filename)

        # INSERT CODE TO CONVERT TO ALEX'S STYLE/SHAPE OF DATA

        # Format from Accumap to desired shape
        file_data = self.formatAccumapData(file_data)  # THIS FUNCTION CURRENTLY RETURNS WHAT YOU PUT IN

        # END OF CODE TO CONVERT TO ALEX'S STYLE/SHAPE OF DATA

        # Trim the data to contain only the desired features
        file_data = self.trimFeatures(file_data)

        # Trim the data to only include steady_state numbers
        if self.steady_state_only:
            file_data = self.trimSteadyState(file_data)

        # Save the formatted data as a csv in the output directory
        self.saveDataFrameAsCsv(filename, file_data)

        return

    def debugPrint(self, string):
        if self.debugPrint:
            print(string)

        return
