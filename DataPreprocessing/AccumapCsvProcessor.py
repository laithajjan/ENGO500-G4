from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd


class AccumapCsvProcessor:  # untested
    def __init__(self, input_directory, output_directory, start_row=8, debug_printing=False):
        # Class constructor

        # Set the directories to be used
        self.input_directory = input_directory
        self.output_directory = output_directory

        # Set the first row of data without headers
        self.start_row = start_row

        # Set debug_printing printing
        self.debug_printing = debug_printing

        self.preprocessed_file_count = 0

        return

    def loadCsvAsNumpy(self, file_name):
        # Read a comma deliminated csv file with file name file_name and return numpy array

        # Using pandas read_csv(), setting the delimiter to ',' and header=None, we get a DataFrame from a csv file
        csv = pd.read_csv(self.input_directory + file_name, sep=',', header=None)

        # Change the blank Nan values to 0s
        csv = csv.fillna(0)

        # Returning the DataFrame as a numpy array
        return np.asarray(csv.to_numpy())

    def formatAccumapData(self, data, start_row):
        # Format Accumap data into numpy array

        # Setting the start row of the data to the first line of actual data
        formatted_data = data[start_row:]
        return formatted_data

    def saveNumpyAsCsv(self, filename, data):
        # Convert to pd DataFrame, then convert to csv, save to output directory with no header
        pd.DataFrame(data).to_csv(self.output_directory + filename, index=None, header=None)

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

    def preprocessFile(self, filename):
        self.debugPrint("Preprocessing " + filename)
        # Load the csv file into a numpy array
        file_data = self.loadCsvAsNumpy(filename)

        # Format from Accumap to header-less data
        file_data = self.formatAccumapData(file_data, self.start_row)

        # Save the formatted data as a csv in the output directory
        self.saveNumpyAsCsv(filename, file_data)

        return

    def debugPrint(self, string):
        if self.debugPrint:
            print(string)

        return
