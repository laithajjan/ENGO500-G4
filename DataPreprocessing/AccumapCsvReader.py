import numpy as np


class AccumapCsvReader:  # untested
    def __init__(self, folder_path):
        # Class constructor
        self.folder_path = folder_path
        self.number_of_files_read = 0

        return

    def loadCsv(self, file_name):
        # Read a comma deliminated csv file with file name file_name and return numpy array
        csv = np.genfromtxt(file_name, ',')
        return csv

    def formatAccumapData(self, data):
        # Format Accumap data into numpy array
        # Get rid of first 8 rows of data
        return data[8:, :]

    def loadFormattedAccumapCsv(self, file_name):
        return self.formatAccumapData(self.loadCsv(file_name))

    def loadAllAccumapDataFromFolder(self):

        return
