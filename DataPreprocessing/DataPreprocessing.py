from os.path import join
import AccumapCsvProcessor
import pandas as pd
import numpy as np
from os import listdir

# Script to perform Data preprocessing

# Setting the input and output directories
input_directory = "../Data/UnprocessedData/"
output_directory = "../Data/ProcessedData/"

# Set the column numbers of the features to extract
steam_injected_row = 10
feature_list = [steam_injected_row, 0, 1]

# Create a new AccumapCsvPreprocessor with the input and output directories
accumapCsvProcessor = AccumapCsvProcessor.AccumapCsvProcessor(input_directory, output_directory,
                                                              feature_list=feature_list, debug_printing=True)

# Run the preprocessing for the set directories
accumapCsvProcessor.preprocessSetDirectory()
