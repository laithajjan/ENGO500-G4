from os.path import join
import AccumapCsvProcessor
import pandas as pd
import numpy as np
from os import listdir

# Script to perform Data preprocessing

# Setting the input and output directories
input_directory = "../Data/UnprocessedData/"
output_directory = "../Data/ProcessedData/"

# Create a new AccumapCsvPreprocessor with the input and output directories
accumapCsvProcessor = AccumapCsvProcessor.AccumapCsvProcessor(input_directory, output_directory, debug_printing=True)

# Run the preprocessing for the set directories
accumapCsvProcessor.preprocessSetDirectory()
