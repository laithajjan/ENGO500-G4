import AccumapCsvProcessor

# Script to perform Data preprocessing

# Setting the input and output directories
input_directory = "../Data/UnprocessedData/"
output_directory = "../Data/ProcessedData/"

# Set the column numbers of the features to extract
feature_list = ['UWI', 'WellName']

# Create a new AccumapCsvPreprocessor with the input and output directories
accumapCsvProcessor = AccumapCsvProcessor.AccumapCsvProcessor(input_directory, output_directory,
                                                              feature_list=feature_list, debug_printing=True,
                                                              steady_state_only=False)

# Run the preprocessing for the set directories
accumapCsvProcessor.preprocessSetDirectory()
