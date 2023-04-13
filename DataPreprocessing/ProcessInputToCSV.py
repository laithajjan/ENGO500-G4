import pandas as pd
from os import listdir
from os.path import isfile, join


inputDir = "../Data/InputXLSX/"
outputDir = "../Data/UnprocessedData/"

# Get file name of all files in given directory
filesOnly = [f for f in listdir(inputDir) if isfile(join(inputDir, f))]

for file in filesOnly:

    # Read content of Excel file
    read_file = pd.read_excel(inputDir + file, skiprows=[], header=0)

    # Replace line breaks in column headers with spaces
    read_file.columns = read_file.columns.str.replace('\n', '', regex=True)
    read_file.columns = read_file.columns.str.replace(' ', '', regex=True)
    read_file.columns = read_file.columns.str.replace('_x000D_', '', regex=True)

    # Get file name without extension
    fileName = file.split(".")[0]

    # Write the dataframe object into csv file
    read_file.to_csv(outputDir+fileName+".csv", index=False)
    print('Saved ' + fileName + ' to a csv')
