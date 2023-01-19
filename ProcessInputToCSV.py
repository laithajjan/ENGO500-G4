import pandas as pd

inputDir = "inputXLSX/"
outputDir = "Data/UnprocessedData/"

#Get file name of all files in given directory
from os import listdir
from os.path import isfile, join
filesOnly = [f for f in listdir(inputDir) if isfile(join(inputDir, f))]

for file in filesOnly:

    #Read content of excel file
    read_file = pd.read_excel(inputDir + file)

    #Get file name without extension
    fileName = file.split(".")[0]

    # Write the dataframe object into csv file
    read_file.to_csv(outputDir+fileName+".csv", index=None, header=True)
