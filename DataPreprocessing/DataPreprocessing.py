import AccumapCsvReader

# Script to perform Data preprocessing

# Read in the data from unprocessed csv downloaded from Accumap, probably best to read all .csv file in a folder together
# Script to perform Data preprocessing
inputDir = "../Data/UnprocessedData/"
outputDir = "../Data/ProcessedData/"

# Read in the data from unprocessed csv downloaded from Accumap, probably best to read all .csv file in a folder together
testDir = "../Data/UnprocessedData/ath_hanging.csv"

accumapCsvReader = AccumapCsvReader.AccumapCsvReader(inputDir)

accumapCsvReader.loadCsv(testDir)