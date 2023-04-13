import AccumapCsvProcessor

# Script to perform Data preprocessing

# Setting the input and output directories
input_directory = "../Data/UnprocessedData/"
output_directory = "../Data/ProcessedData/"

# Set the column numbers of the features to extract
feature_list = ['MonthlyOil(m3)', 'AvgDlyOil(m3/d)', 'CalDlyOil(m3/d)', 'CumPrdOil(m3)','MonthlyGas(E3m3)',
                'AvgDlyGas(E3m3/d)', 'CalDlyGas(E3m3/d)', 'CumPrdGas(E3m3)', 'MonthlyWater(m3)', 'AvgDlyWtr(m3/d)',
                'CalDlyWtr(m3/d)', 'CumPrdWtr(m3)', 'MonthlyFluid(m3)', 'AvgDlyFluid(m3/d)', 'CalDlyFluid(m3/d)',
                'CumPrdFluid(m3)', 'MonInjGas(E3m3)', 'AvgInjGas(E3m3/d)', 'CalInjGas(E3m3/d)', 'CumInjGas(E3m3)',
                'MonInjWtr(m3)', 'AvgInjWtr(m3/d)', 'CalInjWtr(m3/d)', 'CumInjWtr(m3)', 'MonInjSlv(E3m3)',
                'AvgInjSlv(E3m3/d)', 'CalInjSlv(E3m3/d)', 'CumInjSlv(E3m3)', 'InjHours(hr)', 'PrdHours(hr)',
                'Inj/PrdHours(hr)', 'WCT(%)', 'OCT(%)', 'GOR(m3/m3)', 'WGR(m3/E3m3)', 'WOR(m3/m3)', 'NbrofWells',
                'MonInjSteam(m3)', 'AvgInjSteam(m3/d)', 'CalInjSteam(m3/d)', 'CumInjSteam(m3)']

# Create a new AccumapCsvPreprocessor with the input and output directories
accumapCsvProcessor = AccumapCsvProcessor.AccumapCsvProcessor(input_directory, output_directory,
                                                              feature_list=feature_list, debug_printing=True,
                                                              steady_state_only=True, show_plots=False, ncg_only=False,
                                                              sigma=3)

# Run the preprocessing for the set directories
accumapCsvProcessor.preprocessSetDirectory()
