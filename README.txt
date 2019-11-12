
Solution to the EY Data Challenge in which @catyselman and @mradazoury finished Top 8 in Spain and < 10% worldwide.

The data is not included in this repo please contact us if needed.

Please keep in mind that the work was fairly split between @catyselman and @mradazoury and that the contributions graph do not reflect the distribution of the effort.

Model development is spread out in 4 Python scripts and 1 Python notebook.

All developed in Python 3.7.

Libraries used:
	-sklearn
	-pandas
	-numpy
	-keras

Details of files:

1) utils.py: Includes all data processing and helper functions utilized throughout rest of scripts/notebooks.

2) data_prep.py: Inputs train and test raw data and pre-processes velue types and creates initial new features for each trajectory.
	Inputs: data_test.csv, data_train.csv
	Outputs: processed.csv

3) aggregation.py: Inputs processed csv and aggregates per device to create additional features for each trajectory based on previous trajectories on same day for this device.
	Inputs: processed.csv
	Outputs: processed_agged.csv

4) moreProcessing.py: Inputs processed data with aggregated features and adds additional features that require distributed computation.
	Inputs: processed_agged.csv
	Outputs: final_processed.csv

5) FinalModel_DenoisingAutoencorder_PCA_XGBoost.ipynb: Inputs final processed file and builds final. Generates predictions submitted into EY platform.
	Inputs: final_processed.csv
	Outputs: Submission CSV file
