# LESST
Learning with Subset Stacking for Timeseries is a repository that proposes a new forecasting method using Machine-Learning, Cross-Learning, Stacking and Clustering methods.
The is trained using the M4-competition dataset. The method consists of selecting subsets of the timeseries using K-Means clustering on the Timeseries Features.
For each cluster Local Models are trained using these subsets. These Local Model predictions are weighted using the weights defined by the distance to cluster centers.
The Global Model combines these weighted Local Model forecasts into one final prediction.

In this section we describe the details on the code, what files contain what and how to run the results:
Follow these steps if running the code for the first time:

* Install R
* Install all required python libraries in the requirements.txt into the python environment
* Change the path in $os.environ["R\_HOME"] = "E:/documents/work/mini/envs/work/lib/R"$ in the preprocessing.py and tsforecast.py files to the path of your python environment
* Uncomment the R library installations in the preprocessing file
* Run the prepare\_allm4data function in the preprocessing.py file
* Run the prepare\_m4tsfeatures function in the preprocessing.py file for calculating all the timeseries features of all dataset and saving them
* Comment the R library installations in the preprocessing file again

For obtaining the results run theses files:

* ResultsMain.py: main LESST results and benchmark performance
* ResultsLocal.py: performance Local Models
* ResultsGlobal.py: performance Global Model using weighted sum
* ResultsRolling.py: performance Global Model with all training data
* ResultsWeights.py: figure of Global Model coefficients
* formatresults.py: makes excel files out of the dictionaries from the main LESST results

For the dataset information run:
* DataInfo.py: calculates some information on the datasets

Here are details on what the other files contain:

* LESST.py: contains the LESST model
* Models.py: contains the Local and Global Model
* benchmark.py: contains functions for the benchmark method and the performance measures
* clustering.py: contains the timeseries feature clustering method
* seasonality: contains functions for deseasonalizing the data
* tsforecasts.py: contains the ThetaF model and other R forecasting models
* data\_prep.py: contains functions for preparing input and target data
* preprocessing: contains functions for calculating timeseries features and reading m4 data
