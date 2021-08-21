# Predictive Analysis of Flight Times

This project is a data mining approach on prediction of the aircraft movement times, namely, taxi-in, taxi-out, airtime and block time. 
These predicted times are expected to produce more accurate data and are to be used in an OCC (operational control center) decision support system to replace the use of scheduled or OCC-estimated times. 

The models were trained on data from a European Airline air traffic activity from 2013 to 2018, along with scraped weather data retrieved from Iowa Environmental Mesonet. The data was analysed and refactored, along with some feature engineering. 

After extensive experiments, the most successful models were built, making use of a stack of linear estimators with gradient boosting as meta-estimator.

Main Author: Afonso Manuel Maia Lopes Salgado de Sousa (afonsousa2806@gmail.com)

Supervisors: Ana Paula Rocha (arocha@fe.up.pt) and António J. M. Castro (frisky.antonio@gmail.com)

## Application Use Case
One wants to assess the impact of one or more disruptions, on the airline operational plan, for the next X number of days. That is, one wants to know how many flights, for how long and wich flights will be delayed, how many crew members, for how long and individual crew names will be delayed and have their schedule disrupted and, finally, how many, for how long and which passengers will miss connections. 

To do such assessment, one needs to know the expected block times for each flight and/or the flight time plus taxi time out and taxi time in. The schedule block times, typically, do not represent the "real" block times and, as such, should not be used to do this assessment. Additionally, the flight plan for each flight is available only closer to the flight schedule time of departure and, as such, cannot be used for the assessment as well.

The goal of this project is to provide a machine learning model, that also considers weather information, that can predict the block times (or flight times and taxi times) long in advance, so that the impact assessement on the operational plan can be more accurate. 

## Project File Structure:
The root of the project comprises four folders:
* data - for the raw and processed datasets
* images - for any system-generated image (mainly plots)
* models - for storing of the built models
* src - with the source code

## Data:
This folder contains two distinct folders, ```raw``` and ```processed```. The ```raw``` folder is here to hold the raw data from the airline and the chunked extracted weather data. The ```processed``` folder contains the built datasets from code execution.

## Source Code:
This folder contains three main folders, namely, ```preparation```, ```processing```, and ```modelling```, and a couple of files with global-supporting methods.

### Preparation

To build the concatenated dataset of historical flight and weather data, one can first run the script [dataset_builder](src/preparation/dataset_builder.py). This will build chunks of 5000 observations with the flight data (already available) and corresponding weather data (scraped from [this website](https://mesonet.agron.iastate.edu/request/download.phtml)).
Then, the chunks can be concatenated and stored as a whole file running [file_merging](src/preparation/file_merging.py). This will build the ```full_info.csv```.

### Processing

#### First data enhancement
Now, to process ```full_info.csv```, one can run [eda](src/processing/eda.py). This file makes some adjustments to the data along with various plots for data understanding. This will build the ```basic_eda.csv```.

#### Feature Engineering
In this step, some features are created, others deleted and some refactored. This can be achieved running [feature_engineering](src/processing/feature_engineering.py). Additionally, there are some tasks related to data compression and feature selection that can be run.

#### Imputation
For imputation, to test the autoencoders, run [this file](src/processing/imputation/autoencoder/one_hot_main.py) and change the autoencoder import (```python from src.processing.imputation.autoencoder.<autoencoder or masked_ae> import Autoencoder```) to the ```standard autoencoder``` or ```masked autoencoder```. For the VAE, [this file](src/processing/imputation/vae/vae_keras/one_hot_main.py) can be ran to assess the imputation performance. Other versions of VAEs are present in [this folder](src/processing/imputation/vae).

### Modelling

The json file [settings](src/settings.json) should be used to define the main parametres of the modelling process. The available parametre sets are as follows:
* imputed - [~true~, false]
* target - ['air_time', 'taxi_out', 'taxi_in', 'actual_block_time']
* cat_encoding - _currently_ LeaveOneOut hard coded
* fleet_type - ['whole', 'NB', 'WB']

* model_name - ['ridge', 'lasso', 'elastic', 'xgboost', 'gradient_boosting', 'random_forest', 'stacking', 'ffnn'] (note 'stacking' and 'ffnn' do not have hyperparametre tuning step)

Firstly, run [prepare](src/modeling/prepare.py) to save the train and test sets for the specs defined in [settings](src/settings.json). One can run this file also to generate validation sets or training data with fixed distributions of numerical variables.

Secondly, run [modelling](src/modeling/training/modelling.py) where one can tune hyperparameters, cross-validate models or fit them to the training set, using the specs defined in [settings](src/settings.json). Also, cross-validation with Neural Network requires n_jobs=1 because of an existing bug.

Finally, run [validation](src/modeling/validation.py) to assess the performance of an algorithm (defined in [settings](src/settings.json)).

### Deployment
It turns out that it is very hard to integrate sklearn machine learning models into an existing non-Python-based solution. Unfortunately, [this package](https://github.com/jpmml/sklearn2pmml), that would otherwise solve the stated problem, is limited for the moment and does not yet support some of the packages used (like [category encoder](https://contrib.scikit-learn.org/categorical-encoding/) or [mlxtend](http://rasbt.github.io/mlxtend/)). Keep checking if this package already supports the mentioned packages so a seamless usage of the models in a Java environment can be achieved.

A workaround using sklearn is to use H2O.ai instead. [This website](https://www.quora.com/Why-would-one-use-H2O-ai-over-scikit-learn-machine-learning-tool#) shows an interesting discussing on the topic of Scikit-learn vs H2O.ai and views H2O as a good alternative to sklearn when developing specifically for Java, although this package has some limitations.

For the time being, the solution is to serve predictions with an endpoint. In the root, there is a simple flask server that can be run by executing [this file](flask_server.py), serving endpoints in the port 9000. This is a development server with basic endpoints to serve each of the predictive tasks. Remote calls for testing are present at the end of the file.

## Suggestions:
For EDA and Feature Engineering, I suggest the scripts be run in a Jupyter Notebook, for faster reruns of the code. Alternatively, use the annotation ```# %%``` to delimit the cells in a python script and make use of the Python Interactive Interpreter in Visual Studio Code. This requires the installation of Jupyter.

All imports make use of absolute paths to root folder, so the execution of any script requires this to be the current directory. To do so in a code environment, try using the following:
```python
import os
os.chdir(<path_to_folder_root>)
```

## Installation:
There is no main file to be executed. The whole pipeline may take several hours/days to perform from start to finish, so iterative file-by-file execution is advised. The weather data extraction is particularly expensive due to the large volume of data and consequently a large number of download requests.

The [requirements](requirements.txt) file contains the packages required for the full run of the project. One can install these dependencies running (its recommended for this to be done in a new clean environment [conda perhaps]):
```python
pip install --user --requirement requirements.txt
```
