# MSK_ML_gamma
#This file contains the code corresponding to the article "Machine Learning Workflows for Motion Capture-driven Biomechanical Modelling"
#This work rigorously implements and critically evaluate the performance of various ML models for mapping optical motion capture inputs to musculoskeletal outputs.

#Code description
#pytorch_utilities.py -- contains function for various models (Linear, Neural Network, RNN, LSTM, GRU,....) and generate a file with hyperparameters choices explored in this work. 
#read_in_out.py -- contains classes to read input and output data and some other classes for handling the final trained models. 
#pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results, plot outputs

#HPC clusters (JED cluster, SCITAS, EPFL, https://www.epfl.ch/research/facilities/scitas/jed/)  were used to run the complete cross-validation runs which are not feasible on a normap laptop. However, few instances, can be easily done using the function "specific" and "specific_CV" in pytorch.py, respectively.

#Final analysis including all the plots used in main article and supplemental information are generated in main.ipynb. 

#final trained NN model #Files are heavy to upload, can be downloaded from here: https://www.dropbox.com/sh/svuqdy597d6pg60/AACiWr-kVx_W0bU0AD3HWPNha?dl=0
#Test data for reproducing the results are provided here,
Input data --  https://www.dropbox.com/sh/8isp6yl29np6ngo/AAAqRWIc8lTOtYieehUSEQN1a?dl=0
Output data -- https://www.dropbox.com/sh/7h1oncpyru9vupl/AACm0YPmlmdu2rlabUpynCW4a?dl=0


#Test data for reproducing the results are provided here. The full training dataset can be obtained on request to V.HarthikoteNagaraja@salford.ac.uk

The ML techniques and data analysis were done using Python 3.12 (\href{https://www.python.org/}{https://www.python.org/}) and Keras 3.0 (\href{https://keras.io/}{https://keras.io/}). Add details on where the computing was done (technical details of SCITAS, EPFL) % RS to update these details and amend the sentence.

All the libraries and depedicies are in environment.yml and the conda environment can be recreated by running the following command:
conda env create -f environment.yml

