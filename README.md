# MoCap_ML_workflow


## Model Details
This file contains the code corresponding to the article "Machine Learning Workflows for Motion Capture-driven Biomechanical Modelling". 
This work rigorously implements and critically evaluate the performance of various popular ML models for mapping optical motion capture inputs to biomechanical musculoskeletal outputs.

Finally, this research provides first-ever checklist of best practices for ML research in biomechanical modelling, the LearnABLE (Learning mAchines for BiomechanicaL modElling) checklist.

### Model Description
- **Developed by:** {{Vikranth Harthikote Nagaraja, Rahul Sharma, Abhishek Dasgupta and Challenger Mishra}}
- **Funded by [optional]:** {{This project made use of data originally collected for a previous study funded by the _Research Councils UK (RCUK) Digital Economy Programme grant number EP/G036861/1_ (Oxford Centre for Doctoral Training in Healthcare Innovation) and the _Wellcome Trust Affordable Healthcare in India Award (Grant number 103383/B/13/Z)_. CM's Fellowship is supported through the Accelerate Program for Scientific Discovery at the Computer Laboratory, University of Cambridge. We also acknowledge funding from the _Swiss National Science Foundation_ (grant number 200020 182184, supporting RS and Article Processing Charges) and the computational resources provided by _SCITAS, EPFL_.}}
- **License:** {{ license | default("[More Information Needed]", true)}}

### Model Sources
- **Repository:** https://github.com/rahul2512/MoCap_ML_workflow
- **Paper:** {{link to paper}}
- **Demo:** https://github.com/rahul2512/MoCap_ML_workflow/blob/main/main.ipynb
  
## Uses
Research purpose only. The primary purpose of this work was systematic implement and critically evaluate the relevant ML techniques in MoCap driven MSK modelling. 

### Recommendations
LearnAble checklist 

## How to Get Started with the Model
pytorch_utilities.py -- contains function for various models (Linear, Neural Network, RNN, LSTM, GRU,....) and generate a file with hyperparameters choices explored in this work. 
read_in_out.py -- contains classes to read input and output data and some other classes for handling the final trained models. 
pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results, plot outputs

## Training Details
The goal is to approximate MSK model by mapping MoCap inputs to biomechanical MSK outputs.
The ML techniques and data analysis were done using Python 3.12 (\href{https://www.python.org/}{https://www.python.org/}) and Keras 3.0 (\href{https://keras.io/}{https://keras.io/}). Add details on where the computing was done (technical details of SCITAS, EPFL) % RS to update these details and amend the sentence.
All the libraries and depedicies are in environment.yml and the conda environment can be recreated by running the following command:
conda env create -f environment.yml

### Training Data
Note that only test data is provided with the code, the full training dataset can be obtained on request to V.HarthikoteNagaraja@salford.ac.uk.

#### Preprocessing
The preprocessing steps are detailed in the paper. 

#### Training Hyperparameters
- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation
#### Testing Data
Testing data is OMC-driven MSK model biomechanical outputs. Details are in Methods section in the paper.

#### Metrics
Primary error metric: RMSE and NRMSE and Secondary error metric: pearson correlation

## Model Examination
The various aspects of ML implementation are critically evaluated for various aspects including robustness to added noise, transferability, and explainability.
Final analysis including all the plots used in main article and supplemental information are generated in main.ipynb. 

## Environmental Impact
HPC clusters (JED cluster, SCITAS, EPFL, https://www.epfl.ch/research/facilities/scitas/jed/)  were used to run the complete cross-validation runs which are not feasible on a normap laptop. However, few instances, can be easily done using the function "specific" and "specific_CV" in pytorch.py, respectively. 
The carbon footprint calculation is performed following the guidelines by SCITAS, EPFL (https://scitas-doc.epfl.ch/user-guide/co2-estimation/). 
Specifically, we used the ‘JED’ cluster, which produces an emission of 1.02 grams of CO2 equivalent for every CPU core hour used. JED has 2.4 GHz Intel(R) Xeon(R) Platinum 8360Y processors.

### Compute Infrastructure
Specifically, we used the ‘JED’ cluster, which has 2.4 GHz Intel(R) Xeon(R) Platinum 8360Y processors.

