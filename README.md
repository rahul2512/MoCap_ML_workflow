# MoCap_ML_workflow


## Model Details
This file contains the code corresponding to the article "Machine Learning Workflows for Motion Capture-driven Biomechanical Modelling". 
This work rigorously implements and critically evaluate the performance of various popular ML models for mapping optical motion capture inputs to biomechanical musculoskeletal outputs.

Finally, this research provides first-ever checklist of best practices for ML research in biomechanical modelling, the LearnABLE (Learning mAchines for BiomechanicaL modElling) checklist.

### Model Description
- **Developed by:** {{Vikranth Harthikote Nagaraja, Rahul Sharma, Abhishek Dasgupta and Challenger Mishra}}
- **Funded by [optional]:** {{This project made use of data originally collected for a previous study funded by the _Research Councils UK (RCUK) Digital Economy Programme grant number EP/G036861/1_ (Oxford Centre for Doctoral Training in Healthcare Innovation) and the _Wellcome Trust Affordable Healthcare in India Award (Grant number 103383/B/13/Z)_. CM's Fellowship is supported through the Accelerate Program for Scientific Discovery at the Computer Laboratory, University of Cambridge. We also acknowledge funding from the _Swiss National Science Foundation_ (grant number 200020 182184, supporting RS and Article Processing Charges) and the computational resources provided by _SCITAS, EPFL_.}}
- **License:** {{ license | default("[More Information Needed]", true)}}

### Model Sources [optional]

- **Repository:** https://github.com/rahul2512/MoCap_ML_workflow
- **Paper [optional]:** {{link to paper}}
- **Demo [optional]:** https://github.com/rahul2512/MoCap_ML_workflow/blob/main/main.ipynb
  
## Uses
Research purpose only -- add more

## Bias, Risks, and Limitations


### Recommendations

LearnAble

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}



This file contains the code corresponding to the article "Machine Learning Workflows for Motion Capture-driven Biomechanical Modelling"
This work rigorously implements and critically evaluate the performance of various popular ML models for mapping optical motion capture inputs to musculoskeletal outputs.
Finally, this research provides first-ever checklist of best practices for ML research in biomechanical modelling, the LearnABLE (Learning mAchines for BiomechanicaL modElling) checklist.

#Code description
#pytorch_utilities.py -- contains function for various models (Linear, Neural Network, RNN, LSTM, GRU,....) and generate a file with hyperparameters choices explored in this work. 
#read_in_out.py -- contains classes to read input and output data and some other classes for handling the final trained models. 
#pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results, plot outputs

#HPC clusters (JED cluster, SCITAS, EPFL, https://www.epfl.ch/research/facilities/scitas/jed/)  were used to run the complete cross-validation runs which are not feasible on a normap laptop. However, few instances, can be easily done using the function "specific" and "specific_CV" in pytorch.py, respectively.

#Final analysis including all the plots used in main article and supplemental information are generated in main.ipynb. 

Note that only test data is provided with the code, the full training dataset can be obtained on request to V.HarthikoteNagaraja@salford.ac.uk

The ML techniques and data analysis were done using Python 3.12 (\href{https://www.python.org/}{https://www.python.org/}) and Keras 3.0 (\href{https://keras.io/}{https://keras.io/}). Add details on where the computing was done (technical details of SCITAS, EPFL) % RS to update these details and amend the sentence.

All the libraries and depedicies are in environment.yml and the conda environment can be recreated by running the following command:
conda env create -f environment.yml

