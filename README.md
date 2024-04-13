# Enhancing Breast Cancer Diagnosis with Combined MRI and Clinical Data Analysis Using Convolutional Neural Networks

## Introduction

This project aims to enhance the diagnosis of breast cancer by combining MRI and clinical data into a single Convolutional Neural Network (CNN). This project is based the Dynamic Affine Feature Map Transform ([DAFT](https://github.com/ai-med/DAFT)) model of Pölsterl et al.. The DAFT model is a deep neural network that combines MRI and clinical data to predict the onset of Alzheimer's disease, and here we adapt the DAFT model to predict the breast cancer hormone receptor status and tumor size. 

## Data

We have used the [Duke-Breast-Cancer-MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) data set, containing MRI scans and tabular data for 922 patients. Feature extraction was used, along with preprocessing of the MRI scan before training the model on an 80/20 train/validation split.

## Usage

### Preprocessing

The MRI data was preprocessed using the MRI_Data_preprocessing.py script, which was run on a high performance computing cluster using the CPSC571Dataprocess.slurm script. 

Feature extraction was done using **** python notebook. 

### Training

The model was trained and evaluated using the DAFTTrain.py script. The script takes two command line arguments: the MRI data directory and the tabular data directory. 

```bash
python DAFTTrain.py [MRI_dir] [tabular_dir]
```

Given the massive size of the dataset, the training process was done on a high performance computing cluster using the CPSC571BC.slurm script, using a100 or v100 GPUs depending on availability.

### Evaluation

At the end of trainig, two csv files are generated in the a100Outputs directory: One for the training records and one for the validation records. Theses files have the predictions and true values for the samples tested.

Model proformance is calculated and graphed from these files in the preformaceMetrics.ipynb notebook.

