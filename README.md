# Enhancing Breast Cancer Diagnosis with Combined MRI and Clinical Data Analysis Using Convolutional Neural Networks

## Introduction

This project aims to enhance the diagnosis of breast cancer by combining MRI and clinical data into a single Convolutional Neural Network (CNN). This project is based the Dynamic Affine Feature Map Transform ([DAFT](https://github.com/ai-med/DAFT)) model of Pölsterl et al.. The DAFT model is a deep neural network that combines MRI and clinical data to predict the onset of Alzheimer's disease, and here we adapt the DAFT model to predict the breast cancer hormone receptor status and tumor size. 

## Data

We have used the [Duke-Breast-Cancer-MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) data set, containing MRI scans and tabular data for 922 patients. Feature extraction was used, along with preprocessing of the MRI scan before training the model on an 80/20 train/validation split. Due to the massive size of this dataset, we have not included it in this repository or in our final submission, however it can be downloaded at the link above. 

## Usage

### Preprocessing

To run all scripts in the project, the following libraries are required:

```bash
pip install numpy pandas pydicom torch scikit-learn monai matplotlib
```

The MRI data was preprocessed using the MRI_Data_preprocessing.py script, which was run on a high performance computing cluster using the CPSC571Dataprocess.slurm script. The preprocessing script reads the MRI data from the RawDataset/MRIs directory, pads the images to the same size and saves them as tensors. 

The preprocessing script takes two command line arguments: the MRI data directory and the output directory. 

```bash
python MRI_Data_preprocessing.py [MRI_dir] [output_dir]

# Example
python MRI_Data_preprocessing.py "./RawDataset/MRIs/" "./ProcessedDataset/MRIs/"
```

Feature extraction was done using feature_extraction.ipynb python notebook, with the three outputs shown in the RawDataset directory.

### Training

The model was trained and evaluated using the DAFTTrain.py script. The model architecture is defined in the DAFT directory, with some hyperparameters defined in the DAFTTrain.py script. The DAFT model is used thanks to its ability to combine image and tabular data into the same convolutional neural network framework. 

The script takes two command line arguments: the MRI data directory and the tabular data directory. 

```bash
python DAFTTrain.py [MRI_dir] [tabular_dir]

# Example
python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/TumorSize_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_tumor_size.txt
python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/ER_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_ER.txt
python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/HER2_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_HER2.txt
```

Given the massive size of the dataset, the training process was done on a high performance computing cluster using the CPSC571BC.slurm script, using a100 or v100 GPUs depending on availability.

### Evaluation

At the end of trainig, two csv files are generated in the a100Outputs directory: One for the training records and one for the validation records. Theses files have the predictions and true values for the samples tested.

Model proformance is calculated and graphed from these files in the preformaceMetrics.ipynb notebook.

