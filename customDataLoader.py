# This file is a custom data loader built off of the MONAI Dataset class. It is used
# to load the MRI tensors and tabular data for the DAFT model. The MRI tensors are loaded
# from disk and the tabular data is loaded from a CSV file. 


import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import monai
from monai.data import Dataset
import pydicom

# label_columns = ["Tumor Progression_1.0","Tumor Progression_2.0","Tumor Progression_3.0","Tumor Progression_4.0"]
label_columns = ["Tumor Progression_0","Tumor Progression_1"]

def resize(img):
    print(f"img shape: {img.shape}")
    desired_shape = (256, 512, 512)
    
    pad_width = []
    for i in range(len(desired_shape)):
        width = max(desired_shape[i] - img.shape[i], 0)
        pad_width.append((int(width // 2), int(width - (width // 2))))

    padded_image = np.pad(img, pad_width, mode='constant')
    padded_image = torch.tensor(padded_image)
    print(f"padded image shape: {padded_image.shape}")
    return padded_image
    

class CombinedDataset(Dataset):    
    """
    Custom dataset class for loading the MRI tensors and tabular data.
    
    Attributes:
        data_dir (str): Directory containing the data files.
        tabular_dir (str): Directory containing the tabular data files.
        dcm_patient_filenames (list): List of filenames in the data directory.
        tabular_data (pd.DataFrame): DataFrame containing the tabular data.
        patient_MRI_tensors (list): List of MRI tensors for each patient.
    """

    def __init__(self, data_dir, tabular_dir, workers = 1):
        self.data_dir = data_dir
        self.tabular_dir = tabular_dir
        self.dcm_patient_filenames = os.listdir(data_dir)
        
        # sort the filenames to make sure patient IDs match list index
        self.dcm_patient_filenames.sort()
        print(f"workers: {workers}")
        print("dcm_patient_filenames: ", self.dcm_patient_filenames)

        # Create a multiprocessing Pool
        with Pool(workers) as p:
            self.patient_MRI_tensors = p.map(self.load_tensor, self.dcm_patient_filenames)

        # load in the tabular data in pandas
        self.tabular_data = pd.read_csv(tabular_dir)
        print(f"row 0: {self.tabular_data.iloc[0]}")


    def load_tensor(self, tensor_filename):
        return torch.load(os.path.join(self.data_dir, tensor_filename))


    def __len__(self):
        """Returns the total number of files in the dataset."""
        return len(self.dcm_patient_filenames)
    
    # This function is called to get the data for a given index
    def __getitem__(self, index):
        # Load the tensor data
        tensor = self.patient_MRI_tensors[index]

        # Get the corresponding tabular data
        tabular_row = self.tabular_data.iloc[index]

        # Get the label
        label = torch.tensor([float(tabular_row[label_column]) for label_column in label_columns])

        # Remove the label from the tabular data
        tabular_data = tabular_row.drop(label_columns)
        tabular_data = tabular_data.drop("Patient ID")

        # Convert the tabular data to a tensor
        tabular_data = torch.tensor(tabular_data.values.astype(np.float16))

        # Return the MRI tensor, target label, and tabular data for patient
        return tensor, label, tabular_data

