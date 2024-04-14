
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
    # desired_shape = (170, 512, 512)
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
    Custom dataset class for loading and preprocessing medical imaging data
    alongside tabular information from a specified directory.
    
    Attributes:
        data_dir (str): Directory containing the data files.
        tabular_dir (str): Directory containing the tabular data files.
        dcm_patient_filenames (list): List of filenames for the tensor data.
        tabular_filenames (list): List of filenames for the tabular data.
        tabular_data (DataFrame): DataFrame containing the tabular data.
    """

    def __init__(self, data_dir, tabular_dir, workers = 1):
        self.data_dir = data_dir
        self.tabular_dir = tabular_dir
        self.dcm_patient_filenames = os.listdir(data_dir)
        #sort the filenames
        self.dcm_patient_filenames.sort()
        print(f"workers: {workers}")
        print("dcm_patient_filenames: ", self.dcm_patient_filenames)

        # load in the tensors from disk
        '''
        self.patient_MRI_tensors = []
        for tensor_filename in self.dcm_patient_filenames:
            tensor = torch.load(os.path.join(data_dir, tensor_filename))
            self.patient_MRI_tensors.append(tensor)
        '''

        # Create a multiprocessing Pool
        with Pool(workers) as p:
            self.patient_MRI_tensors = p.map(self.load_tensor, self.dcm_patient_filenames)

        '''
        for tensor_filename in self.dcm_patient_filenames:
            image_folder = os.path.join(self.data_dir, tensor_filename)
            patient_imgs = []
            for root, _, filenames in os.walk(image_folder):
                for filename in filenames:
                    dcm_path = Path(root, filename)
                    if dcm_path.suffix == ".dcm":
                        try:
                            dicom = pydicom.dcmread(dcm_path, force=True)
                            patient_imgs.append(dicom.pixel_array.astype(np.float32))
                        except IOError as e:
                            print(f"Can't import {dcm_path.stem}")
            
            patient_imgs = resize(np.array(patient_imgs))
            tensor = np.stack(patient_imgs, axis=-1)
            # Normalize the tensor to the range 0-1
            tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
            tensor = torch.unsqueeze(torch.tensor(tensor), 0)
            self.patient_MRI_tensors.append(tensor)
        '''

        # load in the tabular data in pandas
        self.tabular_data = pd.read_csv(tabular_dir)
        # self.tabular_data = pd.read_excel(tabular_dir)
        # print(self.tabular_data.head())
        print(f"row 0: {self.tabular_data.iloc[0]}")


    def load_tensor(self, tensor_filename):
        return torch.load(os.path.join(self.data_dir, tensor_filename))


    def __len__(self):
        """Returns the total number of files in the dataset."""
        return len(self.dcm_patient_filenames)
    
    def __getitem__(self, index):
        print("checkpoint 1")
        # tensor_filename = self.dcm_patient_filenames[index]
        # image_folder = os.path.join(self.data_dir, tensor_filename)

        # # Load the DICOM file
        # # dicom = pydicom.dcmread(tensor_path)
        # patient_MRI_tensors = []
        # for root, _, filenames in os.walk(image_folder):
        #     for filename in filenames:
        #         dcm_path = Path(root, filename)
        #         if dcm_path.suffix == ".dcm":
        #             try:
        #                 dicom = pydicom.dcmread(dcm_path, force=True)
        #                 patient_MRI_tensors.append(dicom.pixel_array.astype(np.float32))
        #                 # patient_MRI_tensors.append(dicom.pixel_array)
        #             except IOError as e:
        #                 print(f"Can't import {dcm_path.stem}")

        # patient_MRI_tensors = resize(np.array(patient_MRI_tensors))

        # # Stack images into a 3D tensor
        # tensor = np.stack(patient_MRI_tensors, axis=-1)
        # # tensor = dicom.pixel_array

        # # Normalize the tensor to the range 0-1
        # tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

        # # Add a new dimension to the tensor
        # tensor = torch.unsqueeze(torch.tensor(tensor), 0)
        # # print(f"tensor type: {type(tensor)}")
        # print(f"tensor shape: {tensor.shape}")
        # tensor = tensor.half()
        # print(f"tensor type: {type(tensor)}")
        # print(f"tensor shape: {tensor.shape}")

        # Load the tensor data
        tensor = self.patient_MRI_tensors[index]

        # Get the corresponding tabular data
        tabular_row = self.tabular_data.iloc[index]
        # print("Tabular row is: ", tabular_row)

        # Get the label
        # label = torch.tensor(float(tabular_row[label_column]))
        label = torch.tensor([float(tabular_row[label_column]) for label_column in label_columns])
        print("Label is: ", label)
        # label = torch.tensor(1)

        # Remove the label from the tabular data
        # tabular_data = tabular_row.drop(label_column)
        tabular_data = tabular_row.drop(label_columns)
        print(f"feature count: {len(tabular_data)}")
        tabular_data = tabular_data.drop("Patient ID")
        # print("Tabular data is: ", tabular_data)
        # tabular_data = tabular_row
        print("checkpoint 3")

        # Convert the tabular data to a tensor
        # tabular_data = torch.tensor(tabular_data.values.astype(np.float64))
        tabular_data = torch.tensor(tabular_data.values.astype(np.float16))
        # tabular_data = torch.tensor(float(tabular_data.values))
        # tabular_data = torch.tensor(tabular_data.values).float()

        # convert the tabular_data to list 
        # tabular_data = tabular_data.tolist()
        # tabular_data = [float(i) for i in tabular_data]


        return tensor, label, tabular_data

