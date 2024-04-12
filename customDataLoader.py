import pickle
import os
from pathlib import Path
import sys
import torch
# import torch.nn.functional as F
from monai.data import Dataset, DataLoader
import numpy as np
import monai
from monai.data import Dataset, DataLoader
import pandas as pd
import pydicom

# label_column = "Tumor Progression"
label_columns = ["Tumor Progression_1.0","Tumor Progression_2.0","Tumor Progression_3.0","Tumor Progression_4.0"]

def resize(img):
    print(f"img shape: {img.shape}")
    # desired_shape = (170, 512, 512)
    desired_shape = (210, 512, 512)
    
    pad_width = []
    for i in range(len(desired_shape)):
        width = max(desired_shape[i] - img.shape[i], 0)
        pad_width.append((int(width // 2), int(width - (width // 2))))

    padded_image = np.pad(img, pad_width, mode='constant')
    padded_image = torch.tensor(padded_image)
    print(f"padded image shape: {padded_image.shape}")
    return padded_image
    

def load(file_path):
    """
    Loads and unpickles data from the specified file path.
    
    Parameters:
        file_path (str): Path to the file to load.
        
    Returns:
        tuple: A tuple containing the loaded tensor, tabular data, and additional data.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    # Unpickle the file
    with open(file_path, 'rb') as f:
        tensor, tabular_data, _ = pickle.load(f)
        return tensor, tabular_data, _

class CombinedDataset(Dataset):    
    """
    Custom dataset class for loading and preprocessing medical imaging data
    alongside tabular information from a specified directory.
    
    Attributes:
        data_dir (str): Directory containing the data files.
        tabular_dir (str): Directory containing the tabular data files.
        tensor_filenames (list): List of filenames for the tensor data.
        tabular_filenames (list): List of filenames for the tabular data.
        tabular_data (DataFrame): DataFrame containing the tabular data.
    """

    def __init__(self, data_dir, tabular_dir):
        self.data_dir = data_dir
        self.tabular_dir = tabular_dir
        self.tensor_filenames = os.listdir(data_dir)
        print("Tensor filenames: ", self.tensor_filenames)
        # load in the tabular data in pandas
        self.tabular_data = pd.read_csv(tabular_dir)
        # self.tabular_data = pd.read_excel(tabular_dir)
        # print(self.tabular_data.head())
        print(f"row 0: {self.tabular_data.iloc[0]}")

    def __len__(self):
        """Returns the total number of files in the dataset."""
        return len(self.tensor_filenames)

    # def __getitem__(self, index):
    #     """
    #     Retrieves a tensor and its corresponding label from the dataset at the specified index.
        
    #     Parameters:
    #         index (int): Index of the item to retrieve.
        
    #     Returns:
    #         tuple: A tuple containing the preprocessed tensor, its label, and is tabular data
    #     """
    #     tensor_filename = self.tensor_filenames[index]
    #     tensor_path = os.path.join(self.data_dir, tensor_filename)
    #     tensor, tabular_row, _ = load(tensor_path) 

    #     tabular_row = tabular_row[1:]  

    #     tensor = torch.tensor(tensor)
    #     if tensor.ndim == 3:
    #         tensor = tensor.unsqueeze(0)

    #     try:
    #         tensor = F.avg_pool3d(tensor, self.downsample_2)
    #     except Exception as e:
    #         print(f"Error during pooling operation: {e}")
    #         print(f"Tensor shape before pooling: {tensor.shape}")
    #         print(f"Downsample size: {self.downsample_2}")
    #         return None

    #     label = torch.tensor(float(tabular_row[22])) if self.bone_type == 'r' else torch.tensor(float(tabular_row[23]))
    #     tabular_row = tabular_row[:22] + tabular_row[24:]
    #     if self.transform is not None:
    #         tensor = self.transform(tensor)

    #     if isinstance(tabular_row, list):
    #         for i in tabular_row:
    #             try:
    #                 float(i) 
    #             except ValueError:
    #                 print(f"Error converting to float: {i} in row {tabular_row}")
    #         tabular_row = [float(i) if i != '' else 0.0 for i in tabular_row]

    #     if self.tabular_transform is not None:
    #         tabular_data = self.tabular_transform(tabular_row)
    #     else:
    #         tabular_data = torch.tensor(tabular_row)
    #     print("tensor shape is: ", tensor.shape)
    #     print("Label is:  ", label)
    #     print("Tabular row is: ", tabular_data)
    #     print("_ is: ", _)
    #     return tensor, label, tabular_data
    
    def __getitem__(self, index):
        print("checkpoint 1")
        tensor_filename = self.tensor_filenames[index]
        image_folder = os.path.join(self.data_dir, tensor_filename)

        # maxXYSize = 512
        # maxZSize = 160
        
        # Load the DICOM file
        # dicom = pydicom.dcmread(tensor_path)
        dicom_images = []
        for root, _, filenames in os.walk(image_folder):
            for filename in filenames:
                dcm_path = Path(root, filename)
                if dcm_path.suffix == ".dcm":
                    try:
                        dicom = pydicom.dcmread(dcm_path, force=True)
                        dicom_images.append(dicom.pixel_array.astype(np.float32))
                        # dicom_images.append(dicom.pixel_array)
                    except IOError as e:
                        print(f"Can't import {dcm_path.stem}")

        dicom_images = resize(np.array(dicom_images))

        # Stack images into a 3D tensor
        tensor = np.stack(dicom_images, axis=-1)
        # tensor = dicom.pixel_array

        # Normalize the tensor to the range 0-1
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

        # Add a new dimension to the tensor
        tensor = torch.unsqueeze(torch.tensor(tensor), 0)
        # print(f"tensor type: {type(tensor)}")
        print(f"tensor shape: {tensor.shape}")
        # tensor = tensor.half()
        # print(f"tensor type: {type(tensor)}")
        # print(f"tensor shape: {tensor.shape}")

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
        # tabular_data = tabular_data.drop("Patient ID")
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

