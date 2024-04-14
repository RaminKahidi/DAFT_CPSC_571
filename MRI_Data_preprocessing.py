import multiprocessing
import os
from pathlib import Path
import numpy as np
import pydicom
import torch
from customDataLoader import resize



if len(sys.argv) != 3:
    print("Usage: python MRI_Data_preprocessing.py <MRI_input_dir> <output_dir>")
    sys.exit(1)

MRI_dir = sys.argv[1]
outputDir = sys.argv[2]

dcm_patient_filenames = os.listdir(MRI_dir)
print("dcm_patient_filenames: ", dcm_patient_filenames)

# Load all DICOM images into memory
patient_MRI_tensors = {}

def process_tensor(tensor_filename):
    image_folder = os.path.join(MRI_dir, tensor_filename)
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
    #patient_MRI_tensors[tensor_filename] = tensor
    
    # save the tensor to disk
    torch.save(tensor, os.path.join(outputDir, tensor_filename + ".pt"))

# now we want to save the processed MRI tensors to a folder
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=40)
    pool.map(process_tensor, dcm_patient_filenames)
    pool.close()
    pool.join()


print("MRI tensors saved to disk.")

