from customDataLoader import CombinedDataset
from torch.utils.data import DataLoader

MRI_dir = "./RawDataset/MRIs/"
# tabular_dir = "./RawDataset/Clinical_and_Other_Features.xlsx"
tabular_dir = "./RawDataset/cleaned.csv"

# Instantiate your dataset
dataset = CombinedDataset(data_dir=MRI_dir, tabular_dir=tabular_dir)

# Instantiate the DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the DataLoader and print the shape of the outputs
for i, (img, label, tab_data) in enumerate(dataloader):
    print(f"Batch {i+1}")
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
    print(f"Tabular data shape: {tab_data.shape}")
    print("-----------------------------")

    # For testing, you might want to break the loop after a few batches
    if i == 10:
        break
