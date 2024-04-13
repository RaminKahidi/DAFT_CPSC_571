
from DAFT.daft.models.base import BaseModel
from DAFT.daft.networks.vol_blocks import ConvBnReLU, DAFTBlock, FilmBlock, ResBlock
from DAFT.daft.networks.vol_networks import DAFT
import torch

import numpy as np
import pandas as pd

import monai
import logging
import sys
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from customDataLoader import CombinedDataset
from torch.utils.data import DataLoader, random_split
import sys

print(torch.cuda.is_available())

outPrefix = "./a100Output/"
# outPrefix = "./"


def main(MRI_dir, tabular_dir):
    """
    Main function to configure and execute the model training.

    Parameters:
    - data_dir (str): Directory containing the data files.
    - bone_type (str): Type of bone to focus on during training.
    """

    # grab the name of the file in tabular directory
    tabularFileName = tabular_dir.split("/")[-1].split(".")[0]
    print(f"tabularFileName: {tabularFileName}")

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DAFT')

    bn_momentum = 0.99
    basefilters = 4
    classCount = 2

    # model = DAFT(
    #     in_channels=1,  
    #     n_outputs=4, 
    #     bn_momentum=bn_momentum,
    #     n_basefilters=basefilters
    # )
    model = DAFT(
        in_channels=1,  
        n_outputs=2, 
        bn_momentum=bn_momentum,
        n_basefilters=basefilters
    )
    model.to(device)
    loss_function = torch.nn.MSELoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)

    scaler = GradScaler()
    # writer = SummaryWriter()
    early_stop_patience = 20
    epochs_without_improvement = 0 
    # batch_size = 4
    # batch_size = 1
    # batch_size = 10
    batch_size = 4
    val_interval = 2
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Instantiate your dataset
    dataset = CombinedDataset(data_dir=MRI_dir, tabular_dir=tabular_dir, workers = 20)


    fold_results = pd.DataFrame(columns=['Fold', 'Best Val Loss', 'Best Val MAE', 'Best Val R2'])

    # workers = 1
    workers = 4

    fold = 0
    # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f"FOLD {fold}")
        # print("--------------------------------")
        # print(f"Train IDs: {train_ids}")
        # print(f"Test IDs: {test_ids}")
    epochs_without_improvement = 0
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    # val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    # train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    # val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)



    # Define the ratio of train/test split
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate the number of samples to include in each set.
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)


    best_val_loss = np.inf
    best_val_mae = np.inf
    best_val_r2 = -np.inf

    train_records = pd.DataFrame(columns=['Epoch', 'Batch', 'Prediction', 'Loss', 'True Value'])
    val_records = pd.DataFrame(columns=['Epoch', 'Batch', 'Prediction', 'Loss', 'True Value'])

    epochCount = 50

    for epoch in range(epochCount):
        print(f"Epoch {epoch + 1}/{epochCount}")
        model.train()
        epoch_loss = 0

        for batch_data in train_loader:
            inputs, labels, tabular_data = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
            # inputs = inputs.half()
            if epoch == 0:
                print(f"Training labels: {labels}")
            optimizer.zero_grad()
            with autocast():
                output_dict = model(inputs, tabular_data)
                logits = output_dict["logits"]
                print(f"Logits shape before squeeze: {logits.shape}")
                for i in range(logits.size(0)):  
                    print(logits[i].tolist())
                logits = logits.squeeze(1)
                print(f"Logits shape: {logits.shape}")
                print(f"Labels shape: {labels.shape}")
                for i in range(logits.size(0)):  
                    print(logits[i].tolist())  
                loss = loss_function(logits.float(), labels.float())
                print("This is the real loss : ", loss)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            train_records = train_records.append({
                'Epoch': epoch,
                'Prediction': logits.detach().cpu().numpy(),
                'Loss': loss.item(),
                'True Value': labels.cpu().numpy()
            }, ignore_index=True)
            print(f"train_records: {train_records}")

        epoch_loss /= len(train_loader)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels, val_tabular_data = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                    if epoch == 0:
                        print(f"Validation labels: {val_labels}")
                        print(f"Val_tabualr data labels: {val_tabular_data}")
                        print(f"Validation Images: {val_images.shape}")
                    output_dict = model(val_images, val_tabular_data)
                    logits = output_dict["logits"]
                    logits = logits.squeeze(1)
                    print(f"Logits shape: {logits.shape}")
                    print(f"Labels shape: {val_labels.shape}")
                    for i in range(logits.size(0)):
                        print(logits[i].tolist())

                        
                    val_loss += loss_function(logits.float(), val_labels.float()).item()
                    print("This is the real loss : ", val_loss)
                    # val_preds.extend(logits.view(-1).cpu().numpy())
                    val_preds.extend(logits.view(-1).cpu().numpy().reshape(-1, classCount).tolist())
                    val_targets.extend(val_labels.cpu().numpy())
                    print(f"val_targets: {val_targets}")
                    print(f"val_preds: {val_preds}")
                    val_records = val_records.append({
                    'Epoch': epoch,
                    'Prediction': logits.detach().cpu().numpy(),
                    'Loss': loss.item(),
                    'True Value': val_labels.cpu().numpy()
                }, ignore_index=True)

            val_loss /= len(val_loader)
            print(f"Batch size (logits): {logits.shape[0]}, Batch size (labels): {val_labels.shape[0]}")
            print(f"Total Predictions: {len(val_preds)}, Total Targets: {len(val_targets)}")
            # val_mae = mean_absolute_error(val_targets, val_preds)
            val_mae = mean_absolute_error(np.array(val_targets).flatten(), np.array(val_preds).flatten())
            val_r2 = r2_score(val_targets, val_preds)

            print(f"Validation Loss - Fold {fold}, Epoch {epoch + 1}: {val_loss:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                best_val_r2 = val_r2
                epochs_without_improvement = 0
                # Save the model checkpoint if it's the best so far
                torch.save(model.state_dict(), f"Best_Model_Fold_{fold}_Epoch_{epoch}.pth")
            else:
                epochs_without_improvement = epochs_without_improvement + 1

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping triggered for fold:", fold)
            break

        # save a temporary output file every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_records.to_csv(f'{outPrefix}tempOutputs/train_records_tabular{tabularFileName}_epoch_{epoch}_momentum_{bn_momentum}_basefilters{basefilters}_batch_size{batch_size}.csv', index=False)
            val_records.to_csv(f'{outPrefix}tempOutputs/val_records_tabular{tabularFileName}_epoch_{epoch}_momentum_{bn_momentum}_basefilters{basefilters}_batch_size{batch_size}.csv', index=False)
        

    fold_results = fold_results.append({
        'Fold': fold,
        'Best Val Loss': best_val_loss,
        'Best Val MAE': best_val_mae,
        'Best Val R2': best_val_r2
    }, ignore_index=True)
    train_records.to_csv(f'{outPrefix}train_records_tabular{tabularFileName}_epoch_{epoch}_momentum_{bn_momentum}_basefilters{basefilters}_batch_size{batch_size}.csv', index=False)
    val_records.to_csv(f'{outPrefix}val_records_tabular{tabularFileName}_epoch_{epoch}_momentum_{bn_momentum}_basefilters{basefilters}_batch_size{batch_size}.csv', index=False)

    # print(f"Completed Fold {fold}")

# writer.close()
print("Training completed.")
# print("Fold Results:\n", fold_results)

#MRI_dir = "./RawDataset/MRIs/"
# MRI_dir = "./ProcessedDataset/MRIs/"

# tabular_dir = "./RawDataset/Clinical_and_Other_Features.xlsx"
# tabular_dir = "./RawDataset/cleaned.csv"
# tabular_dir = "./RawDataset/ER_target.csv"
# tabular_dir = "./RawDataset/HER2_target.csv"

# main(MRI_dir=MRI_dir, tabular_dir=tabular_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide two command line arguments: MRI data directory and tabular data directory")
        sys.exit(1)
    
    MRI_dir = sys.argv[1]
    tabular_dir = sys.argv[2]

    print(f"MRI Directory: {MRI_dir}")
    print(f"Tabular Directory: {tabular_dir}")
    
    main(MRI_dir, tabular_dir)
