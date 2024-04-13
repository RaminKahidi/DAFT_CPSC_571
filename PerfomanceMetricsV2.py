import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to convert string representation of arrays to actual arrays
def parse_array(s):
    # Handling multi-row array strings
    return np.array([np.fromstring(line.strip(' []'), sep=' ') for line in s.splitlines()])

# Load the data
data = pd.read_csv('outputs/train_records_fold_0_epoch_43.csv')

# Parse 'Prediction' and 'True Value' columns
data['Prediction'] = data['Prediction'].apply(parse_array)
data['True Value'] = data['True Value'].apply(parse_array)

# Extract epoch numbers to calculate metrics per epoch
epochs = data['Epoch'].unique()

# Dictionary to store performance metrics for each epoch
performance_metrics = {}

for epoch in epochs:
    # Filter data for the current epoch
    epoch_data = data[data['Epoch'] == epoch]
    predicted_scores = np.vstack(epoch_data['Prediction'])
    true_labels = np.vstack(epoch_data['True Value'])

    # Find the index of the maximum value in each prediction array
    predicted_labels = np.argmax(predicted_scores, axis=1)
    true_labels_flat = np.argmax(true_labels, axis=1)

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels_flat, predicted_labels)
    precision = precision_score(true_labels_flat, predicted_labels, average='macro')
    recall = recall_score(true_labels_flat, predicted_labels, average='macro')
    f1 = f1_score(true_labels_flat, predicted_labels, average='macro')

    # Store metrics for the current epoch
    performance_metrics[epoch] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Output the results
for epoch, metrics in performance_metrics.items():
    print(f"Epoch {epoch}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()  # Newline for better readability between epochs
