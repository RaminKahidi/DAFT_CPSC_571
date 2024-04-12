#Calculates the performance metrics for the machine learning model.

import numpy as np
import pandas as pd

data = pd.read_csv('val_records_fold_0_epoch_0.csv')

# Function to convert string representation of arrays to actual arrays
def parse_array(s):
    return np.array([float(x) for x in s.strip('[]').split()])

# Parse 'Prediction' and 'True Value' columns
Predicted = data['Prediction'].apply(lambda x: parse_array(x[1:-1]))
True_vals = data['True Value'].apply(lambda x: parse_array(x[1:-1]).astype(int))

# Convert the columns to NumPy arrays
predicted_scores = np.array(Predicted.tolist())
true_labels = np.array(True_vals.tolist())

# Find the index of the maximum value in each prediction array
predicted_indices = np.argmax(predicted_scores, axis=1)
predicted_labels = np.zeros_like(true_labels)
predicted_labels[np.arange(len(predicted_labels)), predicted_indices] = 1
true_labels_flat = true_labels.argmax(axis=1)

# Compute true positives, false positives, and false negatives
true_positives = np.sum(predicted_labels[np.arange(len(predicted_labels)), true_labels_flat])
false_positives = np.sum(predicted_labels[np.arange(len(predicted_labels)), ~true_labels_flat])
false_negatives = np.sum(~predicted_labels[np.arange(len(predicted_labels)), true_labels_flat])  # Logical AND with negated predicted_labels

accuracy = np.mean(predicted_labels)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

# Print precision and recall
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

