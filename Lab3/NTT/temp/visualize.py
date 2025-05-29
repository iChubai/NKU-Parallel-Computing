import matplotlib.pyplot as plt
import re
import numpy as np

# --- Data Extraction from result.csv content ---
log_content = """
[INFO] Loading bandwidth data...
[INFO] Loaded 3394 bandwidth records.
[INFO] Data normalized. Min: 8.75, Max: 406.41
[INFO] Created dataset with 3384 samples.
[INFO] Training samples: 2707
[INFO] Testing samples: 677
[INFO] MLP weights and biases initialized.
[INFO] Device memory allocated and initial parameters copied.
[INFO] Starting training...
[Epoch 1/200] Loss: 0.26118, Time: 9 ms
[Epoch 2/200] Loss: 0.180429, Time: 7 ms
[Epoch 3/200] Loss: 0.12743, Time: 7 ms
[Epoch 4/200] Loss: 0.0911636, Time: 7 ms
[Epoch 5/200] Loss: 0.065766, Time: 7 ms
[Epoch 6/200] Loss: 0.0477118, Time: 7 ms
[Epoch 7/200] Loss: 0.0348008, Time: 7 ms
[Epoch 8/200] Loss: 0.0255827, Time: 7 ms
[Epoch 9/200] Loss: 0.0190346, Time: 7 ms
[Epoch 10/200] Loss: 0.0144147, Time: 7 ms
[Epoch 11/200] Loss: 0.011179, Time: 7 ms
[Epoch 12/200] Loss: 0.00892826, Time: 7 ms
[Epoch 13/200] Loss: 0.00737258, Time: 7 ms
[Epoch 14/200] Loss: 0.00630312, Time: 7 ms
[Epoch 15/200] Loss: 0.005571, Time: 7 ms
[Epoch 16/200] Loss: 0.0050712, Time: 7 ms
[Epoch 17/200] Loss: 0.00473037, Time: 7 ms
[Epoch 18/200] Loss: 0.00449772, Time: 7 ms
[Epoch 19/200] Loss: 0.00433836, Time: 7 ms
[Epoch 20/200] Loss: 0.00422846, Time: 7 ms
[Epoch 21/200] Loss: 0.00415182, Time: 7 ms
[Epoch 22/200] Loss: 0.00409748, Time: 7 ms
[Epoch 23/200] Loss: 0.00405808, Time: 7 ms
[Epoch 24/200] Loss: 0.0040287, Time: 7 ms
[Epoch 25/200] Loss: 0.00400599, Time: 7 ms
[Epoch 26/200] Loss: 0.00398777, Time: 7 ms
[Epoch 27/200] Loss: 0.00397257, Time: 7 ms
[Epoch 28/200] Loss: 0.00395943, Time: 7 ms
[Epoch 29/200] Loss: 0.00394766, Time: 7 ms
[Epoch 30/200] Loss: 0.00393681, Time: 7 ms
[Epoch 31/200] Loss: 0.00392659, Time: 7 ms
[Epoch 32/200] Loss: 0.00391682, Time: 7 ms
[Epoch 33/200] Loss: 0.00390739, Time: 7 ms
[Epoch 34/200] Loss: 0.00389818, Time: 7 ms
[Epoch 35/200] Loss: 0.00388918, Time: 7 ms
[Epoch 36/200] Loss: 0.00388035, Time: 7 ms
[Epoch 37/200] Loss: 0.00387165, Time: 7 ms
[Epoch 38/200] Loss: 0.00386307, Time: 7 ms
[Epoch 39/200] Loss: 0.00385461, Time: 7 ms
[Epoch 40/200] Loss: 0.00384626, Time: 7 ms
[Epoch 41/200] Loss: 0.00383797, Time: 7 ms
[Epoch 42/200] Loss: 0.00382976, Time: 7 ms
[Epoch 43/200] Loss: 0.00382161, Time: 7 ms
[Epoch 44/200] Loss: 0.00381354, Time: 7 ms
[Epoch 45/200] Loss: 0.00380551, Time: 7 ms
[Epoch 46/200] Loss: 0.00379756, Time: 7 ms
[Epoch 47/200] Loss: 0.00378968, Time: 7 ms
[Epoch 48/200] Loss: 0.00378188, Time: 7 ms
[Epoch 49/200] Loss: 0.00377417, Time: 7 ms
[Epoch 50/200] Loss: 0.00376656, Time: 7 ms
[Epoch 51/200] Loss: 0.00375903, Time: 7 ms
[Epoch 52/200] Loss: 0.00375155, Time: 7 ms
[Epoch 53/200] Loss: 0.00374415, Time: 7 ms
[Epoch 54/200] Loss: 0.00373683, Time: 7 ms
[Epoch 55/200] Loss: 0.00372954, Time: 7 ms
[Epoch 56/200] Loss: 0.0037223, Time: 7 ms
[Epoch 57/200] Loss: 0.00371505, Time: 7 ms
[Epoch 58/200] Loss: 0.00370781, Time: 7 ms
[Epoch 59/200] Loss: 0.0037006, Time: 7 ms
[Epoch 60/200] Loss: 0.00369345, Time: 7 ms
[Epoch 61/200] Loss: 0.00368633, Time: 7 ms
[Epoch 62/200] Loss: 0.00367925, Time: 7 ms
[Epoch 63/200] Loss: 0.00367222, Time: 7 ms
[Epoch 64/200] Loss: 0.00366527, Time: 7 ms
[Epoch 65/200] Loss: 0.00365841, Time: 7 ms
[Epoch 66/200] Loss: 0.00365161, Time: 7 ms
[Epoch 67/200] Loss: 0.00364486, Time: 7 ms
[Epoch 68/200] Loss: 0.0036381, Time: 7 ms
[Epoch 69/200] Loss: 0.00363136, Time: 7 ms
[Epoch 70/200] Loss: 0.00362469, Time: 7 ms
[Epoch 71/200] Loss: 0.00361807, Time: 7 ms
[Epoch 72/200] Loss: 0.00361148, Time: 7 ms
[Epoch 73/200] Loss: 0.00360483, Time: 7 ms
[Epoch 74/200] Loss: 0.00359818, Time: 7 ms
[Epoch 75/200] Loss: 0.00359156, Time: 7 ms
[Epoch 76/200] Loss: 0.00358497, Time: 7 ms
[Epoch 77/200] Loss: 0.00357845, Time: 7 ms
[Epoch 78/200] Loss: 0.00357194, Time: 7 ms
[Epoch 79/200] Loss: 0.0035655, Time: 7 ms
[Epoch 80/200] Loss: 0.00355917, Time: 7 ms
[Epoch 81/200] Loss: 0.00355294, Time: 7 ms
[Epoch 82/200] Loss: 0.00354675, Time: 7 ms
[Epoch 83/200] Loss: 0.00354063, Time: 7 ms
[Epoch 84/200] Loss: 0.00353458, Time: 7 ms
[Epoch 85/200] Loss: 0.00352855, Time: 7 ms
[Epoch 86/200] Loss: 0.00352252, Time: 7 ms
[Epoch 87/200] Loss: 0.00351653, Time: 7 ms
[Epoch 88/200] Loss: 0.00351057, Time: 7 ms
[Epoch 89/200] Loss: 0.00350468, Time: 7 ms
[Epoch 90/200] Loss: 0.00349883, Time: 7 ms
[Epoch 91/200] Loss: 0.00349302, Time: 7 ms
[Epoch 92/200] Loss: 0.00348724, Time: 7 ms
[Epoch 93/200] Loss: 0.00348151, Time: 7 ms
[Epoch 94/200] Loss: 0.00347581, Time: 7 ms
[Epoch 95/200] Loss: 0.00347019, Time: 7 ms
[Epoch 96/200] Loss: 0.00346462, Time: 7 ms
[Epoch 97/200] Loss: 0.00345913, Time: 7 ms
[Epoch 98/200] Loss: 0.00345369, Time: 7 ms
[Epoch 99/200] Loss: 0.00344832, Time: 7 ms
[Epoch 100/200] Loss: 0.00344298, Time: 7 ms
[Epoch 101/200] Loss: 0.0034377, Time: 7 ms
[Epoch 102/200] Loss: 0.00343251, Time: 7 ms
[Epoch 103/200] Loss: 0.00342737, Time: 7 ms
[Epoch 104/200] Loss: 0.00342228, Time: 7 ms
[Epoch 105/200] Loss: 0.00341726, Time: 7 ms
[Epoch 106/200] Loss: 0.00341231, Time: 7 ms
[Epoch 107/200] Loss: 0.00340742, Time: 7 ms
[Epoch 108/200] Loss: 0.00340257, Time: 7 ms
[Epoch 109/200] Loss: 0.00339778, Time: 7 ms
[Epoch 110/200] Loss: 0.00339304, Time: 7 ms
[Epoch 111/200] Loss: 0.00338837, Time: 7 ms
[Epoch 112/200] Loss: 0.00338376, Time: 7 ms
[Epoch 113/200] Loss: 0.00337918, Time: 7 ms
[Epoch 114/200] Loss: 0.00337467, Time: 7 ms
[Epoch 115/200] Loss: 0.0033702, Time: 7 ms
[Epoch 116/200] Loss: 0.00336575, Time: 7 ms
[Epoch 117/200] Loss: 0.00336135, Time: 7 ms
[Epoch 118/200] Loss: 0.00335698, Time: 7 ms
[Epoch 119/200] Loss: 0.00335267, Time: 7 ms
[Epoch 120/200] Loss: 0.0033484, Time: 7 ms
[Epoch 121/200] Loss: 0.00334419, Time: 7 ms
[Epoch 122/200] Loss: 0.00334003, Time: 7 ms
[Epoch 123/200] Loss: 0.00333591, Time: 7 ms
[Epoch 124/200] Loss: 0.00333185, Time: 7 ms
[Epoch 125/200] Loss: 0.00332782, Time: 7 ms
[Epoch 126/200] Loss: 0.00332385, Time: 7 ms
[Epoch 127/200] Loss: 0.00331993, Time: 7 ms
[Epoch 128/200] Loss: 0.00331603, Time: 7 ms
[Epoch 129/200] Loss: 0.00331218, Time: 7 ms
[Epoch 130/200] Loss: 0.00330836, Time: 7 ms
[Epoch 131/200] Loss: 0.00330458, Time: 7 ms
[Epoch 132/200] Loss: 0.00330085, Time: 7 ms
[Epoch 133/200] Loss: 0.00329717, Time: 7 ms
[Epoch 134/200] Loss: 0.00329352, Time: 7 ms
[Epoch 135/200] Loss: 0.00328991, Time: 7 ms
[Epoch 136/200] Loss: 0.00328634, Time: 7 ms
[Epoch 137/200] Loss: 0.00328282, Time: 7 ms
[Epoch 138/200] Loss: 0.00327935, Time: 7 ms
[Epoch 139/200] Loss: 0.00327592, Time: 7 ms
[Epoch 140/200] Loss: 0.00327253, Time: 7 ms
[Epoch 141/200] Loss: 0.00326919, Time: 7 ms
[Epoch 142/200] Loss: 0.00326587, Time: 7 ms
[Epoch 143/200] Loss: 0.0032626, Time: 7 ms
[Epoch 144/200] Loss: 0.00325936, Time: 7 ms
[Epoch 145/200] Loss: 0.00325614, Time: 7 ms
[Epoch 146/200] Loss: 0.00325296, Time: 7 ms
[Epoch 147/200] Loss: 0.0032498, Time: 7 ms
[Epoch 148/200] Loss: 0.00324667, Time: 7 ms
[Epoch 149/200] Loss: 0.00324359, Time: 7 ms
[Epoch 150/200] Loss: 0.00324054, Time: 7 ms
[Epoch 151/200] Loss: 0.00323755, Time: 7 ms
[Epoch 152/200] Loss: 0.00323459, Time: 7 ms
[Epoch 153/200] Loss: 0.00323168, Time: 7 ms
[Epoch 154/200] Loss: 0.00322879, Time: 7 ms
[Epoch 155/200] Loss: 0.00322593, Time: 7 ms
[Epoch 156/200] Loss: 0.00322311, Time: 7 ms
[Epoch 157/200] Loss: 0.0032203, Time: 7 ms
[Epoch 158/200] Loss: 0.0032175, Time: 7 ms
[Epoch 159/200] Loss: 0.00321474, Time: 7 ms
[Epoch 160/200] Loss: 0.00321201, Time: 7 ms
[Epoch 161/200] Loss: 0.00320931, Time: 7 ms
[Epoch 162/200] Loss: 0.00320664, Time: 7 ms
[Epoch 163/200] Loss: 0.003204, Time: 7 ms
[Epoch 164/200] Loss: 0.00320139, Time: 7 ms
[Epoch 165/200] Loss: 0.00319882, Time: 7 ms
[Epoch 166/200] Loss: 0.00319627, Time: 7 ms
[Epoch 167/200] Loss: 0.00319375, Time: 7 ms
[Epoch 168/200] Loss: 0.00319126, Time: 7 ms
[Epoch 169/200] Loss: 0.00318881, Time: 7 ms
[Epoch 170/200] Loss: 0.00318637, Time: 7 ms
[Epoch 171/200] Loss: 0.00318396, Time: 7 ms
[Epoch 172/200] Loss: 0.00318158, Time: 7 ms
[Epoch 173/200] Loss: 0.00317922, Time: 7 ms
[Epoch 174/200] Loss: 0.00317688, Time: 7 ms
[Epoch 175/200] Loss: 0.00317457, Time: 7 ms
[Epoch 176/200] Loss: 0.00317229, Time: 10 ms
[Epoch 177/200] Loss: 0.00317002, Time: 7 ms
[Epoch 178/200] Loss: 0.00316778, Time: 7 ms
[Epoch 179/200] Loss: 0.00316556, Time: 7 ms
[Epoch 180/200] Loss: 0.00316335, Time: 7 ms
[Epoch 181/200] Loss: 0.00316117, Time: 7 ms
[Epoch 182/200] Loss: 0.00315902, Time: 7 ms
[Epoch 183/200] Loss: 0.00315688, Time: 7 ms
[Epoch 184/200] Loss: 0.00315477, Time: 7 ms
[Epoch 185/200] Loss: 0.00315269, Time: 7 ms
[Epoch 186/200] Loss: 0.00315062, Time: 7 ms
[Epoch 187/200] Loss: 0.00314858, Time: 7 ms
[Epoch 188/200] Loss: 0.00314656, Time: 7 ms
[Epoch 189/200] Loss: 0.00314455, Time: 7 ms
[Epoch 190/200] Loss: 0.00314256, Time: 7 ms
[Epoch 191/200] Loss: 0.0031406, Time: 7 ms
[Epoch 192/200] Loss: 0.00313867, Time: 7 ms
[Epoch 193/200] Loss: 0.00313675, Time: 7 ms
[Epoch 194/200] Loss: 0.00313486, Time: 7 ms
[Epoch 195/200] Loss: 0.00313298, Time: 7 ms
[Epoch 196/200] Loss: 0.00313112, Time: 7 ms
[Epoch 197/200] Loss: 0.00312928, Time: 7 ms
[Epoch 198/200] Loss: 0.00312745, Time: 7 ms
[Epoch 199/200] Loss: 0.00312565, Time: 7 ms
[Epoch 200/200] Loss: 0.00312385, Time: 7 ms
[INFO] Training finished.

[INFO] Starting inference on test set...
[WARN] Test samples (677) > BATCH_SIZE (256). Reallocating intermediate buffers for inference.
[INFO] Test MSE: 0.00364974
[INFO] Sample Predictions (denormalized):
  Predicted: 225.773, Actual: 259.88
  Predicted: 250.119, Actual: 256.84
  Predicted: 246.509, Actual: 252.1
  Predicted: 252.644, Actual: 246.65
  Predicted: 249.311, Actual: 258.93
  Predicted: 248.718, Actual: 239.48
  Predicted: 256.432, Actual: 179.27
  Predicted: 208.397, Actual: 169.52
  Predicted: 198.217, Actual: 174.92
  Predicted: 192.675, Actual: 171.29
[INFO] Device memory freed.
"""

epochs_data = []
losses_data = []
predicted_values = []
actual_values = []
test_mse = None

for line in log_content.splitlines():
    epoch_loss_match = re.search(r"\[Epoch (\d+)/(\d+)\] Loss: ([\d\.]+)", line)
    if epoch_loss_match:
        epochs_data.append(int(epoch_loss_match.group(1)))
        losses_data.append(float(epoch_loss_match.group(3)))
    
    test_mse_match = re.search(r"\[INFO\] Test MSE: ([\d\.]+)", line)
    if test_mse_match:
        test_mse = float(test_mse_match.group(1))
        
    prediction_match = re.search(r"Predicted: ([\d\.]+), Actual: ([\d\.]+)", line)
    if prediction_match:
        predicted_values.append(float(prediction_match.group(1)))
        actual_values.append(float(prediction_match.group(2)))

if test_mse is not None:
    print("Test MSE: {:.6f}".format(test_mse))
else:
    print("Test MSE: N/A")

# Plotting Training Loss
if epochs_data and losses_data:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_data, losses_data, label='Training Loss', marker='o', linestyle='-')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_curve.png')  # Save the plot
    plt.show()
else:
    print("No training loss data found to plot.")

# Plotting Predicted vs. Actual values
if predicted_values and actual_values:
    plt.figure(figsize=(10, 6))
    num_samples_to_plot = len(predicted_values)
    sample_indices = np.arange(num_samples_to_plot)
    
    plt.plot(sample_indices, actual_values, label='Actual Values', marker='x', linestyle='--')
    plt.plot(sample_indices, predicted_values, label='Predicted Values', marker='o', linestyle='-')

    if test_mse is not None:
        title_text = 'Sample Predictions vs. Actual Values (Test MSE: {:.6f})'.format(test_mse)
    else:
        title_text = 'Sample Predictions vs. Actual Values (Test MSE: N/A)'
    plt.title(title_text)

    plt.xlabel('Sample Index')
    plt.ylabel('Bandwidth Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')  # Save the plot
    plt.show()
else:
    print("No prediction data found to plot.")
