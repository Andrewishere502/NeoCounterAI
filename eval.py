'''Script to evaluate a given CNN model.'''

import pathlib
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def get_img_data(meta_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    '''Return an array of each image as a 2D array of pixels, and an
    array of labels corresponding to each image.
    '''
    # Load in the image arrays and the number of shrimp in them
    img_arrays = []
    img_labels = []
    for row in meta_df.itertuples():
        # First element in row is the index, so zip with columns
        # starting from the first element.
        index = row[0]
        labeled_row = dict(zip(meta_df.columns, row[1:]))

        # Construct the path to where
        img_path = pathlib.Path(labeled_row['NewDir'],
                                labeled_row['NewName'])

        # load in the image
        img_array = plt.imread(img_path)
        
        # Add the image array and its label to
        # their respective lists
        img_arrays.append(img_array)
        img_labels.append(labeled_row['NShrimp'])
        
        # Display progress bar
        print(f'\r{len(img_arrays)} of {len(meta_df)} loaded', end='')
    print()

    # Convert img_arrays and img_labels from lists to arrays
    img_arrays = np.array(img_arrays)
    img_labels = np.array(img_labels, dtype=int)
    return (img_arrays, img_labels)


# model_path = pathlib.Path('Models/20250422-181628/model.keras')  # 1 epochs
# model_path = pathlib.Path('Models/20250422-222318/model.keras')  # 5 epochs
# model_path = pathlib.Path('Models/20250422-224232/model.keras')  # 10 epochs
# model_path = pathlib.Path('Models/20250422-225100/model.keras')  # 20 epochs
model_path = pathlib.Path('Models/20250423-002634/model.keras')  # 50 epochs
model = tf.keras.models.load_model(model_path)


# Crudely copy and pasted from resnet.py
validation_is = [389, 448, 93, 259, 554, 675, 1038, 170, 254, 731, 602,
                 252, 306, 547, 1069, 1070, 1040, 428, 186, 269, 699, 630,
                 162, 413, 917, 1078, 98, 6, 950, 686, 1010, 611, 229,
                 264, 368, 856, 1020, 1003, 729, 1044, 799, 403, 863, 843,
                 610, 189, 332, 303, 260, 144, 627, 698, 283, 1022, 11,
                 965, 594, 693, 531, 469, 127, 736, 869, 57, 33, 237, 1055,
                 282, 586, 362, 243, 603, 92, 210, 337, 862, 791, 668, 301,
                 959, 85, 190, 39, 500, 623, 1016, 507, 634, 764, 822, 516,
                 723, 104, 988, 233, 676, 105, 341, 753, 697, 253, 408,
                 422, 774, 208, 927, 1083, 937, 82, 497, 1054, 583, 924,
                 801, 87, 64, 270, 605, 790, 274, 1087, 1027, 941, 125,
                 153, 613, 871, 928, 410, 830, 888, 1037, 957, 517, 44,
                 625, 390, 220, 140, 349, 357, 722, 133, 261, 278, 898,
                 79, 195, 115, 967, 992, 885, 718, 380, 1062, 184, 958,
                 358, 1051, 36, 858, 339, 993, 754, 420, 149, 9, 1081, 840,
                 45, 913, 95, 1046, 534, 548, 560, 298, 290, 415, 313, 421,
                 657, 122, 997, 968, 361, 336, 135, 1007, 827, 709, 977,
                 769, 651, 367, 88, 86, 619, 8, 97, 1076, 348, 509, 211,
                 703, 945, 746, 21, 482, 570, 939, 405, 784, 887, 1053,
                 1089, 1005, 478, 590, 491]
# Path to data
data_dir = pathlib.Path('DataNoSubstrate')
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')
# Only keep the rows that belong to the validation set
meta_df = meta_df.loc[validation_is]

# Load the images and their labels
X_valid, y_valid = get_img_data(meta_df)

# Let the model predict off of X_valid. Don't forget to transform the
# images
resnet_prep = tf.keras.applications.resnet.preprocess_input
y_valid_pred = model.predict(resnet_prep(X_valid)).flatten()

# Print some images and their predicted number of shrimp
NROWS = 4
NCOLS = 5
fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
for i, img_array in enumerate(X_valid[:NROWS*NCOLS]):
    axs[i//NCOLS][i%NCOLS].imshow(img_array)
    axs[i//NCOLS][i%NCOLS].set_title(f'{y_valid_pred[i]:.0f} ({y_valid_pred[i]:.2f}) | {y_valid[i]:.0f}')
    axs[i//NCOLS][i%NCOLS].axis('off')
fig.tight_layout()
plt.show()

# Count the number of correct and incorrect
n_correct = 0
n_incorrect = 0
for pred, correct in zip(y_valid_pred, y_valid):
    # Round properly...
    if pred // 1 + int((pred % 1) > 0.5) == correct:
        n_correct += 1
    else:
        n_incorrect += 1
print(f'Labeled {n_correct} images correctly')
print(f'Labeled {n_incorrect} images incorrectly')

# Display some images that were incorrectly labeled
# Print some images and their predicted number of shrimp
fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
ax_i = 0
for i, img_array in enumerate(X_valid):
    pred = y_valid_pred[i]
    pred_round = round(y_valid_pred[i])
    correct = y_valid[i]
    if pred_round != correct:
        axs[ax_i//NCOLS][ax_i%NCOLS].imshow(img_array)
        axs[ax_i//NCOLS][ax_i%NCOLS].set_title(f'{pred_round} ({pred:.2f}) | {correct:.0f}')
        axs[ax_i//NCOLS][ax_i%NCOLS].axis('off')
        ax_i += 1
        # Stop when 20 images have been found
        if ax_i == NROWS * NCOLS:
            break
fig.tight_layout()
plt.show()

# Create histograms to compare validation dataset true labels with
# the model's predictions
fig, axs = plt.subplots(nrows=1, ncols=2)
# Create y_valid histogram
axs[0].hist(y_valid, bins=np.unique(y_valid))
axs[0].set_xticks(np.unique(y_valid))
axs[0].set_xlabel('Label Value')
axs[0].set_ylabel('Frequency')
axs[0].set_title(f'Distribution of True Labels for Validation Data')
# Create y_valid_pred histogram
axs[1].hist(y_valid_pred, bins=np.unique(y_valid))
axs[1].set_xticks(np.unique(y_valid))
axs[1].set_xlabel('Label Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title(f'Distribution of Predicted Labels for Validation Data')
plt.show()
