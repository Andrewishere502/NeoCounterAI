'''Script to evaluate a given CNN model.'''

import pathlib
from typing import Tuple, Dict, Any, List
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats as stats


class ModelManager:
    def __init__(self, hex_hash: str, model_name: str=None) -> None:
        # Set model_name to the hex_hash if no name provided
        if model_name == None:
            self.__model_name: str = hex_hash
        else:
            self.__model_name: str = model_name
        self.__hex_hash: str = hex_hash

        # Directory where the model and its accompanying data is stored
        self.__model_dir: pathlib.Path = pathlib.Path('Models', hex_hash)

        # Settings used to build this model
        self.__settings: Dict = self.__parse_settings()

        # NOTE: What type is this exactly?
        self.__model = tf.keras.models.load_model(self.model_file)
        return
    
    @property
    def model_name(self) -> str:
        '''Return the model's name.'''
        return self.__model_name
    
    @property
    def hex_hash(self) -> str:
        '''Return the hex encoded hash of the model.'''
        return self.__hex_hash
    
    @property
    def model_dir(self) -> pathlib.Path:
        '''Return the directory in which this model is found.'''
        return self.__model_dir

    @property
    def model_file(self) -> pathlib.Path:
        '''Return the path to the model file.'''
        return self.model_dir / 'model.keras'
    
    @property
    def settings_file(self) -> pathlib.Path:
        '''Return the path to the settings file for this model.'''
        return self.model_dir / 'settings.txt'
    
    @property
    def partitions_file(self) -> pathlib.Path:
        '''Return the path to the data partitions file.'''
        return self.model_dir / 'data_partitions.txt'

    @property
    def model(self):  # NOTE: What time is this, update later
        '''Return the keras model instance.'''
        return self.__model

    def get_setting(self, name: str) -> Any:
        '''Return the value of a setting given its name.'''
        return self.__settings[name]

    @staticmethod
    def __dtype_convert(dtype: str, value: str) -> Any:
        '''Return the value as the given data type.'''
        if dtype == 'str':
            # Seems redundant... but feels right to do this just in case
            return str(value)
        elif dtype == 'int':
            return int(value)
        elif dtype == 'float':
            return float(value)
        elif dtype == 'NoneType':
            return None
        elif dtype == 'list':
            # Get list values as a list of strings
            str_values =  value[1:-1].split(', ')
            # List was empty, return empty list
            if len(str_values) == 1 and str_values[0] == '':
                return []
            return list(map(int, str_values))
        else:
            raise TypeError(f'Conversion to dtype \'{dtype}\' not supported')
    
    def __parse_settings(self) -> Dict:
        '''Parse the settings.txt file, loading in its values to a
        dictionary.
        '''
        # Read lines of the settings file, then close it
        with open(self.settings_file, 'r') as file:
            lines = file.readlines()
        # Strip white space from end of each line
        lines = map(lambda l: l.strip(), lines)

        # Construct the settings dictionary
        settings = {}
        for line in lines:
            name_dtype, value = line.split('=')
            name, dtype = name_dtype.split(':')
            # Convert the value to the appropriate data type
            settings.update({name: self.__dtype_convert(dtype, value)})
        return settings

    def load_partition(self, partition_name: str) -> List[int]:
        '''Return the indices of images belonging to a partition of
        the dataset, namely the training (train), validation (valid),
        or testing (test) partition.
        '''
        p = None
        if partition_name == 'train':
            # First line is training partition
            p = 0
        elif partition_name == 'valid':
            # First line is validation partition
            p = 1
        elif partition_name == 'test':
            # First line is testing partition
            p = 2
        else:
            raise ValueError(f'No partition of name \'{partition_name}\'')

        # Open the file and grab the partition(s)
        with open(self.partitions_file, 'r') as file:
            # Remove the first and last characters, which are
            part_line = file.readlines()[p]
        # Get the data component from the line, removing the partition name
        part_data = part_line.split('=')[1].strip()
        # Convert to the proper data type
        partition_indices: List[int] = self.__dtype_convert('list', part_data)
        return partition_indices


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


def true_round(value: float) -> int:
    '''Round to the nearest whole number (>0.5 -> 1), NOT using bankers rounding
    like python's builtin round function.
    '''
    return value // 1 + int((value % 1) > 0.5)


def create_pred_fig(nrows:int, ncols:int) -> Tuple[Figure, Axes]:
    '''Create a grid of images titled by their predicted number of
    shrimp and true number of shrimp.
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * nrows, int(1.2 * ncols)))
    for i, img_array in enumerate(X_valid[:nrows*ncols]):
        axs[i//ncols][i%ncols].imshow(img_array)
        axs[i//ncols][i%ncols].set_title(f'{y_pred[i]:.0f} ({y_pred[i]:.2f}) | {y_true[i]:.0f}')
        axs[i//ncols][i%ncols].axis('off')
    fig.tight_layout()
    return fig, axs


# Path to file to write a summary file for all the models
summary_file = pathlib.Path('Models', 'models_summary.csv')
summary_df = pd.read_csv(summary_file, index_col='HexHash')
# Create a row to represent this model
model_row = OrderedDict(zip(summary_df.columns, [''] * len(summary_df.columns)))

# Settings: epoch 1, max_weight 1, DataNoSubstrate
# hex_hash = '0x104f0ddc'
# Settings: epoch 5, max_weight 1, DataNoSubstrate
# hex_hash = '0x1086addc'
# Settings: epoch 10, max_weight 1, DataNoSubstrate
# hex_hash = '0x105099dc'
# Settings: epoch 20, max_weight 1, DataNoSubstrate
# hex_hash = '0x1068a1dc'
# Settings: epoch 50, max_weight 1, DataNoSubstrate
# hex_hash = '0x108a15dc'
# Settings: epoch 100, max_weight 1, DataNoSubstrate, 0.001 limit
hex_hash = '0x108e55dc'
# Settings: epoch 50, max_weight None, DataNoSubstrate
# hex_hash = '0x110a30e5'

print(f'Loading model {hex_hash}')
model_manager = ModelManager(hex_hash)

# Store some settings to be written to summary_df
model_row['Epochs'] = model_manager.get_setting('epochs')
model_row['MaxWeight'] = model_manager.get_setting('max_weight')
model_row['CollectionName'] = model_manager.get_setting('collection_name')

# Path to this model's data
data_dir = pathlib.Path(model_manager.get_setting('collection_name'))
meta_df = pd.read_csv(data_dir / 'metadata.csv', index_col='ID')

# Only keep the rows that belong to the validation set
validation_is = model_manager.load_partition('valid')
meta_df = meta_df.loc[validation_is]

# Load the images and their labels
X_valid, y_true = get_img_data(meta_df)

# Let the model predict off of X_valid. Don't forget to transform the
# images
resnet_prep = tf.keras.applications.resnet.preprocess_input
y_pred = model_manager.model.predict(resnet_prep(X_valid)).flatten()
y_pred_round = np.array(list(map(true_round, y_pred)))

# Calculate by how much each guess is off, then count the frequencies
# of these differences
y_pred_difs = y_pred_round - y_true
diff_count = dict(Counter(y_pred_difs))

# Array of unique shrimp counts
unique_shrimp_counts = np.unique(y_true)

# Get the models predictions, grouped by the true number of shrimp for
# each prediction
y_pred_by_true = {}
for usc in unique_shrimp_counts:
    # Get all predicted values for images with this many shrimp in them
    pred_for_count = y_pred_round[y_true == usc]
    y_pred_by_true[usc] = pred_for_count

# Group the differences in prediction and truth by the true number of
# shrimp
y_pred_diffs_by_true = {}
for usc in unique_shrimp_counts:
    # Get all predicted value differences for images with this many shrimp in them
    pred_diffs_for_count = y_pred_difs[y_true == usc]
    y_pred_diffs_by_true[usc] = pred_diffs_for_count


# Paired t-test to determine if predicted n shrimp is statistically
# different from true n shrimp, on average
result = stats.ttest_rel(y_pred, y_true)
print(f'Paired t-test; pvalue = {result.pvalue:.4f}, df={result.df}')
model_row['PredVsTrue_pval'] = result.pvalue
model_row['PredVsTrue_diff'] = result.pvalue < 0.05


# ANOVA to see if any image labels are predicted differently than
# others, on average
result = stats.f_oneway(*y_pred_diffs_by_true.values())
print(f'ANOVA; pvalue = {result.pvalue:.4f}')
model_row['PredVsTrueGrouped_pval'] = result.pvalue
model_row['PredVsTrueGrouped_diff'] = result.pvalue < 0.05


# Count total number of images predicted
model_row['NValid'] = len(y_pred)
# Count the number of correct
n_correct = 0
for pred, correct in zip(y_pred, y_true):
    # Round properly...
    if true_round(pred) == correct:
        n_correct += 1
model_row['NValidCorrect'] = n_correct


# Print some images and their predicted number of shrimp
NROWS = 4
NCOLS = 5
fig, axs = create_pred_fig(NROWS, NCOLS)
fig.savefig(model_manager.model_dir / 'valid-pred-plot.png')
plt.cla()
plt.clf()


# Make a histogram like this so we avoid fiddling with bins
plt.bar(diff_count.keys(), diff_count.values(), width=0.9)
# Label the bars by their height
for key, value in diff_count.items():
    plt.text(key, value, value)
plt.xlabel('Difference in Shrimp Predictions')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(model_manager.model_dir / 'valid-pred-diffs.png')
plt.cla()
plt.clf()


# Violin plot of predictions for images grouped by number of shrimp
# in image
plt.violinplot(y_pred_by_true.values(), y_pred_by_true.keys())
plt.plot(unique_shrimp_counts, unique_shrimp_counts, linestyle='dashed')
plt.xlabel('True Number of Shrimp')
plt.ylabel('Predicted Number of Shrimp')
plt.xticks(unique_shrimp_counts)
plt.savefig(model_manager.model_dir / 'valid-pred-violin.png')
plt.cla()
plt.clf()


# Violin plot of how much predictions differ from truth for images
# grouped by number of shrimp in image
plt.violinplot(y_pred_diffs_by_true.values(), y_pred_diffs_by_true.keys())
plt.plot(unique_shrimp_counts, [0] * len(unique_shrimp_counts), linestyle='dashed')
plt.xlabel('True Number of Shrimp')
plt.ylabel('Difference Between Predicted and True Number of Shrimp')
plt.xticks(unique_shrimp_counts)
plt.savefig(model_manager.model_dir / 'valid-pred-diff-violin.png')
plt.cla()
plt.clf()


# # Correlate true values and with the difference 
# # When regression line is above y=0 we are over predicting, when the
# # regression line is below y=0 we are under predicting.
# plt.plot(y_true, y_pred_difs, marker='o', linestyle='')
# m, b = np.polyfit(y_true, y_pred_difs, 1)
# plt.plot(y_true, m*y_true+b)  # Plot regression
# plt.plot(y_true, 0*y_true)  # Plot y = 0
# plt.show()


# Save this model's data as a row in models_summary.csv
summary_df.loc[hex_hash] = model_row
summary_df.to_csv(summary_file)

# # Display some images that were incorrectly labeled
# # Print some images and their predicted number of shrimp
# fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
# ax_i = 0
# for i, img_array in enumerate(X_valid):
#     pred = y_pred[i]
#     pred_round = round(y_pred[i])
#     correct = y_true[i]
#     if pred_round != correct:
#         axs[ax_i//NCOLS][ax_i%NCOLS].imshow(img_array)
#         axs[ax_i//NCOLS][ax_i%NCOLS].set_title(f'{pred_round} ({pred:.2f}) | {correct:.0f}')
#         axs[ax_i//NCOLS][ax_i%NCOLS].axis('off')
#         ax_i += 1
#         # Stop when 20 images have been found
#         if ax_i == NROWS * NCOLS:
#             break
# fig.tight_layout()
# plt.show()

# Create histograms to compare validation dataset true labels with
# the model's predictions
# fig, axs = plt.subplots(nrows=1, ncols=2)
# # Create y_true histogram
# axs[0].hist(y_true, bins=np.unique(y_true))
# axs[0].set_xticks(np.unique(y_true))
# axs[0].set_xlabel('Label Value')
# axs[0].set_ylabel('Frequency')
# axs[0].set_title(f'Distribution of True Labels for Validation Data')
# # Create y_pred histogram
# axs[1].hist(y_pred, bins=np.unique(y_true))
# axs[1].set_xticks(np.unique(y_true))
# axs[1].set_xlabel('Label Value')
# axs[1].set_ylabel('Frequency')
# axs[1].set_title(f'Distribution of Predicted Labels for Validation Data')
# plt.show()


# Do a *paired* t-test comparing predicted vs true shrimp number