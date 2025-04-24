'''Script to evaluate a given CNN model.'''

import pathlib
from typing import Tuple, Dict, Any, List
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


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


# Path to file to write a summary file for all the models
summary_file = pathlib.Path('Models', 'models_summary.csv')
summary_df = pd.read_csv(summary_file, index_col='HexHash')
# Create a row to represent this model
model_row = OrderedDict(zip(summary_df.columns, [''] * len(summary_df.columns)))

# Settings: epoch 1, max_weight None, DataNoSubstrate
hex_hash = '0x104f0ddc'
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
X_valid, y_valid = get_img_data(meta_df)

# Let the model predict off of X_valid. Don't forget to transform the
# images
resnet_prep = tf.keras.applications.resnet.preprocess_input
y_valid_pred = model_manager.model.predict(resnet_prep(X_valid)).flatten()

# Print some images and their predicted number of shrimp
NROWS = 4
NCOLS = 5
fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
for i, img_array in enumerate(X_valid[:NROWS*NCOLS]):
    axs[i//NCOLS][i%NCOLS].imshow(img_array)
    axs[i//NCOLS][i%NCOLS].set_title(f'{y_valid_pred[i]:.0f} ({y_valid_pred[i]:.2f}) | {y_valid[i]:.0f}')
    axs[i//NCOLS][i%NCOLS].axis('off')
fig.tight_layout()
plt.savefig(model_manager.model_dir / 'pred-plot.png')
plt.cla()
plt.clf()

# Count total number of images predicted
model_row['N'] = len(y_valid_pred)
# Count the number of correct
n_correct = 0
for pred, correct in zip(y_valid_pred, y_valid):
    # Round properly...
    if pred // 1 + int((pred % 1) > 0.5) == correct:
        n_correct += 1
model_row['NCorrect'] = n_correct

# Save this model's data as a row in models_summary.csv
summary_df.loc[hex_hash] = model_row
summary_df.to_csv(summary_file)

# # Display some images that were incorrectly labeled
# # Print some images and their predicted number of shrimp
# fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
# ax_i = 0
# for i, img_array in enumerate(X_valid):
#     pred = y_valid_pred[i]
#     pred_round = round(y_valid_pred[i])
#     correct = y_valid[i]
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

# # Create histograms to compare validation dataset true labels with
# # the model's predictions
# fig, axs = plt.subplots(nrows=1, ncols=2)
# # Create y_valid histogram
# axs[0].hist(y_valid, bins=np.unique(y_valid))
# axs[0].set_xticks(np.unique(y_valid))
# axs[0].set_xlabel('Label Value')
# axs[0].set_ylabel('Frequency')
# axs[0].set_title(f'Distribution of True Labels for Validation Data')
# # Create y_valid_pred histogram
# axs[1].hist(y_valid_pred, bins=np.unique(y_valid))
# axs[1].set_xticks(np.unique(y_valid))
# axs[1].set_xlabel('Label Value')
# axs[1].set_ylabel('Frequency')
# axs[1].set_title(f'Distribution of Predicted Labels for Validation Data')
# plt.show()
