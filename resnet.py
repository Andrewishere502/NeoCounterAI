'''Script to train a CNN model based on the ResNet50 architecture
(loaded with imagenet weights) to estimate the number of shrimp in a
240 x 320 RGB image. Images must be transformed using 
tf.keras.applications.resnet.preprocess_input
'''

import datetime
import pathlib
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from settings import Settings, save_settings


# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(Settings.seed)


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


def get_frequencies(array: np.ndarray[Any]) -> Dict[Any, np.float32]:
    '''Return the frequency of all unique values within an array as a
    dict, where the key is the unique value and the value (paired with
    the key) is the frequency of that unique value.
    
    Arguments:
    array -- An array of values
    '''
    freqs = {}
    for unique_value in np.unique(array):
        # Add this unique value to the dataframe
        freqs[unique_value] = np.sum(array == unique_value)
    return freqs


def get_weights(array: np.ndarray[Any], max_weight: float=None) -> Dict[Any, np.float32]:
    '''Return weights for all unique values within an array as a
    dict, where the key is the unique value and the value (paired with
    the key) is the frequency of that unique value divided by the max
    frequency for any unique value.
    
    Arguments:
    array -- An array of values
    '''
    weights = {}
    freqs = get_frequencies(array)
    max_freq = max(freqs.values())
    for value, freq in freqs.items():
        # Potentially put an upper limit on the weight for a class
        if max_weight == None:
            weights[value] = max_freq / freq
        else:
            weights[value] = min(max_freq / freq, max_weight)
    return weights


def get_metrics() -> List[tf.keras.Metric]:
    '''Return a list of Metric objects.'''
    metrics = [
        tf.keras.metrics.MeanSquaredError(),
        # tf.keras.metrics.RootMeanSquaredError(),
        # tf.keras.metrics.MeanAbsoluteError(),
    ]
    return metrics


# Path to data
data_dir = pathlib.Path(Settings.collection_name)
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')
# meta_df = meta_df[meta_df['Glare'] == 0]

# Path for saving this model and its info
datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_dir = pathlib.Path('Models', f'{datestr}/')
if not model_dir.exists():
    model_dir.mkdir()
model_file = model_dir / 'model.keras'
log_file = model_dir / 'training.log'
settings_file = pathlib.Path(model_dir / 'settings.txt')

# Load the image data
img_arrays, img_labels = get_img_data(meta_df)

# Weight less frequent labels as more important
label_weights = get_weights(img_labels, max_weight=Settings.max_weight)
# Print out the weights for each label
for label, weight in label_weights.items():
    print(f'{label}: {weight:.2f}x')

# Split data into a train, validation, and test set
img_indices = np.arange(len(img_arrays))
np.random.shuffle(img_indices)
index_partitions = np.split(img_indices, [int(0.7 * len(img_indices)),  # 0-70% for training
                                          int(0.9 * len(img_indices)),  # 70-90% for validation
                                          ])  # 90-100% for testing
# print(index_partitions[1])  # Print indices of images that are in the validation set
X_train, X_valid, X_test = img_arrays[index_partitions[0]], img_arrays[index_partitions[1]], img_arrays[index_partitions[2]]
y_train, y_valid, y_test = img_labels[index_partitions[0]], img_labels[index_partitions[1]], img_labels[index_partitions[2]]
print(len(X_train), len(X_valid), len(X_test))
assert len(X_train) == len(y_train)
assert len(X_valid) == len(y_valid)
assert len(X_test) == len(y_test)


# Load in the ResNet50 layers with imagenet weights,
# and a different input dimensionality
model = tf.keras.models.Sequential()
model.add(tf.keras.applications.ResNet50(input_shape=(240, 320, 3),
                                         weights='imagenet',
                                         include_top=False,
                                         pooling='avg'))
# Make resnet layers untrainable
for layer in model.layers:
    layer.trainable = False
# Take a look at what our model looks like so far
model.summary()

# Add dense layer on top to learn how to interpret the output of the
# convolutional layers
for units in Settings.dense_layers:
    model.add(tf.keras.layers.Dense(units=units,  # Number of neurons
                                    activation='tanh',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=Settings.seed),
                                    bias_initializer='zeros'
                                    ))
# Add the final dense layer to do the actual regression
model.add(tf.keras.layers.Dense(units=1,  # Number of neurons
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=Settings.seed),
                                bias_initializer='zeros'
                                ))
# Take a look at what our model looks like so far
model.summary()

# Compile the model so it's ready for training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(reduction='mean_with_sample_weight'),
    metrics=get_metrics()
)
# Train the model
resnet_prep = tf.keras.applications.resnet.preprocess_input
csv_logger = tf.keras.callbacks.CSVLogger(log_file)
early_stop = tf.keras.callbacks.EarlyStopping(min_delta=Settings.min_delta,
                                              patience=Settings.patience)
history = model.fit(resnet_prep(X_train),
                    y_train,
                    validation_data=(resnet_prep(X_valid), y_valid),
                    class_weight=label_weights,
                    epochs=Settings.epochs,
                    callbacks=[csv_logger, early_stop]
                    )

# Save the model
model.save(model_file)

# Calculate the hash of this model
model_hash = hash(model)

# Save the settings for this model, and also save its hash in hex
save_settings(settings_file, Settings, model_hash=hex(model_hash))

# Print out the hash for this model, unique to the result of training
print('Model hash:', hex(hash(model)))
