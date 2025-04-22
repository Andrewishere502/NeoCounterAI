import pathlib
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(2)


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
    img_labels = np.array(img_labels)
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
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError()
    ]
    return metrics


# Path to data
data_dir = pathlib.Path('DataNoSubstrate')
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')
# meta_df = meta_df[meta_df['Glare'] == 0]

# Load the image data
img_arrays, img_labels = get_img_data(meta_df)

# Weight less frequent labels as more important
label_weights = get_weights(img_labels)
# Print out the weights for each label
for label, weight in label_weights.items():
    print(f'{label}: {weight:.2f}x')


# Split data into a train, validation, and test set
img_indices = np.arange(len(img_arrays))
np.random.shuffle(img_indices)
index_partitions = np.split(img_indices, [int(0.7 * len(img_indices)),  # 0-70% for training
                                          int(0.9 * len(img_indices)),  # 70-90% for validation
                                          ])  # 90-100% for testing
X_train, X_valid, X_valid = img_arrays[index_partitions[0]], img_arrays[index_partitions[1]], img_arrays[index_partitions[2]]
y_train, y_valid, y_valid = img_labels[index_partitions[0]], img_labels[index_partitions[1]], img_labels[index_partitions[2]]
assert len(X_train) == len(y_train)
assert len(X_valid) == len(y_valid)
assert len(X_valid) == len(y_valid)


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


# Add dense layers on top to learn how to interpret
# the output of the convolutional layers
# model.add(tf.keras.layers.Dense(units=128,  # Number of neurons
#                                 activation='tanh',
#                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2),
#                                 bias_initializer='zeros'
#                                 ))
model.add(tf.keras.layers.Dense(units=1,  # Number of neurons
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2),
                                bias_initializer='zeros'
                                ))
# Take a look at what our model looks like so far
model.summary()


# Compile the model so it's ready for training
metrics = get_metrics()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=metrics
)
# Train the model
resnet_prep = tf.keras.applications.resnet.preprocess_input
csv_logger = tf.keras.callbacks.CSVLogger('training.log')
early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=2)  # After 2 epochs of <0.001 change, stop
history = model.fit(
    resnet_prep(X_train),
    y_train,
    validation_data=(resnet_prep(X_valid), y_valid),
    class_weight=label_weights,
    epochs=25,
    callbacks=[csv_logger, early_stop]
    )
# print(history.history)

# Let the model predict off of X_valid
y_valid_pred = model.predict(resnet_prep(X_valid))

# Print some images and their predicted number of shrimp
NROWS = 3
NCOLS = 5
fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(8, 6))
for i, img_array in enumerate(X_valid[:NROWS*NCOLS]):
    axs[i//5][i%5].imshow(X_valid[i])
    axs[i//5][i%5].set_title(f'{y_valid_pred[i][0]:.0f} | {y_valid[i]:.0f}')
    axs[i//5][i%5].axis('off')
plt.show()

# Create histograms to compare validation dataset true labels with
# the model's predictions
fig, axs = plt.subplots(nrows=1, ncols=2)
# Create y_valid histogram
axs[0].hist(y_valid)
axs[0].set_xlabel('Label Value')
axs[0].set_ylabel('Frequency')
axs[0].set_title(f'Distribution of True Labels for Validation Data')
# Create y_valid_pred histogram
axs[1].hist(y_valid_pred)
axs[1].set_xlabel('Label Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title(f'Distribution of Predicted Labels for Validation Data')
plt.show()
