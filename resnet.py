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
from tensorflow.keras.applications.resnet import preprocess_input

from settings import Settings, save_settings


# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(Settings.seed)


class MeanPowError(tf.keras.Loss):
    '''Raise the error to the specified power, and return that value.'''
    def __init__(self, p, **kwargs):
        '''
        
        Arguments:
        pow -- power to raise the error to
        '''
        super().__init__(self, **kwargs)

        self.p = p
        return
    
    def call(self, y_true, y_pred):
        '''...'''
        return tf.keras.ops.mean((y_true - y_pred) ** self.p)

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


def construct_model() -> tf.keras.models.Sequential:
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

    # Add the final dense layer to do the actual regression
    model.add(tf.keras.layers.Dense(units=1,  # Number of neurons
                                    activation='relu',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=Settings.seed),
                                    bias_initializer='zeros'
                                    ))
    return model


def compile_model(model: tf.keras.models.Sequential):
    # Compile the model so it's ready for training
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError(reduction='mean_with_sample_weight')
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=get_metrics()
    )
    return

# Path to data
data_dir = pathlib.Path(Settings.collection_name)
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')

# Load the image data
img_arrays, img_labels = get_img_data(meta_df)

# Weight less frequent labels as more important
label_weights = get_weights(img_labels, max_weight=Settings.max_weight)
# Print out the weights for each label
for label, weight in label_weights.items():
    print(f'{label}: {weight:.2f}x')

# Shuffle array of indices
img_indices = np.arange(len(img_arrays))
np.random.shuffle(img_indices)
# Split dataset into 80% training, 20% testing
data_partitions = np.split(img_indices, [int(0.80 * len(img_indices))])
X_train, X_test = img_arrays[data_partitions[0]], img_arrays[data_partitions[1]]
y_train, y_test = img_labels[data_partitions[0]], img_labels[data_partitions[1]]
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)


# Load the model architecture
model = construct_model()

# Compile the model
compile_model(model)

# Path for saving this the training log for this model
in_progress_dir = pathlib.Path('Models', 'InProgress')
if not in_progress_dir.exists():
    in_progress_dir.mkdir()
log_file = in_progress_dir / 'training.log'

# Training logger callback, saves information about each epoch
# as training progresses
csv_logger = tf.keras.callbacks.CSVLogger(log_file)

# Early stopping callback, terminating training if no progress is
# actually being made
early_stop = tf.keras.callbacks.EarlyStopping(min_delta=Settings.min_delta,
                                              patience=Settings.patience)

# Train the model
model.fit(preprocess_input(X_train),
          y_train,
          validation_split=Settings.validation_split,
          class_weight=label_weights,
          epochs=Settings.epochs,
          callbacks=[csv_logger, early_stop]
          )

# Now that the model has finished training, we can compute its hash
# which acts as a unique ID for this model's "brain"
model_hash = hex(hash(model))

# Save the model
model_dir = pathlib.Path('Models', model_hash)
# Ideally, this exact model has never been trained before because that
# would be a waste of time training replicate models. If for some
# reason this happened, add '-n' to the model_dir, where n indicates
# the model replicate number.
n = 1
while True:
    if model_dir.exists():
        model_dir = pathlib.Path('Models', f'{model_hash}-{n}')
        n += 1
    else:
        model_dir.mkdir()
        break
model_file = model_dir / 'model.keras'
model.save(model_file)

# Save the indices from meta_df that were partitioned into the
# training and testing sets.
partition_file = model_dir / 'data_partitions.txt'
with open(partition_file, 'w') as file:
    # Save training partition
    file.write(f'train_partition = {sorted(map(int, data_partitions[0]))}\n')
    # Save testing partition
    file.write(f'test_partition = {sorted(map(int, data_partitions[1]))}\n')

# Move the training.log file into the same directory as the model
log_file.rename(model_dir / log_file.name)

# Save the settings for this model in the same place the model was
# stored
settings_file = model_dir / 'settings.txt'
date_trained = datetime.datetime.now().strftime('%Y/%m/%d')
time_trained = datetime.datetime.now().strftime('%H:%M:%S')
save_settings(settings_file,
              Settings,
              date_trained=date_trained,
              time_trained=time_trained,
              model_hash=model_hash,
              loss_function=model.loss.name
              )


print(model_hash)
