'''Script to train a CNN model based on the ResNet50 architecture
(loaded with imagenet weights) to estimate the number of shrimp in a
240 x 240 RGB image. Images must be transformed using 
tf.keras.applications.resnet.preprocess_input
'''

import datetime
import pathlib
import json
from typing import Tuple, Dict, Any
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from scipy import stats

from settings import Settings, save_settings
from enhancements import crop_img


# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(Settings.seed)


###
# Dataset functions
###

def get_img_data(meta_df: pd.DataFrame, cropping: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''Return an array of each image as a 2D array of pixels, and an
    array of labels corresponding to each image. Before returning each
    image, crop it to remove the glare on the right of each image.

    Arguments:
    meta_df -- 
    cropping -- Whether or not to crop the image to remove glare from
                the right side
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

        if cropping:
            # Crop the image to remove glare from the right
            new_width = 240
            new_height = 240
            img_array = crop_img(img_array, 0, new_height-1, 0, new_width-1)
            
            # Recount the shrimp in this image
            shrimp_positions = labeled_row['ShrimpPos'][1:-1].split(')(')
            shrimp_positions = [tuple(map(int, pos.split(' '))) for pos in shrimp_positions]
            n_shrimp = 0
            for pos in shrimp_positions:
                if pos[0] < new_width and pos[1] < new_height:
                    n_shrimp += 1
            img_labels.append(n_shrimp)
        else:
            # Don't modify the image shape
            img_labels.append(labeled_row['NShrimp'])

        # Add the image array and its label to
        # their respective lists
        img_arrays.append(img_array)
        
        # Display progress bar
        print(f'\r{len(img_arrays)} of {len(meta_df)} loaded', end='')
    print()

    # Convert img_arrays and img_labels from lists to arrays
    img_arrays = np.array(img_arrays)
    img_labels = np.array(img_labels, dtype=int)
    return (img_arrays, img_labels)


def max_limit_freqs(meta_df: pd.DataFrame) -> None:
    '''Return a copy of the dataframe with ...
    
    Arguments:
    meta_df -- 
    '''
    img_freqs = Counter(np.array(meta_df['NShrimp']))
    img_total = sum(img_freqs.values())
    n_shrimp_groups = len(img_freqs.keys())
    exp_prop = 1 / n_shrimp_groups
    all_rm_is = []
    for n_shrimp, img_count in img_freqs.items():
        img_prop = img_count / img_total
        # Drop images that occur more than expected in a uniform
        # distribution of this size
        if img_prop > exp_prop:
            # Calculate the number of images to remove
            rm_imgs = int(img_count - img_total * exp_prop)
            # Get indices for images in this group
            img_group_is = meta_df[meta_df['NShrimp'] == n_shrimp].index
            # Choose which indices to remove
            rm_is = np.random.choice(img_group_is, size=(rm_imgs,))
            # Extend list of indices to drop
            all_rm_is.extend(rm_is)
    return meta_df.drop(index=all_rm_is)


def min_limit_freqs(meta_df: pd.DataFrame) -> None:
    '''Return a copy of the dataframe with ...
    
    Arguments:
    meta_df -- 
    '''
    img_freqs = Counter(np.array(meta_df['NShrimp']))
    img_total = sum(img_freqs.values())
    n_shrimp_groups = len(img_freqs.keys())
    exp_prop = 1 / n_shrimp_groups
    all_rm_is = []
    for n_shrimp, img_count in img_freqs.items():
        img_prop = img_count / img_total
        # Drop images that occur less than a tenth of the expect proportion
        if img_prop < exp_prop * 1/10:
            rm_is = meta_df[meta_df['NShrimp'] == n_shrimp].index
            all_rm_is.extend(rm_is)
    return meta_df.drop(index=all_rm_is)

###
# End dataset functions
###


def construct_model(input_shape: Tuple[int, int, int]) -> Sequential:
    # Load in the ResNet50 layers with imagenet weights,
    # and a different input dimensionality
    model = Sequential()
    resnet50 = ResNet50(input_shape=input_shape,
                        weights='imagenet',
                        include_top=False,
                        pooling='avg'
                        )
    resnet50.trainable = False
    model.add(resnet50)
    # Make resnet layers untrainable
    # for layer in model.layers:
    #     layer.trainable = False
    model.summary()

    # Add the final dense layer to do the actual regression
    model.add(Dense(units=1,  # Number of neurons
              activation='relu',
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=Settings.seed),
              bias_initializer='zeros'
              ))
    return model


def compile_model(model: Sequential) -> None:
    # Compile the model so it's ready for training
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError(reduction='mean_with_sample_weight')
    metrics = [
        tf.keras.metrics.MeanSquaredError()
        ]
    # loss = MeanPowError(6)  # x ** 6
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return


def get_weights(array: np.ndarray[Any], max_weight: float=None) -> Dict[Any, np.float32]:
    '''Return weights for all unique values within an array as a
    dict, where the key is the unique value and the value (paired with
    the key) is the frequency of that unique value divided by the max
    frequency for any unique value.
    
    Arguments:
    array -- An array of values
    '''
    weights = {}
    freqs = Counter(array)
    max_freq = max(freqs.values())
    for value, freq in freqs.items():
        # Potentially put an upper limit on the weight for a class
        if max_weight == None:
            weights[value] = max_freq / freq
        else:
            weights[value] = min(max_freq / freq, max_weight)
    return weights


# def unfreeze_blocks(model: Sequential, target_conv_id: str=None, target_block_id: str=None):
#     # The first "layer" in our model is resent50, get it here
#     resnet50 = model.layers[0]
#     # First layer is input, last layer is avg pooling
#     conv_layers = resnet50[1:-1].layers
#     for layer in conv_layers:
#         layer_name_parts = layer.name.split('_')
#         conv_id = None
#         block_id = None
#         other = None  # Catch all for stuff between
#         layer_type = None
#         if len(layer_name_parts) == 2:
#             conv_id, layer_type = layer_name_parts
#         elif len(layer_name_parts) >= 3:
#             conv_id, block_id, *other, layer_type = layer_name_parts
#         else:
#             raise ValueError(f'Unexpected layer name {layer.name} with < 2 parts')

#         # Switch on target layers, switch off other layers
#         if conv_id == target_conv_id and block_id == target_block_id:
#             layer.trainable = True
#         else:
#             layer.trainable = False
#     return


def true_round(value: float) -> int:
    '''Round to the nearest whole number (>0.5 -> 1), NOT using bankers rounding
    like python's builtin round function.
    '''
    return value // 1 + int((value % 1) > 0.5)


def plt_clear() -> None:
    '''Clear the axis and figure.'''
    plt.cla()
    plt.clf()
    plt.close('all')
    return


# Load in the meta data for the dataset/collection 
data_dir = pathlib.Path(Settings.collection_name)
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')

# Remove NSVs with very few images
if Settings.lim_min_prop:
    total_imgs = len(meta_df)
    meta_df = min_limit_freqs(meta_df)
    print(f'Dropped {total_imgs - len(meta_df)} images which were under represented')
    del total_imgs

# Limit image frequencies to what would be expected in a uniform
# distribution
if Settings.lim_max_prop:
    total_imgs = len(meta_df)
    meta_df = max_limit_freqs(meta_df)
    print(f'Dropped {total_imgs - len(meta_df)} images which were over represented')
    del total_imgs

# Load the image data
img_arrays, img_labels = get_img_data(meta_df, cropping=Settings.crop_glare)

# Split data into training and testing sets, 80/20
X_train, X_test, y_train, y_test = train_test_split(img_arrays, img_labels, train_size=0.80)

# Load the model architecture
input_shape = img_arrays[0].shape
model = construct_model(input_shape)

# Compile the model
compile_model(model)


# Path for saving this the training log for this model
model_dir = pathlib.Path('Models', 'InProgress')
if model_dir.exists():
    raise ValueError('Error: InProgress dir already exists.')
else:
    model_dir.mkdir()

# Save the model's compile configuration
with open(model_dir / 'compile_config.json', 'w') as file:
    config = model.get_compile_config()
    file.writelines(json.dumps(config))


# Training logger callback, saves information about each epoch
# as training progresses
csv_logger = CSVLogger(model_dir / 'training.log')

# Early stopping callback, terminating training if no progress is
# actually being made
early_stop = EarlyStopping(min_delta=Settings.min_delta,
                           patience=Settings.patience,
                           restore_best_weights=Settings.restore_best_weights)

# Train the model
model.fit(preprocess_input(X_train),
          y_train,
          validation_split=Settings.validation_split,
          class_weight=get_weights(img_labels, max_weight=Settings.max_weight),
          epochs=Settings.epochs,
          callbacks=[csv_logger, early_stop]
          )

# Now that the model has finished training, we can compute its hash
# which acts as a unique ID for this model's "brain"
model_hash = hex(hash(model))
print(model_hash)

# Rename the directory used for storing files associated with this
# model to the hash of this model.
model_dir = model_dir.rename(str(pathlib.Path(*model_dir.parts[:-1], model_hash)))

# Save the model
model_file = model_dir / 'model.keras'
model.save(model_file)

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


# Save a histogram of the data the model was trained on
plt.hist(y_train, bins=np.unique(y_train))
plt.xlabel('Number of Visible Shrimp')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'train-nvs-hist.png')
plt_clear()


# Save a histogram of the data the model will be tested on
plt.hist(y_test, bins=np.unique(y_test))
plt.xlabel('Number of Visible Shrimp')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'test-nvs-hist.png')
plt_clear()


# Get the model's predictions on X_test
y_pred = np.round(model.predict(preprocess_input(X_test))).flatten()
# Save a histogram of the model's predictions
plt.hist(y_pred, bins=np.unique(y_pred))
plt.xticks(np.unique(y_pred))
max_height = max(Counter(y_pred).values())
step = max(int(max_height / 3), 1)
plt.yticks(range(0,max_height+step,step))
plt.xlabel('Predicted Number of Visible Shrimp')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'pred-hist.png')
plt_clear()


# Calculate by how much each guess is off, then count the frequencies
# of these differences
y_err = y_pred - y_test
# Save a histogram of the model's prediction errors
pred_err_bins = [-3, -2, -1, 0, 1, 2, 3]
plt.hist(y_err, bins=pred_err_bins)
plt.xticks(pred_err_bins)
max_height = max(Counter(y_err).values())
step = max(int(max_height / 3), 1)
plt.yticks(range(0,max_height+step,step))
plt.xlabel('Predicted Number of Visible Shrimp')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'pred-err-hist.png')
plt_clear()


# Get the models predictions and prediction errors, grouped by the true
# number of shrimp for each prediction
y_pred_by_true = {}
y_err_by_true = {}
for usc in np.unique(y_test):
    # Get all predicted values for images with this many shrimp in them
    y_pred_by_true[usc] = y_pred[y_test == usc]
    # Get all prediction errors for images with this many shrimp in them
    y_err_by_true[usc] = y_err[y_test == usc]


# Histogram of average prediction error for each image label
avg_diff = [sum(errs)/len(errs) for errs in y_err_by_true.values()]
se_diff = [stats.sem(errs) for errs in y_err_by_true.values()]
plt.bar(y_err_by_true.keys(), avg_diff)
# Add error bars using SE
plt.errorbar(y_err_by_true.keys(), avg_diff, yerr=se_diff, fmt='.', color='r', capsize=8)
# Label bars with number of predictions represented
for true_nvs, errs in y_err_by_true.items():
    avg = sum(errs)/len(errs)
    plt.text(true_nvs-0.1, 0.2 if avg < 0 else -0.2, f'N={len(errs)}')
plt.xlabel('Number of Visible Shrimp')
plt.ylabel('Additional NVS Predicted')
plt.xticks(np.unique(y_train))
plt.tight_layout()
plt.savefig(model_dir / f'avg-pred-err.png')
plt_clear()


# Violin plot of predictions for images grouped by number of shrimp
# in image
plt.violinplot(y_pred_by_true.values(), y_pred_by_true.keys())
plt.plot(y_pred_by_true.keys(), y_pred_by_true.keys(), linestyle='dashed')
plt.xlabel('Number of Visible Shrimp')
plt.ylabel('Predicted Number of Visible Shrimp')
plt.xticks(list(y_pred_by_true.keys()))
plt.savefig(model_dir / 'pred-violin.png')
plt_clear()
