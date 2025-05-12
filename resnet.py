'''Script to train a CNN model based on the ResNet50 architecture
(loaded with imagenet weights) to estimate the number of shrimp in a
240 x 240 RGB image. Images must be transformed using 
tf.keras.applications.resnet.preprocess_input
'''

import datetime
import pathlib
import json
from typing import Tuple, Dict, Any, List, Callable
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from scipy import stats

from settings import Settings, save_settings
from enhancements import crop_img, randcrop_img


# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(Settings.seed)


###
# Dataset functions
###

def get_img_data(meta_df: pd.DataFrame, cropping: bool, rand_crop_n: int, rand_crop_width: int, rand_crop_height:int) -> Tuple[np.ndarray, np.ndarray]:
    '''Return an array of each image as a 2D array of pixels, and an
    array of labels corresponding to each image. Before returning each
    image, crop it to remove the glare on the right of each image.

    Arguments:
    meta_df -- 
    cropping -- Whether or not to crop the image to remove glare from
                the right side
    rand_crop_n -- 
    rand_crop_width -- 
    rand_crop_height -- 
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

        # Location of shrimp in the original image
        shrimp_pos_str = labeled_row['ShrimpPos']
        # Convert shrimp pos to empty string if it's nan (float)
        if pd.isna(shrimp_pos_str):
            shrimp_pos_str = ''

        # Original image's number of shrimp
        n_shrimp = int(labeled_row['NShrimp'])

        # Crop image to remove glare if specified
        if cropping:
            # Crop the image to remove glare from the right
            new_width = 240
            new_height = 240
            img_array = crop_img(img_array, 0, 0, new_width, new_height)
            
            # Recount the shrimp in this image
            shrimp_pos = parse_shrimp_pos(shrimp_pos_str)
            n_shrimp = recount_shrimp(shrimp_pos, 0, 0, new_width, new_height)

        # Option to subset images randomly into multiple smaller images
        if rand_crop_n > 0:
            # Create rand_crop_n randomly cropped images from the img_array,
            # and add those to img_arrays instead. Also recount the shrimp
            # in the image to add that to img_labels.
            for _ in range(rand_crop_n):
                # Subset the image
                x_offset, y_offset, sub_img_array = randcrop_img(img_array, rand_crop_width, rand_crop_height)
                
                # Recount the shrimp in the image
                shrimp_pos = parse_shrimp_pos(shrimp_pos_str)
                n_shrimp = recount_shrimp(shrimp_pos, x_offset, y_offset, rand_crop_width, rand_crop_height)

                # Save the cropped image and its label
                img_arrays.append(sub_img_array)
                img_labels.append(n_shrimp)
        else:
            # Add the image array and its label to
            # their respective lists
            img_arrays.append(img_array)
            img_labels.append(n_shrimp)
        
        # Display progress bar
        print(f'\r{len(img_arrays)} of {len(meta_df)} loaded', end='')
    print()  # Just print a new line

    # Convert img_arrays and img_labels from lists to arrays
    img_arrays = np.array(img_arrays)
    img_labels = np.array(img_labels)
    return (img_arrays, img_labels)


def parse_shrimp_pos(shrimp_pos_str: str) -> List[Tuple[int, int]]:
    '''Parse the coordinates for a shrimp as in the ShrimpPos column.'''
    # If no shrimp positions, return an empty list
    if shrimp_pos_str == '':
        return []

    # Only parse positions now that there is at least one shrimp
    shrimp_positions = shrimp_pos_str[1:-1].split(')(')
    shrimp_positions = [tuple(map(int, pos.split(' '))) for pos in shrimp_positions]
    return shrimp_positions


def recount_shrimp(shrimp_pos: List[Tuple[int, int]], x_offset: int, y_offset: int, width: int, height: int):
    '''Count the number of shrimp that are within the new image.'''
    n_shrimp = 0
    for pos in shrimp_pos:
        x_in_img = (pos[0] >= x_offset and pos[0] < x_offset + width)
        y_in_img = (pos[1] >= y_offset and pos[1] < y_offset + height)
        if x_in_img and y_in_img:
            n_shrimp += 1
    return n_shrimp


def min_limit_freqs(img_arrays: np.ndarray, img_labels: np.ndarray) -> None:
    '''Remove images from img_arrays if they are part of a label group
    that does not occur at a minimum frequency within the dataset. This
    does not modify img_arrays in place! img_arrays seems to be
    protected from being modified inside this function
    
    Arguments:
    img_arrays -- An array of images
    img_labels -- An array of image labels, corresponding
                  to each index in img_arrays
    '''
    # How many images belong to each group
    img_freqs = Counter(np.array(img_labels))
    # Total number of images in the dataset
    total_imgs = sum(img_freqs.values())
    # Number of label groups
    n_groups = len(img_freqs.keys())
    # Expected proportion of images in a group if the data was
    # uniformly distributed
    exp_prop = 1 / n_groups
    for n_shrimp, img_count in img_freqs.items():
        # Calculate proportion images with this label
        img_prop = img_count / total_imgs
        # If this label is only represented by one-tenth the expected
        # images, then 
        if img_prop < exp_prop * 1/10:
            # Get indices for images which are *not* part of the
            # critically under-represented group
            keep_is = img_labels != n_shrimp
            # Filter to include only images that don't have this label
            img_arrays = img_arrays[keep_is]
            # Filter labels
            img_labels = img_labels[keep_is]
            print(f'\tRemoved {len(keep_is) - len(img_arrays)} images with label {n_shrimp}')
    return img_arrays, img_labels


def max_limit_freqs(meta_df: pd.DataFrame) -> pd.DataFrame:
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
    # Disable training of resnet layers
    resnet50.trainable = False
    model.add(resnet50)

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
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return


def fit_model(model: Sequential, callbacks, validation_split: float, epochs: int, max_weight: float):
    # Train the model
    model.fit(preprocess_input(X_train),
            y_train,
            validation_split=validation_split,
            class_weight=get_weights(img_labels, max_weight=max_weight),
            epochs=epochs,
            callbacks=callbacks
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


def isolate_block(model: Sequential, target_conv_id: str, target_block_id: str=None):
    # The first "layer" in our model is resent50, get it here
    resnet50 = model.layers[0]
    # First layer is input, last layer is avg pooling
    conv_layers = resnet50.layers[1:-1]
    for layer in conv_layers:
        layer_name_parts = layer.name.split('_')
        conv_id = None
        block_id = None
        other = None  # Catch all for stuff between
        layer_type = None
        if len(layer_name_parts) == 2:
            conv_id, layer_type = layer_name_parts
        elif len(layer_name_parts) >= 3:
            conv_id, block_id, *other, layer_type = layer_name_parts
        else:
            raise ValueError(f'Unexpected layer name {layer.name} with < 2 parts')

        # Switch on target layers, switch off other layers
        if conv_id == target_conv_id and (block_id == target_block_id or target_block_id == None):
            layer.trainable = True
        else:
            layer.trainable = False
    return


def true_round(value: float) -> int:
    '''Round to the nearest whole number (>=0.5 -> 1), NOT using bankers rounding
    like python's builtin round function.
    '''
    if value - int(value) >= 0.5:
        return int(value) + 1
    else:
        return int(value)


def make_img_grid(nrows: int, ncols: int, img_arrays: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Figure, Axes]:
    '''Create an axes showing a grid of images titled by their
    predicted number of shrimp and true number of shrimp.

    Arguments:
    nrows -- Number of rows in the image display grid
    ncols -- Number of columns in the image display grid
    img_arrays -- Randomly select nrows * ncols images
                  from this array
    y_true -- Correct NVS for each image in img_arrays
    y_pred -- Predicted NVS for each image in img_arrays
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)#, figsize=(2 * nrows, int(1.2 * ncols)))
    n_imgs = nrows * ncols
    img_is = np.random.choice(np.arange(n_imgs), size=n_imgs, replace=False)
    for ax_i, img_i in enumerate(img_is):
        if ax_i == len(img_arrays):
            break
        # Apply a filter inverting the image colors
        img = img_arrays[img_i]
        axs[ax_i//ncols][ax_i%ncols].imshow(img)
        axs[ax_i//ncols][ax_i%ncols].set_title(f'{y_pred[img_i]} | {y_true[img_i]}')
        axs[ax_i//ncols][ax_i%ncols].axis('off')
    fig.tight_layout()
    return fig, axs


def apply_img_filter(px_filter: Callable, img: np.ndarray) -> np.ndarray:
    '''Apply a per-pixel filter to the image and return it.

    Arguments:
    img_filter -- Filter that acts on individual pixels
    img -- Image to which to apply a filter
    '''
    filtered_image = np.apply_along_axis(px_filter, axis=2, arr=img)
    return filtered_image


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


# Limit image frequencies to what would be expected in a uniform
# distribution
if Settings.lim_max_prop:
    total_imgs = len(meta_df)
    meta_df = max_limit_freqs(meta_df)
    print(f'Dropped {total_imgs - len(meta_df)} images which were over represented')
    del total_imgs

# Load the image data
img_arrays, img_labels = get_img_data(meta_df,
                                      Settings.crop_glare,
                                      Settings.rand_crop_n,
                                      Settings.rand_crop_width,
                                      Settings.rand_crop_height)


# Remove NSVs with very few images, excluding under represented groups.
if Settings.lim_min_prop:
    img_arrays, img_labels = min_limit_freqs(img_arrays, img_labels)
    print(f'\tUsing {len(img_arrays)} images')


# Split data into training and testing sets, 80/20
X_train, X_test, y_train, y_test = train_test_split(img_arrays, img_labels, train_size=0.80)


# for img in X_train:
#     plt.imshow(img)
#     plt.show()


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
callbacks = [csv_logger, early_stop]

# model.summary()

fit_model(model,
          callbacks,
          Settings.validation_split,
          Settings.epochs,
          Settings.max_weight)


if Settings.retrain_conv:
    # Retrain the cnn layers in reverse order
    for conv_id in ['conv5', 'conv4', 'conv3', 'conv2', 'conv1']:
        print(f'Isolating conv block {conv_id}')
        # Isolate one conv layer at a time
        isolate_block(model, conv_id)
        fit_model(model,
                  callbacks,
                  Settings.validation_split,
                  Settings.epochs,
                  Settings.max_weight)


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
date_trained = datetime.datetime.now().strftime('%Y/%m/%d')
time_trained = datetime.datetime.now().strftime('%H:%M:%S')
save_settings(model_dir / 'settings.txt',
              Settings,
              date_trained=date_trained,
              time_trained=time_trained,
              model_hash=model_hash
              )


# Plot MSE to epoch
training_df = pd.read_csv(model_dir / 'training.log')
plt.plot(training_df['epoch'], training_df['mean_squared_error'])
plt.plot(training_df['epoch'], training_df['val_mean_squared_error'])
plt.xlabel('Epoch #')
plt.ylabel('MSE')
plt.xticks(range(training_df['epoch'].max() + 1))
plt.legend(['Training', 'Validation'])
plt.savefig(model_dir / 'mse-plot.png')
plt_clear()
del training_df  # Remove for memory


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
y_pred = model.predict(preprocess_input(X_test)).flatten()
# Round the model's predictions to the nearest whole number
y_pred = np.array(list(map(true_round, y_pred)))  # Avoid bankers rounding done by numpy, pandas, and Python itself

# Save a histogram of the model's predictions
plt.hist(y_pred, bins=np.unique(y_pred))
plt.xticks(np.unique(y_pred))
max_height = max(Counter(y_pred).values())
step = max(int(max_height / 3), 1)
plt.yticks(range(0, max_height+step, step))
plt.xlabel('Predicted Number of Visible Shrimp')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'pred-hist.png')
plt_clear()


# TODO: Create a 3x3 figure of correctly labeled images
correct_slice = y_test == y_pred
make_img_grid(3, 3, X_test[correct_slice], y_test[correct_slice], y_pred[correct_slice])
plt.savefig(model_dir / 'imgs-correct.png')
plt_clear()

# TODO: Create a 3x3 figure of overpredicted labeled images
under_slice = y_test < y_pred
make_img_grid(3, 3, X_test[under_slice], y_test[under_slice], y_pred[under_slice])
plt.savefig(model_dir / 'imgs-over.png')
plt_clear()

# TODO: Create a 3x3 figure of underpredicted labeled images
over_slice = y_test > y_pred
make_img_grid(3, 3, X_test[over_slice], y_test[over_slice], y_pred[over_slice])
plt.savefig(model_dir / 'imgs-under.png')
plt_clear()


# Calculate by how much each guess is off, then count the frequencies
# of these differences
y_err = y_pred - y_test
err_bins = [-3, -2, -1, 0, 1, 2, 3]
# Save a histogram of the model's prediction errors
plt.hist(y_err, bins=err_bins)
plt.xticks(err_bins)
max_height = max(Counter(y_err).values())
step = max(int(max_height / 3), 1)
plt.yticks(range(0,max_height+step,step))
plt.xlabel('Additional Number of Visible Shrimp Predicted')
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
avg_err = [sum(errs)/len(errs) for errs in y_err_by_true.values()]
se_err = [stats.sem(errs) for errs in y_err_by_true.values()]
plt.bar(y_err_by_true.keys(), avg_err)
# Add error bars using SE
plt.errorbar(y_err_by_true.keys(), avg_err, yerr=se_err, fmt='.', color='r', capsize=8)
# Label bars with number of predictions represented
for true_nvs, errs in y_err_by_true.items():
    avg = sum(errs)/len(errs)
    plt.text(true_nvs-0.1, 0.2 if avg < 0 else -0.2, f'N={len(errs)}')
plt.xlabel('Number of Visible Shrimp')
plt.ylabel('Additional NVS Predicted')
plt.xticks(np.unique(list(y_err_by_true.keys())))
plt.yticks(err_bins)
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


####
# Conduct some stats test
####

stats_file = model_dir / 'stats.txt'

# Good old fashioned accuracy, although this is an odd metric for a
# regression
with open(stats_file, 'a') as file:
    file.write('Dataset Descriptive Stats:\n')
    file.write(f'Mean {np.mean(img_labels)}\n')
    file.write(f'Median {np.median(img_labels)}\n')
    file.write(f'STD {np.std(img_labels)}\n')
    file.write(f'SE {stats.sem(img_labels)}\n')

    # Conduct shapiro-wilk's test
    shap_results = stats.shapiro(img_labels)
    file.write(f'Shapiro test; statistic = {shap_results.statistic}, p-value = {shap_results.pvalue}\n')
    file.write(f'Normally distributed: {shap_results.pvalue >= 0.05}\n\n')

    # Good old fashioned accuracy, although this is an odd metric for a
    # regression
    file.write('Accuracy:\n')
    file.write(f'{sum(y_err == 0)} / {len(y_err)} = {sum(y_err == 0) / len(y_err)}\n\n')

    # Paired t-test to determine if predicted n shrimp is statistically
    # different from true n shrimp, on average
    ttest_result = stats.ttest_rel(y_pred, y_test)
    paired_t_p = ttest_result.pvalue
    file.write('Paired t-test assuming equal variance:\n')
    file.write(f'Paired t-test; pvalue = {paired_t_p:.2e}, df = {ttest_result.df}\n\n')

    # ANOVA to see if any image labels are predicted differently than
    # others, on average
    anova_result = stats.f_oneway(*y_err_by_true.values())
    anova_stat = anova_result.statistic
    anova_p = anova_result.pvalue
    file.write(f'ANOVA; statistic = {anova_stat:.2e}, pvalue = {anova_p:.2e}\n\n')

    # Regression to determine if there is a significant relationship
    # (positive or negative) between the true NVS in an image and the
    # model's prediction error. If significant, this would suggest the
    # model is specializing in identifying some NVS and not generalizing
    # as well.
    reg_result = stats.linregress(y_test, y_err)
    reg_p = reg_result.pvalue
    reg_m = reg_result.slope
    reg_b = reg_result.intercept
    file.write('Least-squares regression:\n')
    file.write(f'Linear regression; pvalue = {reg_p:.2e}, slope = {reg_m:.2e}, intercept = {reg_b:.2e}\n\n')
