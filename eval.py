'''Script to evaluate a given CNN model.'''

import pathlib
from typing import Tuple
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats as stats

from model_manager import ModelManager


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
    for i, img_array in enumerate(X_img[:nrows*ncols]):
        axs[i//ncols][i%ncols].imshow(img_array)
        axs[i//ncols][i%ncols].set_title(f'{y_pred[i]:.0f} ({y_pred[i]:.2f}) | {y_true[i]:.0f}')
        axs[i//ncols][i%ncols].axis('off')
    fig.tight_layout()
    return fig, axs


def create_nshrimp_hist(meta_df, partition=None) -> Tuple[Figure, Axes]:
    '''Create a histogram of NShrimp labels for all images or a subset.
    Return the figure and axis of the histogram.
    '''
    # If partition was specified, subset the data before histogram
    if partition == None or partition == 'all':
        hist_data = meta_df
    else:
        hist_data = meta_df.loc[model_manager.load_partition(partition)]

    fig, ax = plt.subplots()
    # Create histogram showing distribution of images with various
    # NShrimp labels within the train partition
    ax.hist(hist_data['NShrimp'], bins=np.unique(hist_data['NShrimp']))
    ax.set_xlabel('Shrimp Counted')
    ax.set_ylabel('Number of Images')
    fig.tight_layout()
    return fig, ax


# Path to file to write a summary file for all the models
summary_file = pathlib.Path('Models', 'models_summary.csv')

# Identify all the models (i.e. model hashes) to assess
hex_hashes = [
    # Settings: epoch 1, max_weight 1, DataNoSubstrate
    '0x104f0ddc',
    # Settings: epoch 5, max_weight 1, DataNoSubstrate
    '0x1086addc',
    # Settings: epoch 10, max_weight 1, DataNoSubstrate
    '0x105099dc',
    # Settings: epoch 20, max_weight 1, DataNoSubstrate
    '0x1068a1dc',
    # Settings: epoch 50, max_weight 1, DataNoSubstrate
    '0x108a15dc',
    # Settings: epoch 50, max_weight None, DataNoSubstrate
    '0x110a30e5',
    # Settings: epoch 100, max_weight 1, DataNoSubstrate, 0.001 limit
    '0x108e55dc'
]

# Image processor necesasry for ResNet50 with imagenet weights
resnet_prep = tf.keras.applications.resnet.preprocess_input

# Assess each model
for hex_hash in hex_hashes:
    print(f'Loading model {hex_hash}')
    model_manager = ModelManager(hex_hash)

    # Create a row to represent this model in the summary dataframe
    model_row = OrderedDict()

    # Store some settings to be written to summary_df
    model_row['Epochs'] = model_manager.get_setting('epochs')
    model_row['MaxWeight'] = model_manager.get_setting('max_weight')
    model_row['CollectionName'] = model_manager.get_setting('collection_name')

    # Path to this model's data
    meta_df = pd.read_csv(model_manager.meta_file, index_col='ID')

    # Create histograms visualizing distribution of nshrimp labels
    # amongst all images and some subsets
    partitions = ['all', 'train', 'valid', 'test']
    for partition in partitions:
        fig, ax = create_nshrimp_hist(meta_df)
        fig.savefig(model_manager.model_dir / f'{partition}-nshrimp-hist.png')
    plt.close()  # Close all figures at once

    # Read the indices in meta_df that are included for this partition
    # of the data
    partition_name = 'test'
    partition_is = model_manager.load_partition(partition_name)
    # Load the images from this partition and their labels
    X_img, y_true = get_img_data(meta_df.loc[partition_is])

    # Let the model predict off of X_img. Don't forget to transform the
    # images
    y_pred = model_manager.model.predict(resnet_prep(X_img)).flatten()
    y_pred_round = np.array(list(map(true_round, y_pred)))

    # Calculate by how much each guess is off, then count the frequencies
    # of these differences
    y_pred_difs = y_pred_round - y_true
    diff_count = dict(Counter(y_pred_difs))

    # Array of unique shrimp counts
    unique_shrimp_counts = np.unique(y_true)
    color_scale = []
    color_inc = 0.8 / len(unique_shrimp_counts)
    for i, _ in enumerate(unique_shrimp_counts):
        # Decrement r and g as i increases to make higher-shrimp-count
        # bars more blue
        color = (0.8 - color_inc * i, 0.8 - color_inc * i, 0.8)
        color_scale.append(color)

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
    ttest_result = stats.ttest_rel(y_pred, y_true)
    paired_t_p = ttest_result.pvalue
    print(f'Paired t-test; pvalue = {paired_t_p:.2e}, df = {ttest_result.df}')
    model_row['PredVsTrueP'] = paired_t_p


    # ANOVA to see if any image labels are predicted differently than
    # others, on average
    anova_result = stats.f_oneway(*y_pred_diffs_by_true.values())
    anova_p = anova_result.pvalue
    print(f'ANOVA; pvalue = {anova_p:.2e}')
    model_row['PredDiffByTrueANOVAP'] = anova_p


    # Regression to determine if there is a significant relationship
    # (positive or negative) between the true number of shrimp visible in
    # an image and the model's prediction error. If significant, this would
    # suggest the model is specializing in identifying 4 shrimp in an image
    # and not generalizing as well.
    reg_result = stats.linregress(y_pred_difs, y_true)
    reg_p = reg_result.pvalue
    reg_m = reg_result.slope
    reg_b = reg_result.intercept
    print(f'Linear regression; pvalue = {reg_p:.2e}, slope = {reg_m:.2e}, intercept = {reg_b:.2e}')
    model_row['PredDiffByTrueRegP'] = reg_p
    model_row['PredDiffByTrueRegM'] = reg_m
    model_row['PredDiffByTrueRegB'] = reg_b


    # Count total number of images predicted
    model_row['N'] = len(y_pred)
    # Count the number of correct
    n_correct = 0
    for pred, correct in zip(y_pred, y_true):
        # Round properly...
        if true_round(pred) == correct:
            n_correct += 1
    model_row['NCorrect'] = n_correct


    # Print some images and their predicted number of shrimp
    NROWS = 4
    NCOLS = 5
    fig, axs = create_pred_fig(NROWS, NCOLS)
    fig.savefig(model_manager.model_dir / f'{partition_name}-pred-plot.png')
    plt.close()


    # Make a histogram like this so we avoid fiddling with bins
    plt.bar(diff_count.keys(), diff_count.values(), width=0.9)
    # Label the bars by their height
    for key, value in diff_count.items():
        plt.text(key, value, value)
    plt.xlabel('Additional Shrimp Predicted')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(model_manager.model_dir / f'{partition_name}-pred-diffs.png')
    plt.close()


    # Histogram of average prediction error for each image label
    avg_diff = [sum(diffs)/len(diffs) for diffs in y_pred_diffs_by_true.values()]
    se_diff = [stats.sem(diffs) for diffs in y_pred_diffs_by_true.values()]
    plt.bar(y_pred_diffs_by_true.keys(), avg_diff)#, color=color_scale)
    # Add error bars using SE
    plt.errorbar(y_pred_diffs_by_true.keys(), avg_diff, yerr=se_diff, fmt='.', color='r', capsize=8)
    # Label bars with number of predictions represented
    for key, diffs in y_pred_diffs_by_true.items():
        avg = sum(diffs)/len(diffs)
        plt.text(key-0.1, 0.2 if avg < 0 else -0.2, f'N={len(diffs)}')
    plt.xlabel('True Shrimp Count')
    plt.ylabel('Additional Shrimp Predicted')
    plt.xticks(unique_shrimp_counts)
    plt.tight_layout()
    plt.savefig(model_manager.model_dir / f'{partition_name}-avg-pred-diffs.png')
    plt.close()


    # Violin plot of predictions for images grouped by number of shrimp
    # in image
    plt.violinplot(y_pred_by_true.values(), y_pred_by_true.keys())
    plt.plot(unique_shrimp_counts, unique_shrimp_counts, linestyle='dashed')
    plt.xlabel('True Shrimp Count')
    plt.ylabel('Predicted Shrimp Count')
    plt.xticks(unique_shrimp_counts)
    plt.savefig(model_manager.model_dir / f'{partition_name}-pred-violin.png')
    plt.close()


    # Violin plot of how much predictions differ from truth for images
    # grouped by number of shrimp in image
    plt.violinplot(y_pred_diffs_by_true.values(), y_pred_diffs_by_true.keys())
    plt.plot(unique_shrimp_counts, [0] * len(unique_shrimp_counts), linestyle='dashed')
    plt.xlabel('True Shrimp Count')
    plt.ylabel('Additional Shrimp Predicted')
    plt.xticks(unique_shrimp_counts)
    plt.savefig(model_manager.model_dir / f'{partition_name}-pred-diff-violin.png')
    plt.close()


    # # Correlate true values and with the difference 
    # # When regression line is above y=0 we are over predicting, when the
    # # regression line is below y=0 we are under predicting.
    # plt.plot(y_true, y_pred_difs, marker='o', linestyle='')
    # m, b = np.polyfit(y_true, y_pred_difs, 1)
    # plt.plot(y_true, m*y_true+b)  # Plot regression
    # plt.plot(y_true, 0*y_true)  # Plot y = 0
    # plt.show()


    # Load in the summary csv file as a dataframe
    summary_df = pd.read_csv(summary_file, index_col='HexHash')

    # Ensure all stats added to the model row are valid columns in the
    # summary dataframe
    for col_name in model_row.keys():
        if col_name not in summary_df.columns:
            raise ValueError(f'Invalid column {col_name}, not found in summary file header')

    # Save this model's data as a row in models_summary.csv
    summary_df.loc[hex_hash] = model_row
    summary_df.to_csv(summary_file)

    # # Display some images that were incorrectly labeled
    # # Print some images and their predicted number of shrimp
    # fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2 * NROWS, int(1.2 * NCOLS)))
    # ax_i = 0
    # for i, img_array in enumerate(X_img):
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
