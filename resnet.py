import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Set random seeds for numpy, python, and keras backend
tf.keras.utils.set_random_seed(2)


# Path to data
data_dir = pathlib.Path('DataNoSubstrate')
meta_file = data_dir / 'metadata.csv'
meta_df = pd.read_csv(meta_file, index_col='ID')
# meta_df = meta_df[meta_df['Glare'] == 0]

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

# Calculate frequencies for each label
label_frequencies = {label: np.sum(img_labels == label) for label in np.unique(img_labels)}
# Weight less frequent labels as more important
class_weight = {}
max_frequency = max(label_frequencies.values())
for label in np.unique(img_labels):
    # Cap weight so some aren't over-weighted
    class_weight[label] = max_frequency / label_frequencies[label]
    # class_weight[label] = min(max_frequency / label_frequencies[label], 40)
for label, weight in class_weight.items():
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
model.add(tf.keras.layers.Dense(units=128,  # Number of neurons
                                activation='tanh',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2),
                                bias_initializer='zeros'
                                ))
model.add(tf.keras.layers.Dense(units=1,  # Number of neurons
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2),
                                bias_initializer='zeros'
                                ))
# Take a look at what our model looks like so far
model.summary()


# Compile the model so it's ready for training
metrics = [
    tf.keras.metrics.MeanSquaredError(),
    tf.keras.metrics.RootMeanSquaredError(),
    tf.keras.metrics.MeanAbsoluteError()
]
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=metrics
)
# Train the model
resnet_prep = tf.keras.applications.resnet.preprocess_input
csv_logger = tf.keras.callbacks.CSVLogger('training.log')
early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.025, patience=2)  # After 2 epochs of <0.025 change, stop
history = model.fit(
    resnet_prep(X_train),
    resnet_prep(y_train),
    validation_data=(resnet_prep(X_valid), resnet_prep(y_valid)),
    class_weight=class_weight,
    epochs=25,
    callbacks=[csv_logger, early_stop]
    )
# print(history.history)

# Let the model predict off of X_valid
y_valid_pred = model.predict(X_valid)

# Print some images and their predicted number of shrimp
NROWS = 3
NCOLS = 5
fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS)
for i, img_array in enumerate(X_valid[:NROWS*NCOLS]):
    axs[i//5][i%5].imshow(X_valid[i])
    axs[i//5][i%5].set_title(f'{y_valid_pred[i][0]:.0f} | {y_valid[i]:.0f}')
    axs[i//5][i%5].axis('off')
plt.show()


# # Print some images where the predicted number of shrimp was
# # NOT 3 or 4
# fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS)
# ax_i = 0
# for img_i, img_array in enumerate(X_valid):
#     if int(y_valid_pred[img_i][0]) < 3 or int(y_valid_pred[img_i][0]) > 4:
#         axs[ax_i//5][ax_i%5].imshow(X_valid[img_i])
#         axs[ax_i//5][ax_i%5].set_title(f'{y_valid_pred[img_i][0]:.0f} | {y_valid[img_i]:.0f}')
#         axs[ax_i//5][ax_i%5].axis('off')
#         ax_i += 1
#         if ax_i == NROWS * NCOLS:
#             break
# # Only show the figure if axes have been filled
# if ax_i != 0:
#     plt.show()
# else:
#     plt.cla()
#     plt.clf()

# print('Number of predictions less than 3 or over 4:')
# print('N: ', sum(np.logical_or(y_valid_pred < 3, y_valid_pred > 4)))
# print('Predictions:')
# print('', y_valid_pred[np.logical_or(y_valid_pred < 3, y_valid_pred > 4)])
