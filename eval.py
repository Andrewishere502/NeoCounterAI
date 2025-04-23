'''Script to evaluate a given CNN model.'''

import pathlib
from typing import Tuple

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


# model_path = pathlib.Path('Models/20250422-181628/model.keras')  # 1 epochs,  max_weight None, DataNoSubstrate
# model_path = pathlib.Path('Models/20250422-222318/model.keras')  # 5 epochs,  max_weight None, DataNoSubstrate
# model_path = pathlib.Path('Models/20250422-224232/model.keras')  # 10 epochs, max_weight None, DataNoSubstrate
# model_path = pathlib.Path('Models/20250422-225100/model.keras')  # 20 epochs, max_weight None, DataNoSubstrate
# model_path = pathlib.Path('Models/20250423-002634/model.keras')  # 50 epochs, max_weight None, DataNoSubstrate
# model_path = pathlib.Path('Models/20250423-090519/model.keras')  # 50 epochs, max_weight 100,  DataNoSubstrate
model_path = pathlib.Path('Models/20250423-100417/model.keras')  # 50 epochs, max_weight 1,    DataNoSubstrate
# model_path = pathlib.Path('Models/20250423-121442/model.keras')  # 50 epochs, max_weight 1,    Data_v2.0
model = tf.keras.models.load_model(model_path)


# Crudely copy and pasted from resnet.py
# Validation indices for DataNoSubstrate
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

# Validation indices for Data_v2.0
# validation_is = '''767  189  634 1204  179  149 1121 1778 2119 2369 2521 1728  863 2368
#  2550 1616  345 1811 1274 1319 1660   50 2001 1825  968  262  120  987
#  1316 1833  219  267 1392 1216  843  290  692  697 1285  478 1413  716
#  2270 1570   74 1210  574 2312 1456 1538 1803 1620  773 2479 1035 1789
#   317  963  654 1020 1249 2209 1884 1692 1233 1927  955 2154 2416 1203
#   901 1670  513 1336  440  621 1184  983 2013 1400  205 2161 1312   56
#  1708 1408  500 1159 2004 1255  531  425 1674  741 2113 2378 1334  589
#   237 1260 1450 1164 1173 1605  605 2467  175 1737 1034 1213 1783 1480
#  1959 1434 1978 1390 1341 1517 2124 1695 2115  152 1582 2522  211 2062
#  2089  859 2364  348 1776 2152 2359  883 2434 2473 2400  264  291  253
#  1908 1623 2210 1178 2460   22 2011 2228  700  881  338 2148 1430 1951
#  1282  227 2504 1487  107 1388 1404 1451 2370 2528 1680 1189  857  592
#  1634 2546 1548   30   10 1628  438 1366  971  710 2003 1482 2006  669
#   866 1192  845  125  257 2279 2008 1967 1926 1337 2043 1160 1028 1059
#   198  628  409 2345 1740 2180 1198  282 2187  839 1493 1226  639 1619
#   631 1201 1299 2365  412   13  483 2519 1058 1995  475 2236 1562 1639
#   656 1141  536  967 2212 2463 1665  602 1647 1662 2050  163 1135 1368
#  1593 1458  482 1244 2372  130 2076 1063  233 1362  693 2222 2220   72
#   514 2085 1539 1578 2168   12 1471 2328 2098 1600 1533 1906  811 1370
#   153 1328   42 1240 1418  878 1949 2458 1483  359   55  879 1318   26
#   680 1238 1513  135 1534 1377 2373 1000 1862 1064  396 2492   53  178
#   507  300 1552 1663 2249 1001   78  995  726  228  818 1397  988   29
#  1657  561  225    5 1227  444   32  965 1681  893  668 1358 1406 2185
#  1396  151 2278  222 2130  702 1666 1444  854 1832 1586 1525  113 2096
#  1100  206 2435 1147 1821 1410  889 2537  899  832 1245 2443 1241 1982
#   869 1659  673 1374 1676 2221  139  489  913 1615 1671 1898 2190  812
#   214 1943  314  416 1029 2086 2101  136  746 1474 1122  793   35   25
#   657 1402  449 1401 1258  751 1270 2525 1449 1052  244 2151  276 2158
#   588  970 1947 2198 2240 1759 2181 1323  167  927   24  806  443 1779
#   150 2388 2142  754 1047  780  900  864 1952 1197  100 1049 1627 1271
#  1758 2290 1378  766  750  962 2320  168   73 2538 1883 2166 1138  992
#  2053 1499 1905 1093 1384 1738 2260 1503 1631 2102  266 1033 1829 1844
#  2107 2253 2214  144  331 1973  247  373  380  320 2239  636 1587 1331
#  2433 1214  430 1010  191 2267 1175  435 2339 1690 1942 1280 1899 1186
#  1016 2232  687 1565  638  569  933 1407 1169 1360  533 1822  196 1802
#   495  936  979 2075 1301 1813 1655  395 1705  147  797 1750  623 2286
#  2028 1009 1518  910 1622  895 1473 2415'''.split()
# validation_is = list(map(int, validation_is))

# Path to data
data_dir = pathlib.Path('DataNoSubstrate')
# data_dir = pathlib.Path('Data_v2.0')
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
