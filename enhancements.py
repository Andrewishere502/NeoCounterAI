'''This is a script containing functions for enhancing image datasets.'''

import numpy as np


def crop_img(img: np.array, start_row: int, end_row: int, start_col: int, end_col: int):
    '''Crop an image and return it.'''
    return img[start_row:end_row, start_col:end_col]



# Split images into other images


if __name__ == '__main__':
    test_img = (np.random.random(size=(240, 320, 3)) * 255).astype(np.int16)
    print(test_img.shape)

    cropped_img = crop_img(test_img, 0, 10, 0, 10)
    print(cropped_img.shape)

