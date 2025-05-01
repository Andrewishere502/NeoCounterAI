'''This is a script containing functions for enhancing image datasets.'''

from typing import Tuple

import numpy as np


def crop_img(img: np.ndarray, x: int, y: int, new_width, new_height):
    '''Crop an image and return it.'''
    return img[y:y+new_height, x:x+new_width]


def randcrop_img(img: np.ndarray, new_width: int, new_height: int) -> Tuple[int, int, np.ndarray]:
    '''Crop an image with size new_width and new_height from the given
    image, starting at a random location within the given image. Return
    the starting column and starting row of the cropped image from
    within the original image along with the cropped image.
    '''
    width = img.shape[1]
    height = img.shape[0]

    # Get x and y margin, telling us where a randomly cropped image
    # could start without its borders exceeding the current image's
    x_margin = width - new_width
    y_margin = height - new_height

    start_col = int(np.random.random() * x_margin)
    start_row = int(np.random.random() * y_margin)
    return start_col, start_row, crop_img(img, start_col, start_row, new_width, new_height)


def rot_img(img: np.ndarray, k: int) -> np.ndarray:
    '''Rotate an image 90 degrees k times.'''
    # Rotate the image by 90 degrees k times
    return np.rot90(img, k)


if __name__ == '__main__':
    test_img = (np.random.random(size=(4, 4)) * 255).astype(np.int16)
    print(test_img)

    cropped_img = crop_img(test_img, 0, 0, 2, 2)
    print(cropped_img)

    x_off, y_off, rand_crop = randcrop_img(test_img, 2, 2)
    print(rand_crop)

    rot0_img = rot_img(rand_crop, 0)
    print(rot0_img)

    rot90_img = rot_img(rand_crop, 1)
    print(rot90_img)
