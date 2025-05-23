import pathlib
from typing import Callable


class Settings:
    '''A class to store constants for the resnet.py script.'''

    # Seed for random generators
    seed: int = 2

    # Path to dataset to use
    collection_name: str = 'DataNoSubstrate'
    # collection_name: str = 'Data_v2.0'
    # Drop image groups that are extremely under represented
    lim_min_prop = True
    # Drop images from a group that is under represented for balance
    lim_max_prop = False
    # Whether or not to crop out the glare from the right side of each
    # image
    crop_glare = True
    # How many randomly cropped images to create from each base image.
    # 0 disables this functionality
    rand_crop_n = 0
    rand_crop_width = 200   # Unused if rand_crop_n = 0
    rand_crop_height = 200  # Unused if rand_crop_n = 0

    # Consts for model.fit()
    epochs: int = 50
    min_delta: float = 0.01
    patience: int = 1
    max_weight: float = None #1.0
    validation_split: float = 0.10  # Portion of training data to use as validation
    restore_best_weights = False

    # Whether or not to retrain the resnet layers
    retrain_conv = False

    @classmethod
    def modify(cls, **kwargs):
        '''Modify the setting values'''
        for attr_name, attr_val in kwargs:
            cls.__setattr__(attr_name, attr_val)


def save_settings(filename: pathlib.Path, settings: Settings, **kwargs) -> None:
    '''Save the settings from the Settings object to a specified (text)
    file.

    Arguments:
    filename -- name of file to write settings to
    settings -- Settings object to write to file, or None
    **kwargs -- any additional variables that should
                be saved in the file with the settings
    '''
    # Function to format each line in settings.txt
    line_fmtr = lambda name, val: f'{name}:{str(type(val))[8:-2]}={val}\n'

    # Write settings variables and their values to file
    with open(filename, 'w') as file:
        # Only write settings object if it was given
        if settings != None:
            # Get all attribute names that aren't dudner attrs/methods.
            # NOTE: attr_names will be sorted alphabetically by default.
            attr_names = [attr for attr in dir(settings) if attr[:2] != '__']
            # Write these attributes and their values to the specified
            # directory.
            for attr_name in attr_names:
                attr_val = getattr(Settings, attr_name)
                if isinstance(attr_val, Callable):
                    # Don't save this model's methods
                    continue
                file.write(line_fmtr(attr_name, attr_val))

        # Write the kwargs to the file as well
        for name, val in kwargs.items():
            file.write(line_fmtr(name, val))
    return

