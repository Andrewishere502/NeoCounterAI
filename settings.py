import pathlib
from typing import List


class Settings:
    '''A class to store constants for the resnet.py script.'''

    # Seed for random generators
    seed: int = 2

    # Path to dataset to use
    collection_name: str = 'DataNoSubstrate'
    # collection_name: str = 'Data_v2.0'

    # Add additional dense layers of n neurons before the output layer.
    # The first element is for the first dense layer added, which is
    # the closest to the convolutional layers.
    dense_layers: List = []

    # Consts for model.fit()
    epochs: int = 1
    min_delta: float = 0.01
    patience: int = 2
    max_weight: float = 1.0


def save_settings(filename: pathlib.Path, settings: Settings, **kwargs) -> None:
    '''Save the settings from the Settings object to a specified (text)
    file.
    
    Arguments:
    filename -- name of file to write settings to
    settings -- Settings object to write to file
    **kwargs -- any additional variables that should
                be saved in the file with the settings
    '''
    # Function to format each line in settings.txt
    line_fmtr = lambda name, val: f'{name}:{str(type(val))[8:-2]}={val}\n'

    # Write settings variables and their values to file
    with open(filename, 'w') as file:
        # Get all attribute names that aren't dudner attrs/methods.
        # NOTE: attr_names will be sorted alphabetically by default.
        attr_names = [attr for attr in dir(settings) if attr[:2] != '__']
        # Write these attributes and their values to the specified
        # directory.
        for attr_name in attr_names:
            attr_val = getattr(Settings, attr_name)
            file.write(line_fmtr(attr_name, attr_val))

        # Write the kwargs to the file as well
        for name, val in kwargs.items():
            file.write(line_fmtr(name, val))
    return


if __name__ == '__main__':
    save_settings('test_settings.txt', Settings)
    print(globals())
