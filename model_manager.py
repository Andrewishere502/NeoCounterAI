import pathlib
from typing import Dict, Any, List

import tensorflow as tf


class ModelManager:
    def __init__(self, hex_hash: str, model_name: str=None, load_model: bool=True) -> None:
        '''Initialize a new instance of the ModelManager class.
        
        Argument
        hex_hash -- A hash value identifying the location of a model
        model_name -- A name for this model
        load_model -- Whether or not to load in the actual model object
                      indicated
        '''
        # Set model_name to the hex_hash if no name provided
        if model_name == None:
            self.__model_name: str = hex_hash
        else:
            self.__model_name: str = model_name
        self.__hex_hash: str = hex_hash

        # Directory where the model and its accompanying data is stored
        self.__model_dir: pathlib.Path = pathlib.Path('Models', hex_hash)

        # Settings used to build this model
        self.__settings: Dict = self.__parse_settings()

        # NOTE: What type is this exactly?
        if load_model:
            self.__model = tf.keras.models.load_model(self.model_file)
        return
    
    @property
    def model_name(self) -> str:
        '''Return the model's name.'''
        return self.__model_name
    
    @property
    def hex_hash(self) -> str:
        '''Return the hex encoded hash of the model.'''
        return self.__hex_hash

    @property
    def model_dir(self) -> pathlib.Path:
        '''Return the directory in which this model is found.'''
        return self.__model_dir

    @property
    def model_file(self) -> pathlib.Path:
        '''Return the path to the model file.'''
        return self.model_dir / 'model.keras'
    
    @property
    def settings_file(self) -> pathlib.Path:
        '''Return the path to the settings file for this model.'''
        return self.model_dir / 'settings.txt'
    
    @property
    def partitions_file(self) -> pathlib.Path:
        '''Return the path to the data partitions file.'''
        return self.model_dir / 'data_partitions.txt'
    
    @property
    def log_file(self) -> pathlib.Path:
        '''Return the path to this model's training log file.'''
        return self.model_dir / 'training.log'

    @property
    def data_dir(self) -> pathlib.Path:
        '''Return the path to the data this model was trained on.'''
        return pathlib.Path(self.get_setting('collection_name'))
    
    @property
    def meta_file(self) -> pathlib.Path:
        '''Return the path to the metadata csv file descriibing the
        image data this model was trained on.
        '''
        return self.data_dir / 'metadata.csv'

    @property
    def model(self):  # NOTE: Update with return type once you figure it out
        '''Return the keras model instance.'''
        return self.__model

    def get_setting(self, name: str) -> Any:
        '''Return the value of a setting given its name.'''
        return self.__settings[name]

    @staticmethod
    def __dtype_convert(dtype: str, value: str) -> Any:
        '''Return the value as the given data type.'''
        if dtype == 'str':
            # Seems redundant... but feels right to do this just in case
            return str(value)
        elif dtype == 'int':
            return int(value)
        elif dtype == 'float':
            return float(value)
        elif dtype == 'NoneType':
            return None
        elif dtype == 'list':
            # Get list values as a list of strings
            str_values =  value[1:-1].split(', ')
            # List was empty, return empty list
            if len(str_values) == 1 and str_values[0] == '':
                return []
            return list(map(int, str_values))
        else:
            raise TypeError(f'Conversion to dtype \'{dtype}\' not supported')
    
    def __parse_settings(self) -> Dict:
        '''Parse the settings.txt file, loading in its values to a
        dictionary.
        '''
        # Read lines of the settings file, then close it
        with open(self.settings_file, 'r') as file:
            lines = file.readlines()
        # Strip white space from end of each line
        lines = map(lambda l: l.strip(), lines)

        # Construct the settings dictionary
        settings = {}
        for line in lines:
            name_dtype, value = line.split('=')
            name, dtype = name_dtype.split(':')
            # Convert the value to the appropriate data type
            settings.update({name: self.__dtype_convert(dtype, value)})
        return settings

    def load_partition(self, partition_name: str) -> List[int]:
        '''Return the indices of images belonging to a partition of
        the dataset, namely the training (train), validation (valid),
        or testing (test) partition.
        '''
        p = None
        if partition_name == 'train':
            # First line is training partition
            p = 0
        elif partition_name == 'test':
            # First line is testing partition
            p = 1
        else:
            raise ValueError(f'No partition of name \'{partition_name}\'')

        # Open the file and grab the partition(s)
        with open(self.partitions_file, 'r') as file:
            # Remove the first and last characters, which are
            part_line = file.readlines()[p]
        # Get the data component from the line, removing the partition name
        part_data = part_line.split('=')[1].strip()
        # Convert to the proper data type
        partition_indices: List[int] = self.__dtype_convert('list', part_data)
        return partition_indices
