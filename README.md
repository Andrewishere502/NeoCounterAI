# NeoCounterAI
This is a research project dedicated to developing a convolutional neural network model that can count the number of Neocaridina davidi shrimp in an image.


# Requirements/Dependencies
0. Make sure you have python 3.9.6 installed and aliased to `python3`

1. Before you do anything, make sure to create a proper virtual environment! This will be a place you can put all the packages you need to make your own model. Navigate to your project directory and run the following line to build the virtual environment, named `.venv`:

`python3 -m venv .venv`

2. Activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS and Linux (bash) | `source .venv/bin/activate` |
| Windows (powershell) | `source .venv/bin/Activate.ps1` |

*[More on venv](https://docs.python.org/3/library/venv.html)*

3. The `requirements.txt` file has all the libraries you will need to get started. Run the following line to build all of the libraries for your project.

`python3 -m pip install -r requirements.txt`

# Doing Something Similar?
If you're doing your own work trying to make a model totally take a look at `resnet.py` (and `settings.py`). While you might not be able to "plug and play" using my code, here are some tips on what will likely be helpful to you:

1. Once the libraries have installed, you'll want to import them into your file. Probably something like this:
```
import datetime
import pathlib
from typing import ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from settings import Settings, save_settings
```

2. Load in your data, somehow. You need an array of images, then another array of labels for those images. Don't forget you'll have to transform your images using the ResNet preprocessing function `tf.keras.applications.resnet.preprocess_input`.


4. Create a sequential model like so:

`model = tf.keras.models.Sequential()`

4. Now add the ResNet model to your model:
```
input_shape = (...,)  # Add your desired input shape!
model.add(tf.keras.applications.ResNet50(input_shape=input_shape,
                                         weights='imagenet',
                                         include_top=False,
                                         pooling='avg'))
```

5. Make the ResNet model's layers untrainable:
```
for layer in model.layers:
    layer.trainable = False
```

6. Add your output layer (for a *regression*):
```
model.add(tf.keras.layers.Dense(units=1,
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(...),  # Set a seed for reproduceability
                                bias_initializer='zeros'
                                ))
```

7. Compile your model:
```
model.compile(
    optimizer='adam',
    loss=...,   # Add a loss function
    metrics=... # Add some metric(s)
)
```
Check out [loss](https://keras.io/api/losses/) and [metrics](https://keras.io/api/metrics/) documentation.


8. Train your model:
```
# Make sure X_train has been transformed by tf.keras.applications.resnet.preprocess_input already!
model.fit(X_train, y_train)
```
I recommend adding some [callbacks](https://keras.io/api/callbacks/).

9. After your model is done training, you'll need to assess it. I'll leave that to you to figure out!
