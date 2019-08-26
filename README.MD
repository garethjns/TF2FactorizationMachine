# Factorization machine in TensorFlow 2

Example implementing a factorization machine in TensorFlow 2, along with a framework for generating user-item ratings for testing.

Based on the theory and tf-1 code in [this article](http://nowave.it/factorization-machines-with-tensorflow.html) by Gabriele Modena and the paper by [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) by Steffen Rendle. Please refer to these articles for much more detailed descriptions of the model, and the maths that avoid O(p<sup>2</sup>).

Code is work in progress so expect bugs and inconveniences. See examples/ directory for more detailed examples.

# Install

````bash
git clone https://github.com/garethjns/TF2FactorizationMachine.git
````

# Example

## TensorFlow 2 model
For now the current model only implements .\_\_init__ .\_\_call__ and follows a similar structure as the [TensorFlow linear models example](https://www.tensorflow.org/beta/tutorials/eager/custom_training). Training is done in a loop with a function (train_step) to update the parameters; these will be moved to .fit at some point. 

See also [examples/1_model_development.ipynb](https://github.com/garethjns/TF2FactorizationMachine/blob/master/examples/1_model_development.ipynb) for mode details on model development and [examples/2_movie_lens_data.ipynb](https://github.com/garethjns/TF2FactorizationMachine/blob/master/examples/2_movie_lens_data.ipynb) for an example running it on the MovieLens dataset.

````Python
import numpy as np

import matplotlib.pyplot as plt

from fmachine.model import FactorizationMachine
from fmachine.helpers import train_step, l2_loss

# Features
x = np.array([[1, 0, 0,  1, 0, 0, 0,  0.3, 0.3, 0.3, 0.0,  13,  0, 0, 0, 0 ],
              [1, 0, 0,  0, 1, 0, 0,  0.3, 0.3, 0.3, 0.0,  14,  1, 0, 0, 0 ],
              [1, 0, 0,  0, 0, 1, 0,  0.3, 0.3, 0.3, 0.0,  16,  0, 1, 0, 0 ],
              [0, 1, 0,  0, 0, 1, 0,  0.0, 0.0, 0.5, 0.5,  5,   0, 0, 0, 0 ],
              [0, 1, 0,  0, 0, 0, 1,  0.0, 0.0, 0.5, 0.5,  8,   0, 0, 1, 0 ],
              [0, 0, 1,  1, 0, 0, 0,  0.5, 0.0, 0.5, 0.0,  9,   0, 0, 0, 0 ],
              [0, 0, 1,  0, 0, 1, 0,  0.5, 0.0, 0.5, 0.0,  12,  1, 0, 0, 0 ]])

# Targets (explicit rating)
y = np.array([5, 3, 1, 4, 5, 1, 5])
y.shape = (7, 1)

mod = FactorizationMachine(m=16)

# Training
epochs = 200
bs, ws, vs, losses = [], [], [], []
for e in range(epochs):
    cur_loss = train_step(mod=mod, 
                     x=x, 
                     y_true=y,
                     lr=0.0025,
                     loss_f=l2_loss)

# PLot loss
plt.plot(losses)
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
````

## Keras interface
Todo

## Data generator
Todo
