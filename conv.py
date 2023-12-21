import numpy as np

'''
Note: In this implementation, I make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters, input_depth):
    self.num_filters = num_filters
    self.input_depth = input_depth

    # filters is a 4d array with dimensions (num_filters, 3, 3, input_depth)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3, input_depth) / 9

  def iterate_regions(self, image: np.ndarray):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 3d numpy array.
    '''
    h, w, _ = image.shape # _ is the input depth

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input: np.ndarray):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 3d numpy array
    '''
    self.last_input = input

    h, w, _ = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3))

    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        d_L_d_input[i:i+3, j:j+3] += d_L_d_out[i, j, f] * self.filters[f]

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return d_L_d_input
