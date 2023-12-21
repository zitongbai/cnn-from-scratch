# About this fork

This fork is a fork of [vzhou842/cnn-from-scratch](https://github.com/vzhou842/cnn-from-scratch) with the following changes:
* implemented multi-channel input for Conv3x3 
* refined the computation of backward propagation in Conv3x3
* provided a simple example of building two layers of Conv3x3

The original README.md is as follows.

# A Convolution Neural Network (CNN) From Scratch
This was written for my 2-part blog post series on CNNs:

- [CNNs, Part 1: An Introduction to Convolution Neural Networks](https://victorzhou.com/blog/intro-to-cnns-part-1/)
- [CNNs, Part 2: Training a Convolutional Neural Network](https://victorzhou.com/blog/intro-to-cnns-part-2/)

To see the code (forward-phase only) referenced in Part 1, visit the [forward-only](https://github.com/vzhou842/cnn-from-scratch/tree/forward-only) branch.

## Usage

Install dependencies:

```bash
$ pip install -r requirements.txt
```

Then, run it with no arguments:

```bash
$ python cnn.py
$ python cnn_keras.py
```

You can also [run this code in your browser](https://repl.it/@vzhou842/A-CNN-from-scratch-Part-2).

## More

You may also be interested in [a Neural Network implemented from scratch in Python](https://github.com/vzhou842/neural-network-from-scratch), which was written for my [introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/).
