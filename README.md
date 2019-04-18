This repository contains code for the paper:

Audio-Visual Model Distillation Using Acoustic Images. [arXiv, 2019](https://arxiv.org/abs/1904.07933).

## Requirements

- Python 2
- TensorFlow 1.4.0 >=
- LibROSA 0.6.1 >=

## Contents

We provide several scripts and a package with all the necessary classes for training and testing our model. The code is organized in several folders and a couple of main scripts as follows:

- The `codebase` folder contains several sub-packages with common code used to train our models.

- The `utils` folder contains several utility functions and some scripts to manipulate the data of our audio-visually indicated actions dataset.

- The `main_s1` script is used for training and testing the student and teacher networks from action labels.

- The `main_s2` script is used for training and testing the student networks from the teachers soft predictions and the action labels.
