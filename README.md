# Masked Autoencoders for Bovine Images

This repository contains the code used in my final undergraduate project, which explores the use of masked autoencoders as feature extractors for bovine images. The project is divided into several Jupyter notebooks and a script for defining the model.

## Notebooks

### `pre-training.ipynb`
This notebook is responsible for pre-training the masked autoencoder model on bovine image data.

### `feature_extractor.ipynb`
This notebook utilizes the pre-trained model's encoder to generate latent representations of bovine images.

## Script

### `mae_model.py`
This script contains the implementation of the Masked Autoencoder model and defines key hyperparameters.

## Instructions

1. Set the model size and mask proportion in `mae_model.py`.
2. Choose your task:
   - To pre-train a model on new data, use `pre-training.ipynb`.
   - To use a pre-trained encoder for feature extraction, run `feature_extractor.ipynb`.

## Reference

This implementation is based on the following tutorial: [Masked Image Modeling with Keras](https://keras.io/examples/vision/masked_image_modeling/).

