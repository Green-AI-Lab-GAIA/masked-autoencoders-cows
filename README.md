# masked-autoencoders-cows

This repository contains the code used in my final undergraduation project, which involves using masked autoencoders as feature extractors for bovine images. It is divided into several notebooks.

## Notebooks

### pre-training
Responsible for the model's pre-training.

### feature_extractor
Uses the model's encoder to create latent variables corresponding to the images.


You can download pre-trained models from [this Google Drive folder](https://drive.google.com/drive/folders/1xtkDYHZEf9j_ePhSHT70BkPHmQwHqyvJ?usp=sharing). Each folder is named `mp[number]`, where the number represents the mask proportion of the pre-trained model. These models are useful for the feature extraction code in the `feature_extractor` notebook.

