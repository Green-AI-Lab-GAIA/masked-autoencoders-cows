# **Masked Autoencoders for Bovine Images**  

---

## **PyTorch Implementation (Recommended)**  

This implementation is based on the **Facebook Research** code, originally developed in PyTorch. The code has been modified to accept rectangular images from the dataset.  

### **Instructions**  

1. Use a virtual environment (**`pyenv` is recommended**).  
2. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the dataset:**  
   - All images should be the same size.  
   - Use `pre_processing.ipynb` to resize the images and split the dataset.  

4. **Pre-train the model:**  
   - If using **NumPy files**, run `pytorch/pre-training-new-sample.ipynb`.  
   - **Important:** Make sure to set the same `IMAGE_SIZE` used in `pre_processing.ipynb`.  
   - Configure the hyperparameters in this notebook.  

5.  ...

### **Reference**  

This implementation is based on the following repository: [Masked Autoencoders: A PyTorch Implementation](https://github.com/facebookresearch/mae).  

---

## **TensorFlow Implementation (Not Recommended)**  

This repository contains the code used in my **undergraduate final project**, which explores the use of **Masked Autoencoders** as feature extractors for bovine images. The project is divided into **several Jupyter notebooks** and a script for defining the model.  

### **Instructions**  

1. Use a virtual environment (**`pyenv` is recommended**).  
2. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Set the model size and mask proportion in the `tensorflow/mae_model.py` file.  
4. Choose your task:  
   - To **pre-train** a model on new data, use `tensorflow/pre-training.ipynb`.  
   - To **extract features** using a pre-trained encoder, run `tensorflow/feature_extractor.ipynb`.  

### **Reference**  

This implementation is based on the following tutorial: [Masked Image Modeling with Keras](https://keras.io/examples/vision/masked_image_modeling/).  


