# Fermi-LAT Unassociated Source Classification

This repository contains the Machine Learning implementation for classifying unassociated gamma-ray sources from the **Fermi Large Area Telescope (LAT)**. Using data from the **4FGL-DR2** catalog, the project employs a Supervised Neural Network (SNN) to categorize unknown sources based on their spectral and temporal characteristics.

## 🌌 Project Overview
The Fermi telescope has detected thousands of sources, but many remain "unassociated"—their physical origin is unknown. This project (rebooted from my Master’s thesis in Physics) uses supervised learning to predict if these sources belong to known astrophysical classes.

### Target Categories
Associated sources are divided into three main groups for classification:
1. **Pulsars**
2. **AGNs** (Active Galactic Nuclei)
3. **Others** (Binaries, SNR, etc.)

---

## 🛠 Methodology

### 1. Data Preparation
* **Source:** 4FGL-DR2 Catalog.
* **Preprocessing:** Selection of relevant physical features and handling of catalog data.
* **Splitting:** The associated dataset is split into **Training**, **Validation**, and **Testing** sets.

### 2. Neural Network Architecture
The project uses a Supervised Neural Network with the following configuration:
* **Optimizer:** Adam
* **Regularization:** Early-stopping to prevent overfitting.
* **Hyperparameter Scan:** The code includes scripts to scan for the optimal number of layers, neurons per layer, and batch size.

### 3. Inference
After training and validation, the model is used to evaluate and assign membership probabilities to the unassociated source population.

---

## 🚀 Getting Started

### Prerequisites
You will need Python 3.x and the following libraries:
* NumPy, Pandas, Scikit-learn
* TensorFlow/Keras

### Running the Code
1. **Train & Scan:** Run the main script to perform the hyperparameter scan and train the network.
2. **Evaluate:** Use the trained model to classify the unassociated sources in the 4FGL-DR2 catalog.

---

## 📈 Results
The output provides classification labels and confidence scores for previously unassociated sources, helping to prioritize them for multi-wavelength follow-up observations.

---

## 📚 Thesis Context
This code implements the first part of my Master's Thesis. For the full theoretical background, feature engineering details, and astrophysical analysis, please refer to the documentation:
* [Unveiling Fermi unidentified sources with Machine Learning](./docs/Giovannelli_thesis.pdf)

---

**Author:** Tommaso Giovannelli 