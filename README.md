# Fuzzy Gender Classification

This project demonstrates a simple implementation of gender classification using fuzzy logic in Python.

## Overview

The project consists of the following main components:

1. **Preprocessing**: The input image is loaded and preprocessed to prepare it for feature extraction.

2. **Feature Extraction**: A simple feature extraction method is used to extract features from the image. In this example, a mock gender prediction is used as the extracted feature.

3. **Fuzzy Logic**: Fuzzy sets and rules are defined to perform gender classification based on the extracted features.

4. **Fuzzy Inference**: Fuzzy inference is performed using the defined rules and membership degrees to determine the gender classification.

5. **Defuzzification**: The fuzzy inference result is defuzzified to obtain a crisp output for gender classification.

6. **Display**: The original image is displayed with the predicted gender label overlaid on it.

## Usage

To run the project:

1. Ensure you have Python installed on your system.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Place your input image in the project directory.
4. Run the `Gender-Prediction-fuzzy-System.py` script:

   ```bash
   python Gender-Prediction-fuzzy-System.py
