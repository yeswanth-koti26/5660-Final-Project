# Pest Vision: Advanced Pest Classification Using EfficientNetV2 and XGBoost

## Project Overview

Pest Vision leverages state-of-the-art deep learning and machine learning algorithms to accurately classify pest insects from images. This innovative approach addresses the challenges faced by traditional pest classification methods, including variability in pest species, image quality, and scalability issues.


## Abstract

Pest classification is crucial for agriculture and environmental management, enabling early detection and targeted control measures. Traditional methods struggle with the complexity of pest images, leading to suboptimal classification accuracy. Our project proposes a novel solution using EfficientNetV2L for feature extraction and XGBoost for classification to improve accuracy and efficiency in pest classification tasks.

## Problem Statement

Traditional pest classification methods face several challenges:
- **Variability:** Diverse morphological features and color patterns.
- **Image Quality:** Variations due to different lighting conditions and angles.
- **Scalability:** Manual classification is impractical with increasing data volume.

## Solution Approach

Our solution involves:
- **EfficientNetV2L:** For efficient and effective feature extraction from diverse image data.
- **XGBoost:** For robust classification of high-dimensional datasets.

## Tools and Technologies

- **Programming Languages:** Python
- **Libraries:** Tkinter, NumPy, Matplotlib, OpenCV, Scikit-learn, Keras, XGBoost

## Dataset

- **Source:** Kaggle
- **Contents:** 5494 images of various pest insects.
- **Classes:** 12 classes (Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, Weevils)

## Methodology

1. **Data Collection:** Acquiring a diverse dataset of pest images.
2. **Data Preprocessing:** Normalization, shuffling, and splitting into training and testing sets.
3. **Model Selection:** EfficientNetV2L for feature extraction and XGBoost for classification.
4. **Training Process:** Fine-tuning EfficientNetV2L and training XGBoost.
5. **Evaluation Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix.

## Results

- **Training Progress:** Monitored via accuracy and loss graphs.
- **Model Performance:** Comparative analysis of EfficientNetV2L and XGBoost.
- **Pest Classification:** Demonstrated through test images with predicted labels.

## Challenges and Solutions

- **Data Quality and Quantity:** Addressed using data augmentation techniques.
- **Variability in Pest Images:** Handled through preprocessing techniques.
- **Python Tools and Techniques:** Leveraged for robust image processing, machine learning, and data visualization.

## Conclusion

The integration of EfficientNetV2L and XGBoost significantly improves the accuracy and efficiency of pest classification tasks. This project demonstrates the importance of leveraging advanced technologies to address complex challenges in agriculture and environmental sustainability. Future integration with real-time pest detection systems can enhance decision-making processes in agricultural practices, contributing to global food security efforts.

## Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yeswanth-koti26/Pest-Classification
2. **Navigate to the project directory:**
    ```sh
    cd Pest-classification
4. **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
6. **Run the main script:**
    ```sh
    python pest_classification.py


