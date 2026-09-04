# Machine Learning Final Project

## Image Classification with CIFAR-10 using PyTorch

### Description
This project trains a CNN (Convolutional Neural Network) to classify images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

### Requirements
- Python 3.11+
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Scikit-learn
- Seaborn

### Setup
```bash
# Create virtual environment
conda create -n ml_project python=3.11
conda activate ml_project

# Install dependencies
pip install torch torchvision matplotlib numpy scikit-learn seaborn jupyter
```

### How to Run
```bash
cd Mini_project/ML_Project
jupyter notebook ml_final_project.ipynb
```

### Project Structure
```
Mini_project/ML_Project/
├── ml_final_project.ipynb    # Main notebook (code + report)
├── README.md                 # This file
├── data/                     # CIFAR-10 data (auto-downloaded)
└── models/                   # Saved model weights
```

### Results
See the notebook for detailed results, training curves, and model comparison.
