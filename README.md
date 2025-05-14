
# ecg_contrastive_learning
Exploring Unsupervised Contrastive Learning Methods for ECG Analysis: Experimental Study
This is a pytorch implementation using ptbxl data

Required Installation :
!pip install torch torchvision torchaudio pandas scikit-learn wfdb scipy pywavelets



Features: 

  Supervised baseline classifier on raw 12‑lead ECGs.
  
  SimCLR and MoCo v2 implementations for unsupervised pre‑training.
  
  Pluggable augmentations (scaling, jitter, lead masking, time‑warp, wavelet noise).
  
  Modular PyTorch code; easy to swap backbone or contrastive objective.
  
  Reproducible experiments on the PTB‑XL clinical ECG dataset.

Installation

# 1. Clone the repo
$ git clone https://github.com/sjb048/ecg_contrastive_learning.git

$ cd ecg_contrastive_learning

# 2A. Conda (recommended)
$ conda env create -f environment.yml

$ conda activate ecg-contrastive

# 2B. Pip (alternative)
$ python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

$ pip install -r requirements.txt  # minimal: torch torchvision torchaudio pandas scikit-learn wfdb scipy pywavelets

Dataset

This project is benchmarked on the PTB‑XL ECG dataset
