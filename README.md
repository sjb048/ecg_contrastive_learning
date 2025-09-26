
# ECG Contrastive Learning

**Exploring Unsupervised Contrastive Learning Methods for ECG Analysis**  
A PyTorch implementation of contrastive learning (SimCLR, MoCo v2, etc.) on the 12-lead ECG domain, benchmarked on PTB-XL.


Required Installation :
!pip install torch torchvision torchaudio pandas scikit-learn wfdb scipy pywavelets



Features: 

- Supervised baseline classifier applied to raw 12-lead ECG signals.  
- Unsupervised pretraining via **SimCLR** and **MoCo v2** variants.  
- Modular, pluggable data augmentations (scaling, jitter, lead masking, time-warp, wavelet noise).  
- Clean, modular PyTorch code: easy to swap backbone architectures or contrastive objectives.  
- Reproducible experimental pipelines for ECG tasks on PTB-XL dataset.

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

This project is benchmarked using the PTB-XL ECG dataset.
Make sure you have the data downloaded and stored in the expected directory (specify path) before running experiments.

🧪 Usage & Examples
Training
python train.py --config configs/simclr.yaml

or for MoCo:

python train.py --config configs/moco.yaml
Evaluation / Downstream classification
python eval.py --model_path path/to/checkpoint --dataset ptbxl

You can also run with different augmentations or backbones by passing in configuration files.
🛠️ Configuration
All major parameters (learning rate, batch size, augmentation settings, backbone architecture) are exposed in YAML config files under configs/.
You can create your own or modify existing ones as needed.

ecg_contrastive_learning/
├── configs/ # YAML config files for experiments
├── data/ # Data loading scripts / dataset interface
├── models/ # Model architectures and contrastive modules
├── augmentations/ # ECG augmentation functions
├── train.py # Training entrypoint
├── eval.py # Evaluation / downstream code
├── environment.yml / requirements.txt
├── README.md /# Project documentation

💡 Tips & Notes

Make sure to fix random seeds for reproducibility.

Log experiments (e.g. with TensorBoard, Weights & Biases).

Try ablations: effect of each augmentation, backbone choices, contrastive loss variants.

Be careful with ECG signal preprocessing (filtering, normalization).