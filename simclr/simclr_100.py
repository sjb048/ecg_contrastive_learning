import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging
import csv

from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import random
import math

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_rows = []  # For saving to CSV
# --- WAVELET DENOISING FUNCTION (optional) ---
def wavelet_denoise(ecg_signal_1d, wavelet='db4', level=2, mode='soft'):
    coeffs = pywt.wavedec(ecg_signal_1d, wavelet, level=level)
    detail_coeffs = coeffs[1:]
    detail_array = np.concatenate([np.abs(dc) for dc in detail_coeffs]) if detail_coeffs else np.array([])
    if len(detail_array) == 0:
        return ecg_signal_1d
    median_abs = np.median(detail_array)
    sigma = median_abs / 0.6745
    n = len(ecg_signal_1d)
    threshold = sigma * np.sqrt(2 * np.log(n)) if n > 1 else 0
    new_coeffs = [coeffs[0]]
    for dc in coeffs[1:]:
        dc_t = pywt.threshold(dc, threshold, mode=mode)
        new_coeffs.append(dc_t)
    denoised = pywt.waverec(new_coeffs, wavelet)
    if len(denoised) > n:
        denoised = denoised[:n]
    elif len(denoised) < n:
        denoised = np.pad(denoised, (0, n - len(denoised)), mode='constant')
    return denoised

# --- DATA AUGMENTATION FUNCTION ---

# --- 1D TRANSFORMS for SimCLR augmentations ---
class RandomCrop1D:
    def __init__(self, crop_size): self.crop_size = crop_size
    def __call__(self, signal):
        if signal.size(-1) <= self.crop_size:
            return signal
        start = random.randint(0, signal.size(-1) - self.crop_size)
        return signal[..., start:start+self.crop_size]

class RandomGrayscale1D:
    def __init__(self, p=0.2): self.p = p
    def __call__(self, signal):
        if random.random() < self.p and signal.size(0) > 1:
            return signal.mean(dim=0, keepdim=True)
        return signal

class RandomScaling1D:
    def __init__(self, scale_min=0.8, scale_max=1.2): self.scale_min, self.scale_max = scale_min, scale_max
    def __call__(self, signal): return signal * random.uniform(self.scale_min, self.scale_max)

class RandomAddNoise1D:
    def __init__(self, noise_std=0.02): self.noise_std = noise_std
    def __call__(self, signal): return signal + torch.randn_like(signal) * self.noise_std

class RandomFlip1D:
    def __call__(self, signal): return torch.flip(signal, dims=[-1]) if random.random() < 0.5 else signal

class RandomTimeShift1D:
    def __init__(self, max_shift=10): self.max_shift = max_shift
    def __call__(self, signal): return torch.roll(signal, shifts=random.randint(-self.max_shift, self.max_shift), dims=-1)

class Normalize1D:
    def __init__(self, mean=0.0, std=1.0): self.mean, self.std = mean, std
    def __call__(self, signal): return (signal - self.mean) / self.std

# --- SimCLR-Style Augmentation Dataset ---
class AugmentationSimCLR(Dataset):
    def __init__(self, signals, crop_size=1000):
        self.signals = signals
        # Define three different pipelines
        self.pipes = [
            torch.nn.Sequential(
                RandomCrop1D(crop_size),
                RandomScaling1D(),
                RandomAddNoise1D(),
                Normalize1D(mean=0.5,std=0.5)
            ),
            torch.nn.Sequential(
                RandomFlip1D(),
                RandomGrayscale1D(),
                RandomTimeShift1D(),
                RandomAddNoise1D(),
                Normalize1D(mean=0.5,std=0.5)
            ),
            torch.nn.Sequential(
                RandomCrop1D(crop_size),
                RandomTimeShift1D(),
                RandomAddNoise1D(),
                Normalize1D(mean=0.5,std=0.5)
            )
        ]
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx):
        x = self.signals[idx]  # Tensor [1, L]
        views = [pipe(x.clone()) for pipe in self.pipes]
        return tuple(views)

# --- Simclr TRAINING FUNCTIONS ---
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --- DATA LOADING FOR UNLABELED DATA (for MoCo Pre-training) ---
def load_and_segment_signals_1d_no_folder_limit(
    csv_file,
    base_dir,
    filename_col='filename_hr',
    max_files=5000,
    segment_length=1000,
    fs=125.0,
    do_wavelet=False,
    wavelet='db4',
    level=2,
    mode='soft',
    lead_index=0,
    random_files=False
):
    """
    Loads up to max_files ECG signals from the CSV.
    For each record, extracts the first segment_length samples from one lead.
    Optionally applies wavelet denoising.
    Returns: a DataFrame of segments and a list of missing files.
    """
    df = pd.read_csv(csv_file)
    if random_files:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)


    df = df.iloc[:max_files].copy()

    loaded_segments = []
    missing_files = []

    for _, row in df.iterrows():
        rel_path = row[filename_col]
        full_hea_path = os.path.join(base_dir, rel_path) + '.hea'
        if not os.path.exists(full_hea_path):
            missing_files.append(full_hea_path)
            continue
        try:
            record_path_no_ext = full_hea_path[:-4]
            record = wfdb.rdrecord(record_path_no_ext)
            ecg_signal = record.p_signal  # shape: (num_samples, num_leads)
            if ecg_signal.shape[0] < segment_length:
                missing_files.append(full_hea_path)
                continue
            segment_2d = ecg_signal[:segment_length, :]
            segment_1d = segment_2d[:, lead_index]
            if do_wavelet:
                segment_1d = wavelet_denoise(segment_1d, wavelet=wavelet, level=level, mode=mode)
            seg_df = pd.DataFrame({
                "ecg_1d": segment_1d,
                "record_id": [os.path.basename(record_path_no_ext)] * segment_length
            })
            loaded_segments.append(seg_df)
        except Exception as e:
            missing_files.append(full_hea_path)
            continue
    all_segments_df = pd.concat(loaded_segments, ignore_index=True) if loaded_segments else pd.DataFrame()
    return all_segments_df, missing_files

# Contrastive loss (NT-Xent)
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._get_correlated_mask().to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).unsqueeze(1)
        negatives = sim[self.mask].reshape(N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N, dtype=torch.long).to(self.device)
        loss = self.criterion(logits, labels)
        return loss
# --- ENCODER & PROJECTION ---
# Improved 1D Encoder
class ImprovedECGEncoder(nn.Module):
    """
    A 1D CNN encoder for ECG signals.
    Uses 4 stacked convolutional blocks and a final FC layer to produce an embedding.
    """
    def __init__(self, in_channels=1, base_channels=64, embedding_dim=128):
        super().__init__()
         # Add self-attention layer
        self.attention = nn.Sequential(
            nn.Conv1d(base_channels * 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels*8), nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(nn.Conv1d(base_channels*8,1,1), nn.Sigmoid())
        self.fc = nn.Linear(base_channels*8, embedding_dim)

    def forward(self,x):
        x = self.conv(x)
        x = x * self.attn(x)
        x = F.adaptive_avg_pool1d(x,1).squeeze(-1)
        return F.normalize(self.fc(x),dim=1)

# --- PROJECTION HEAD ---
class ProjectionHead(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2), nn.ReLU(inplace=True),
            nn.Linear(emb_dim*2, proj_dim)
        )
    def forward(self,x): return F.normalize(self.net(x),dim=1)

class SimCLR1D(nn.Module):
    def __init__(self, encoder, emb_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionHead(emb_dim,proj_dim)
    def forward(self,x):
        h = self.encoder(x)
        z = self.proj(h)
        return h,z


# --- Contrastive Trainer ---
class SimCLRTrainer:
    def __init__(self, model, optimizer, scheduler, args):
        self.model, self.opt, self.sched, self.args = model, optimizer, scheduler, args
        self.writer = SummaryWriter()
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.scaler = GradScaler(enabled=args.fp16_precision)

    def info_nce_loss(self, zis, zjs):
        batch_size = zis.size(0)
        representations = torch.cat([zis, zjs],dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2) / self.args.temperature
        mask = (~torch.eye(2*batch_size, dtype=torch.bool)).to(self.args.device)
        positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)])
        negatives = similarity_matrix[mask].view(2*batch_size, -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2*batch_size, dtype=torch.long).to(self.args.device)
        return logits, labels

    def train(self, loader):
        for epoch in range(self.args.epochs):
            total_loss=0
            for views in loader:
                xi, xj = views  # two views
                xi, xj = xi.to(self.args.device), xj.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    hi, zi = self.model(xi)
                    hj, zj = self.model(xj)
                    logits, labels = self.info_nce_loss(zi, zj)
                    loss = self.criterion(logits, labels)
                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                total_loss += loss.item()
            self.sched.step()
            avg_loss=total_loss/len(loader)
            logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            self.writer.add_scalar('Loss/train', avg_loss, epoch)


def create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True )

# --- SUPERVISED CLASSIFIER USING PRE-TRAINED ENCODER ---
class ECGClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, num_classes=2):
        """
        Uses the pre-trained encoder to extract features from ECG,
        then applies a classification head.
        """
        super(ECGClassifier, self).__init__()
        self.encoder = encoder
        # Optionally freeze encoder parameters:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits

def train_classifier(model, train_loader, val_loader=None, device='cpu', epochs=40, lr=1e-3):
    
     # Add class weights to the loss function
    from sklearn.utils.class_weight import compute_class_weight
    
    y_train = [y for _, y in train_loader.dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Replace standard loss with weighted version
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    current_val_accuracy = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for ecg_batch, labels_batch in train_loader:
            ecg_batch, labels_batch = ecg_batch.to(device), labels_batch.to(device)
            outputs = model(ecg_batch)
            loss = criterion(outputs, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ecg_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        wandb.log({"classifier_epoch": epoch+1, "train_loss": train_epoch_loss, "train_acc": train_epoch_acc})

        
        # Only run validation if a validation loader is provided
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for ecg_batch, labels_batch in val_loader:
                    ecg_batch, labels_batch = ecg_batch.to(device), labels_batch.to(device)
                    outputs = model(ecg_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item() * ecg_batch.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels_batch).sum().item()
                    val_total += labels_batch.size(0)
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f} | No validation set provided.")


# --- LABELED DATASET FOR SUPERISED FINETUNING ---
class PTBXLDataset(Dataset):
    def __init__(self, csv_path, root_dir, use_lr=True, folds=[1,2,3,4,5] , transform=None):

        self.root_dir = root_dir
        self.use_lr = use_lr
        # print(self.use_lr)
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        # Ensure the column exists in the CSV.
        if 'strat_fold' not in self.df.columns:
            raise ValueError("CSV file does not contain 'strat_fold' column for fold filtering.")
        self.df = self.df[self.df['strat_fold'].isin(folds)].reset_index(drop=True)


        self._validate_columns()
        self.df['label'] = self.df['scp_codes'].apply(
            lambda s: 0 if "NORM" in ast.literal_eval(s) else 1
        )
        self._filter_missing_files()

    def _validate_columns(self):
        required_cols = {'filename_lr', 'filename_hr', 'scp_codes', 'strat_fold'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        data_dir = 'records100'
        # data_dir = 'records100' if self.use_lr else 'records500'
        # print (data_dir)
        print("Loaded data records:", data_dir)
   
        
        full_path = os.path.join(self.root_dir, data_dir)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"ECG data folder not found: {full_path}")

    def _filter_missing_files(self):
        invalid_rows = []
        filename_col = 'filename_lr' if self.use_lr else 'filename_hr'

        for idx, row in self.df.iterrows():
            wfdb_path = os.path.join(self.root_dir, row[filename_col])
            primary_hea = wfdb_path + '.hea'
            primary_dat = wfdb_path + '.dat'
            alt_path = wfdb_path.replace('_lr', '').replace('_hr', '')
            alt_hea = alt_path + '.hea'
            alt_dat = alt_path + '.dat'
            if not (os.path.exists(primary_hea) and os.path.exists(primary_dat)) and \
               not (os.path.exists(alt_hea) and os.path.exists(alt_dat)):
                invalid_rows.append(idx)

        if invalid_rows:
            self.df.drop(index=invalid_rows, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"Filtered dataset: {len(self.df)} valid records remain.")
        else:
            logger.info(f"No missing files found. {len(self.df)} valid records remain.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        filename_col = 'filename_lr' if self.use_lr else 'filename_hr'
        wfdb_path = os.path.join(self.root_dir, record[filename_col])
        try:
            signals, _ = wfdb.rdsamp(wfdb_path)
        except FileNotFoundError:
            alt_path = wfdb_path.replace('_lr', '').replace('_hr', '')
            try:
                signals, _ = wfdb.rdsamp(alt_path)
            except FileNotFoundError as e:
                logger.error(f"File not found: {wfdb_path} and alternative {alt_path}")
                raise e
        ecg = signals.T.astype(np.float32)  # (12, time)
        # For classification, we flatten the data (or you can keep it 2D if you design a multi-channel model)
        ecg_flat = ecg.reshape(-1)
        ecg_flat = np.expand_dims(ecg_flat, axis=0)  # (1, 12*time)
        if self.transform:
            ecg_flat = self.transform(ecg_flat)
        return (
            torch.tensor(ecg_flat, dtype=torch.float32),
            torch.tensor(record['label'], dtype=torch.long)
        )

def create_dataloaders(dataset, batch_size=8, test_size=0.2, random_seed=42):
    #handled for small dataset label
    if isinstance(dataset, Subset):
        stratify_labels = dataset.dataset.df["label"].iloc[dataset.indices]
    else:
        stratify_labels = dataset.df["label"]

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_labels
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def evaluate_metrics(model, dataloader, device='cpu'):
    """
    Evaluates the given model on the provided dataloader and computes
    accuracy, precision, recall, and F1-score.

    Returns a dictionary with the metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ecg_batch, labels_batch in dataloader:
            ecg_batch = ecg_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(ecg_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'predictions': all_preds,
        'ground_truth': all_labels
    }

def evaluate_and_plot_confusion_matrix(model, dataloader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for ecg_batch, labels in dataloader:
            ecg_batch = ecg_batch.to(device)
            labels = labels.to(device)
            outputs = model(ecg_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Abnormal (1)'])
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix", fontsize=15, pad=20)
    # plt.show()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    return cm

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# --- MAIN PIPELINE ---
if __name__ == "__main__":
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ======================
    # Part A: MoCo Pre-training on Unlabeled Data
    # ======================
    # Load unlabeled data segments using the helper function.
    unlabeled_csv = '../data/train_unlabeled.csv'
    base_dir = '../data'
    wandb.init(project="smclr_umap_100", config={
        "simclr_epochs": 100,
        "simclr_batch_size": 32,
        "classifier_epochs": 40,
        "initial_lr": 1e-3,
        "segment_length": 1000,
        "wavelet": "db4",
        "lead_index": 0,
        
    }, reinit=True)


     # Instantiate SimCLR1D and pre-train
    simclr_model = SimCLR1D(
        in_channels=1,
        base_channels=64,
        emb_dim=128,
        K=65536,
        m=0.999,
        T=0.07
    )