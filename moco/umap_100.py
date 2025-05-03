import os
import random
import time
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pywt
import wfdb
import logging
import csv
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE 

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
            # segment_1d = segment_2d[:, lead_index]
            LEAD_MAP = {
                'arrhythmia': 1,    # Lead II (index 1) for rhythm analysis
                'ischemia': 3,      # Lead V2 (index 3) for ST-segment changes
                'general': 0        # Lead I (index 0) as default
            }
            lead_idx = random.randint(0, 11)  # Random lead per sample
            # lead_index = LEAD_MAP['arrhythmia'] 
            segment_1d = segment_2d[:, lead_idx]
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
# --- DATA AUGMENTATION FUNCTION ---

def augment_ecg(ecg_batch, crop_size=1000, noise_std=0.01):
    """
    Applies random cropping and Gaussian noise to a batch of ECG signals.
    ecg_batch shape: [batch, in_channels, seq_len]
    """
    batch_size, in_channels, seq_len = ecg_batch.shape
    device = ecg_batch.device 
    augmented = ecg_batch.clone()

    # aug_type = random.choice(['none', 'noise', 'baseline', 'scale', 'shift'])
        
    for i in range(batch_size):
        aug_type = random.choice(['none', 'noise', 'baseline', 'scale', 'shift'])
        
        if aug_type == 'noise':
            # Add varying levels of noise
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(augmented[i]) * noise_level
            augmented[i] = augmented[i] + noise
            
        elif aug_type == 'baseline':
            # Add baseline wander
            # Add baseline wander
            wander = torch.linspace(0, random.uniform(-0.2, 0.2), seq_len, device=device)
        
            wander = wander.view(1, -1).repeat(in_channels, 1)
            augmented[i] = augmented[i] + wander
            
        elif aug_type == 'scale':
            # Amplitude scaling
            scale = random.uniform(0.7, 1.3)
            augmented[i] = augmented[i] * scale
            
        elif aug_type == 'shift':
            # Time shift
            shift = random.randint(-100, 100)
            if shift > 0:
                temp = augmented[i, :, :-shift].clone()  # Create temporary copy
                augmented[i, :, shift:] = temp
                augmented[i, :, :shift] = 0
            elif shift < 0:
                temp = augmented[i, :, -shift:].clone()  # Create temporary copy
                augmented[i, :, :shift] = temp
                augmented[i, :, shift:] = 0
                
    return augmented

# --- MOCO MODEL COMPONENTS ---
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
        self.conv_layers = nn.Sequential(
            self._make_conv_block(in_channels, base_channels),           # => base_channels
            self._make_conv_block(base_channels, base_channels * 2),       # => base_channels*2
            self._make_conv_block(base_channels * 2, base_channels * 4),   # => base_channels*4
            self._make_conv_block(base_channels * 4, base_channels * 8)    # => base_channels*8
        )
        self.fc = nn.Linear(base_channels * 8, embedding_dim)
    def _make_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=12),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)

        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

# MoCo1D: MoCo adapted for 1D signals
class MoCo1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, emb_dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = ImprovedECGEncoder(in_channels=in_channels, base_channels=base_channels, embedding_dim=emb_dim)
        self.encoder_k = ImprovedECGEncoder(in_channels=in_channels, base_channels=base_channels, embedding_dim=emb_dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", torch.randn(emb_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels



# --- MOCO TRAINING FUNCTIONS ---
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

def create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True )

def train_moco_1d(model, ecg_tensor, epochs, batch_size, device='cuda'):
    dataloader = create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.encoder_q.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))
        end = time.time()
        total_loss = 0.0
        num_steps = 0
        for i, (batch_data,) in enumerate(dataloader):
            data_time.update(time.time() - end)
            batch_data = batch_data.to(device)
            im_q = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)
            im_k = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            losses.update(loss.item(), batch_data.size(0))
            top1.update(acc1[0], batch_data.size(0))
            top5.update(acc5[0], batch_data.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_steps += 1
            batch_time.update(time.time() - end)
            end = time.time()
            log_rows.append({
                "epoch": epoch + 1,
                "batch": i,
                "time": batch_time.val,
                "data": data_time.val,
                "loss": losses.val,
                "acc1": top1.val.item(),
                "acc5": top5.val.item()
            })

            if i % 10 == 0:
                progress.display(i)
        scheduler.step()
        avg_loss = total_loss / num_steps
        wandb.log({"epoch": epoch+1, "moco_loss": avg_loss, "moco_acc@1": top1.avg, "moco_acc@5": top5.avg})
        # print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
        # print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
        # Show t-SNE plot every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Extract embeddings using the current encoder
             # Import here or at the top of your file
            model.eval()
            with torch.no_grad():
                embeddings = evaluate_moco(model, ecg_tensor, batch_size=batch_size, device=device)

            # Convert embeddings to NumPy array (they are already on CPU from evaluate_moco)
            embeddings_np = embeddings.numpy()
            # Ensure embeddings are in numpy format
            # embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

            # drop zero‐variance dims
            variances = embeddings_np.var(axis=0)
            embeddings_np = embeddings_np[:, variances > 0.0]
            # Standardize the embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_np)

            # Run UMAP
            umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            umap_2d = umap_model.fit_transform(embeddings_scaled)

            # KMeans clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans_labels = kmeans.fit_predict(embeddings_scaled)

            # DBSCAN clustering
            dbscan = DBSCAN(eps=1.5, min_samples=10)
            dbscan_labels = dbscan.fit_predict(embeddings_scaled)

            # Create a DataFrame for plotting
            df_umap = pd.DataFrame({
                "UMAP1": umap_2d[:, 0],
                "UMAP2": umap_2d[:, 1],
                "KMeans": kmeans_labels,
                "DBSCAN": dbscan_labels
            })

            # Plot UMAP with KMeans
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="KMeans", palette="tab10", s=15)
            plt.title("UMAP + KMeans Clustering")
            plt.tight_layout()
            plt.savefig("umap_kmeans.png")
            wandb.log({"UMAP_KMeans": wandb.Image("umap_kmeans.png")})
            plt.close()

            # Plot UMAP with DBSCAN
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="DBSCAN", palette="tab20", s=15)
            plt.title("UMAP + DBSCAN Clustering")
            plt.tight_layout()
            plt.savefig("umap_dbscan.png")
            wandb.log({"UMAP_DBSCAN": wandb.Image("umap_dbscan.png")})
            plt.close()
            # Close the figure after logging to free up memory
            model.train()
    log_path = f"moco_logs_epoch_{epoch+1}.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Batch-wise metrics saved to: {log_path}")
    torch.save(model.state_dict(), 'improved_moco_ecg_model.pth')
    print("MoCo model saved to 'improved_moco_ecg_model.pth'.")

def evaluate_moco(model, ecg_tensor, batch_size=32, device='cuda'):
    model.eval()
    dataloader = create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(device)
            emb = model.encoder_q(batch_data)
            all_embeddings.append(emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

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
def train_classifier_with_comparison(
                model,
                train_loader,
                val_loader,
                device="cpu",
                epochs=40,
                lr=1e-3,
                validation_epoch=10,
                prev_val_accuracy=None,
                improvement_epsilon=0.0  # Set a margin if needed
            ):
    # Add class weights to the loss function
    from sklearn.utils.class_weight import compute_class_weight
    
    # 1) Extract labels as plain Python ints, then into a NumPy array
    y_train = np.array([int(label) for _, label in train_loader.dataset])

    # 2) Call compute_class_weight with keyword-only args
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # Better than vanilla Adam
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    # optimizer = optim.Adam(model.parameters(), lr=lr)
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
            scheduler.step()  # Update learning rate

            train_loss += loss.item() * ecg_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
           # Log training metrics for this epoch
        wandb.log({"classifier_epoch": epoch+1, "train_loss": train_epoch_loss, "train_acc": train_epoch_acc})

        # Validate at the designated epoch
        # Validate every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            val_metrics = evaluate_metrics(model, val_loader, device=device)
            current_val_accuracy = val_metrics['accuracy']
            wandb.log({
                "val_acc_20": val_metrics['accuracy'],
                "val_prec_20": val_metrics['precision'],
                "val_recall_20": val_metrics['recall'],
                "val_f1_20": val_metrics['f1_score'],
                "classifier_epoch": epoch + 1
            })
            # print(f"[Epoch {epoch+1}] Validation → Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            break  # <== break training early if you're only checking once

    if current_val_accuracy is None:
        # Run validation at end if it wasn't done yet
        model.eval()
        val_metrics = evaluate_metrics(model, val_loader, device=device)
        current_val_accuracy = val_metrics['accuracy']

    return True, current_val_accuracy

def train_classifier(model, train_loader, val_loader=None, device='cpu', epochs=40, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
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
    def __init__(self, csv_path, root_dir, use_lr=True, fold_type='train', transform=None):

        self.root_dir = root_dir
        self.use_lr = use_lr
        # print(self.use_lr)
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        if fold_type == 'train':
            self.fold_mask = self.df['strat_fold'].between(1, 8)
        elif fold_type == 'val':
            self.fold_mask = (self.df['strat_fold'] == 9)
        elif fold_type == 'test':
            self.fold_mask = (self.df['strat_fold'] == 10)
            
        self.df = self.df[self.fold_mask].reset_index(drop=True)
        # Ensure the column exists in the CSV.
        if 'strat_fold' not in self.df.columns:
            raise ValueError("CSV file does not contain 'strat_fold' column for fold filtering.")
        # self.df = self.df[self.df['strat_fold'].isin(folds)].reset_index(drop=True)


        self._validate_columns()
        # from scp_statements.csv import diagnostic_superclass
        # self.df['label'] = self.df['scp_codes'].apply(
        #     lambda x: [diagnostic_superclass.get(code, 0) for code in x.keys()]
        # )
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
        # ecg = signals.T.astype(np.float32)  # (12, time)
        ecg = signals.astype(np.float32).T 
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



# --- MAIN PIPELINE ---
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ======================
    # Part A: MoCo Pre-training on Unlabeled Data
    # ======================
    # Load unlabeled data segments using the helper function.
    unlabeled_csv = '../data/train_unlabeled.csv'
    base_dir = '../data'
    wandb.init(project="moco_ecg-new_umap_100", config={
        "moco_epochs": 100,
        "moco_batch_size": 32,
        "classifier_epochs": 40,
        "initial_lr": 1e-3,
        "segment_length": 1000,
        "wavelet": "db4",
        "lead_index": 0,
        
    }, reinit=True)
    all_segments_df, skipped = load_and_segment_signals_1d_no_folder_limit(
        csv_file=unlabeled_csv,
        base_dir=base_dir,
        filename_col='filename_hr',
        max_files=5000,
        segment_length=1000,
        fs=125.0,
        do_wavelet=True,
        wavelet='db4',
        level=2,
        mode='soft',
        lead_index=0,
        random_files=False
    )
    print("Loaded segments shape:", all_segments_df.shape)
    print("Number of skipped files:", len(skipped))
    # Group by record_id and convert each segment to shape (1, 1000)
    all_ecgs = []
    for rid, group_df in all_segments_df.groupby('record_id'):
        ecg_1d = group_df["ecg_1d"].values  # (1000,)
        all_ecgs.append(ecg_1d)
    all_ecgs = np.array(all_ecgs)  # [N, 1000]
    all_ecgs = all_ecgs[:, np.newaxis, :]  # [N, 1, 1000]
    ecg_tensor = torch.from_numpy(all_ecgs).float().to(device)
    print("Unlabeled ecg_tensor shape:", ecg_tensor.shape)

    moco_model = MoCo1D(in_channels=1, base_channels=64, emb_dim=128, K=65536, m=0.999, T=0.07)
    train_moco_1d(moco_model, ecg_tensor, epochs=100, batch_size=32, device=device)
    wandb.finish()
    
    print("MoCo pre-training complete.")
   
   # -------------------------------
    # Part B: t-SNE Analysis
    # -------------------------------

    wandb.init(project="moco_ecg-new_umap_100", name="moco_tsne_analysis", reinit=True)
    moco_model.eval()
    with torch.no_grad():
        dl = create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=False)
        embs = torch.cat([moco_model.encoder_q(b.to(device)).cpu() for (b,) in dl], dim=0)
    tsne = TSNE(n_components=2, random_state=42)
    emb2d = tsne.fit_transform(embs.numpy())

    plt.figure(figsize=(8,6))
    plt.scatter(emb2d[:,0], emb2d[:,1], s=10)
    plt.title("t-SNE of ECG embeddings")
    wandb.log({"final_tsne": wandb.Image(plt.gcf())})
    plt.close()
    wandb.finish()
   
    # ======================
    # Part B: Supervised Fine-tuning on Labeled Data
    # ======================
    # Load labeled dataset using PTBXLDataset
    labeled_csv = "../data/ptbxl_database.csv"
    labeled_root = "../data/"
    # Use official splits:
    train_ds = PTBXLDataset(csv_path=labeled_csv, root_dir=labeled_root,
                             use_lr=False, fold_type='train') # Folds 1-8
    val_dataset   = PTBXLDataset(csv_path=labeled_csv, root_dir=labeled_root,
                             use_lr=False, fold_type='val') # Fold 9
    test_ds  = PTBXLDataset(csv_path=labeled_csv, root_dir=labeled_root,
                             use_lr=False, fold_type='test')  # Fold 10

     # Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_folds    = [1,2,3,4,5,6,7,8,9,10]
    min_lr       = 1e-6
    initial_lr   = 1e-3
    prev_val_acc = -1.0
    fold_results = []

    for test_fold in all_folds:
        print(f"\n--- Fold {test_fold} of {all_folds} ---")
       # stratified subsample of the train split (size 5k)
        labels = train_ds.df['label']
        idxs, _ = train_test_split(
            np.arange(len(train_ds)),
            train_size=5000,
            stratify=labels,
            random_state=42
        )
        sub_train = Subset(train_ds, idxs)

        # build loaders
        train_loader, val_loader = create_dataloaders(sub_train, batch_size=16)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        wandb.init(
            project="moco_ecg-new_umap_100",
            name=f"fine_tune_fold{test_fold}",
            config={"lr": initial_lr, "test_fold": test_fold},
            reinit=True
        )

        train_dataset = PTBXLDataset(
            csv_path=labeled_csv,
            root_dir=labeled_root,
            use_lr=False,          # adjust as needed
            fold_type='train',     # Training folds: 9 folds
            transform=None         # Any transformation you need
        )
        val_dataset = PTBXLDataset(
            csv_path=labeled_csv,
            root_dir=labeled_root,
            use_lr=False,
            fold_type='val'  # Uses fold 9
        )

        
        lr = initial_lr
        while True:
            # Instantiate a new classifier (fresh initialization for each try)
            classifier = ECGClassifier(encoder=moco_model.encoder_q, emb_dim=128, num_classes=2)
            print(f"\nTraining classifier for fold {test_fold} with learning rate {lr} ...")
            training_successful, current_val_acc = train_classifier_with_comparison(
                model=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=40,
                lr=lr,
                validation_epoch=10,
                prev_val_accuracy=prev_val_acc,
                improvement_epsilon=0.0  # Set a margin if needed
            )
            if current_val_acc > prev_val_acc:
                prev_val_acc = current_val_acc
                print("Training improved compared to previous evaluation. Proceeding with this learning rate.")
                break
            else:
                lr = lr * 0.5
                print(f"Validation performance did not improve (prev: {prev_val_acc:.4f}, current: {current_val_acc:.4f}).")
                print(f"Retrying training with adjusted learning rate: {lr}")
                if lr < min_lr:
                    print(f"Learning rate {lr} has dropped below the minimum threshold {min_lr}. Stopping further reduction.")
                    break
        
        val_metrics = evaluate_metrics(classifier, val_loader, device=device)
        fold_results.append(val_metrics)
        print(f"Fold {test_fold} val acc: {val_metrics['accuracy']:.4f}")

        print("Final Fine-Tuning Results on Validation Set:")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1-score:  {val_metrics['f1_score']:.4f}")
        # Optionally, plot the confusion matrix for this fold

        

        def plot_confusion_matrix_from_arrays(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Abnormal (1)'])
            disp.plot(ax=ax, cmap='Blues')
            plt.title("Confusion Matrix", fontsize=15, pad=20)
            wandb.log({"confusion_matrix": wandb.Image(fig)})
            plt.close(fig)
        plot_confusion_matrix_from_arrays(val_metrics['ground_truth'], val_metrics['predictions'])

 
        fold_results.append((test_fold, val_metrics))

        # Log interim results for this fold to wandb
        wandb.log({
            "test_fold": test_fold,
            "fold_accuracy": val_metrics['accuracy'],
            "fold_precision": val_metrics['precision'],
            "fold_recall": val_metrics['recall'],
            "fold_f1": val_metrics['f1_score']
        })
        # Print interim summary
        acc_list  = [res[1]['accuracy']  for res in fold_results]
        prec_list = [res[1]['precision'] for res in fold_results]
        rec_list  = [res[1]['recall']    for res in fold_results]
        f1_list   = [res[1]['f1_score']  for res in fold_results]
        print(f"Interim Mean Accuracy:  {np.mean(acc_list):.4f}")
        print(f"Interim Mean Precision: {np.mean(prec_list):.4f}")
        print(f"Interim Mean Recall:    {np.mean(rec_list):.4f}")
        print(f"Interim Mean F1-score:  {np.mean(f1_list):.4f}")

        # Final summary across folds
        acc_list  = [res[1]['accuracy']  for res in fold_results]
        prec_list = [res[1]['precision'] for res in fold_results]
        rec_list  = [res[1]['recall']    for res in fold_results]
        f1_list   = [res[1]['f1_score']  for res in fold_results]
        mean_acc  = np.mean(acc_list)
        mean_prec = np.mean(prec_list)
        mean_rec  = np.mean(rec_list)
        mean_f1   = np.mean(f1_list)
        print("\n========== 5-Fold CV Final Summary ==========")
        print(f"Accuracy:  {mean_acc:.4f}")
        print(f"Precision: {mean_prec:.4f}")
        print(f"Recall:    {mean_rec:.4f}")
        print(f"F1-score:  {mean_f1:.4f}")

        # Finish the wandb run
        wandb.finish()
     