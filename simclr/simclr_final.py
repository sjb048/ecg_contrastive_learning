import os, random, ast, gc, logging, argparse
import numpy as np
import wfdb, pywt, wandb

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
# import ecg_signals_simclr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from simclr_umap import (
    log_confusion_matrix,          # keep this – rename our own function later
    log_umap_projections,
    log_custom_umap,
    log_tsne_projections
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_rows = []  # For saving to CSV


# --- 1D TRANSFORMS for SimCLR augmentations ---
class RandomCrop1D:
    def __init__(self, crop_size): 
        self.crop_size = crop_size
    def __call__(self, signal):
        if signal.size(-1) <= self.crop_size:
            return signal
        start = random.randint(0, signal.size(-1) - self.crop_size)
        return signal[..., start:start+self.crop_size]

class RandomGrayscale1D:
    def __init__(self, p=0.2): 
        self.p = p
    def __call__(self, signal):
        if random.random() < self.p and signal.size(0) > 1:
            mean_signal = signal.mean(dim=0, keepdim=True)  # Shape: [1, L]
            return mean_signal.repeat(signal.size(0), 1)
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

class RandomBaselineWander1D:
    def __init__(self, amplitude=0.1, frequency=0.5):
        self.amplitude = amplitude
        self.frequency = frequency
    
    def __call__(self, signal):
        t = torch.arange(signal.size(-1), dtype=torch.float32) / signal.size(-1)
        wander = self.amplitude * torch.sin(2 * np.pi * self.frequency * t)
        wander = wander.view(1, -1).expand_as(signal)
        return signal + wander

# --- SimCLR-Style Augmentation Dataset ---
class AugmentationSimCLR(Dataset):
    def __init__(self, signals, crop_size=1000):
        self.signals = signals
        for i in range(min(5, len(signals))):
            print(f"Sample {i} shape: {signals[i].shape}")
        # Define three different pipelines
        self.pipes = [
            [   # pipeline‑1
                RandomScaling1D(),
                RandomAddNoise1D(),
                Normalize1D(mean=0.5, std=0.5)
            ],
            [   # pipeline‑2
                RandomGrayscale1D(),
                RandomBaselineWander1D(amplitude=0.1, frequency=0.5),
                RandomAddNoise1D(),
                Normalize1D(mean=0.5, std=0.5)
            ],
            [  # pipeline-3
                RandomTimeShift1D(max_shift=10),
                RandomFlip1D(),
                Normalize1D(mean=0.5, std=0.5)
            ]
        
        ]
    def __len__(self): 
        return len(self.signals)

    @staticmethod
    def _apply(transforms, signal):
        for t in transforms:
            signal = t(signal)
        return signal

    def __getitem__(self, idx):
        x = self.signals[idx]                    # shape [1, L]
        pipe_indices = random.sample(range(len(self.pipes)), 2)
        view1 = self._apply(self.pipes[pipe_indices[0]], x.clone())
        view2 = self._apply(self.pipes[pipe_indices[1]], x.clone())
        return view1, view2
# Encoder + SimCLR model
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
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=3),
            nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=15, stride=2, padding=3),
            nn.BatchNorm1d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=15, stride=2, padding=3),
            nn.BatchNorm1d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=15, stride=2, padding=3),
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
            nn.Linear(emb_dim, emb_dim*2), nn.ReLU(),
            nn.Linear(emb_dim*2, proj_dim)
        )

    def forward(self, x): return F.normalize(self.net(x), dim=1)


def create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True )

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
    def __init__(self, model, optimizer, scheduler, args, 
                 full_ecg_tensor: torch.Tensor,        # ➊ NEW
                 viz_interval: int = 10,               # ➋ NEW
                 viz_methods: list[str] = ["umap"]):
        self.model = model 
        self.opt = optimizer
        self.sched = scheduler

        self.full_tensor  = full_ecg_tensor      # shape [N,1,1000]  (on **CPU**)
        self.viz_interval = viz_interval
        self.viz_methods  = viz_methods

        self.args = args
        self.writer = SummaryWriter()
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        # self.scaler = GradScaler(enabled=args.fp16_precision)
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16_precision)


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
        accumulation_steps = 4  # Simulate batch size of 64 * 4 = 256
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            accumulated_loss = 0
            for batch_idx, batch in enumerate(loader):
                # print(f"Batch {batch_idx} structure: {len(batch)} items")  # Debug line
                if len(batch) == 2:
                    xi, xj = batch
                    print(f"xi shape: {xi.shape}, xj shape: {xj.shape}")
                else:
                    print(f"Unexpected batch structure: {batch}")
                    continue
                xi, xj = xi.to(self.args.device), xj.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    hi, zi = self.model(xi)
                    hj, zj = self.model(xj)
                    if torch.isnan(zi).any() or torch.isnan(zj).any():
                        wandb.log({"nan_in_embeddings": 1, "epoch": epoch+1})
                        print(f"NaN detected in embeddings at epoch {epoch+1}. Skipping batch.")
                        continue

                    # loss = self.info_nce_loss(zi, zj)
                    logits, labels = self.info_nce_loss(zi, zj)
                    loss = self.criterion(logits, labels) 
                    accumulated_loss += loss

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                        self.opt.zero_grad()
                        self.scaler.scale(accumulated_loss).backward()
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        total_loss += accumulated_loss.item() * accumulation_steps
                        accumulated_loss = 0
                    
               
            self.sched.step()
            avg_loss = total_loss / len(loader)
            wandb.log({"epoch": epoch+1, "simclr_loss": avg_loss})
            # ❶ ----------------  UMAP / t‑SNE every viz_interval  ----------------
            if (epoch + 1) % self.viz_interval == 0:
                print(f"[viz] epoch {epoch+1}")
                visualize_embeddings(self.model,
                                     self.full_tensor,          # on **CPU**
                                     epoch + 1,
                                     batch_size   = 256,        # (<‑‑ any reasonable)
                                     device       = self.args.device,
                                     viz_methods  = self.viz_methods)
            # --------------------------------------------------------------------

            logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

 

def evaluate_simclr(model, ecg_tensor, batch_size=32, device='cuda'):
    model.eval()
    ds = AugmentationSimCLR(ecg_tensor, crop_size=ecg_tensor.shape[-1])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for xi, xj in loader:
            xi = xi.to(device)
            emb, _ = model(xi)
            all_embeddings.append(emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings
            
def visualize_embeddings(model, ecg_tensor, epoch, batch_size, device, viz_methods):
    
    model.eval()
    # Extract embeddings
    embeddings = evaluate_simclr(model, ecg_tensor, batch_size, device)
    
    # Convert to numpy
    emb_np = embeddings.numpy()
    
    # Check for NaNs - critical before visualization
    if np.isnan(emb_np).any():
        print(f"NaN detected in embeddings at epoch {epoch}! Skipping visualization.")
        return
    
    # Filter zero-variance dimensions with fallback
    var = emb_np.var(axis=0)
    if np.any(var > 0):
        emb_filtered = emb_np[:, var > 0]
    else:
        emb_filtered = emb_np + np.random.normal(0, 1e-6, emb_np.shape)
    
    # Standardize using RobustScaler (handles outliers better than StandardScaler)
    emb_scaled = RobustScaler().fit_transform(emb_filtered)
    # Add clustering evaluation
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(emb_scaled)
    if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
        sil_score = silhouette_score(emb_scaled, labels)
        wandb.log({"epoch": epoch, "silhouette_score": sil_score})
        print(f"Epoch {epoch} Silhouette Score: {sil_score:.4f}")
    
    try:
        for method in viz_methods:
            if method.lower() == 'umap':
                # Call existing utility function if it exists
                log_umap_projections(model, 
                                     create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=False), 
                                     device)
                
                # Optional: Also log the custom UMAP visualization if desired
                log_custom_umap(emb_scaled, epoch)
                
            elif method.lower() == 'tsne':
                log_tsne_projections(emb_scaled, epoch)
                
            # elif method.lower() == 'pca':
            #     log_pca_projections(emb_scaled, epoch)
                
    except Exception as e:
        print(f"Error in embedding visualization: {str(e)}")
    # Add garbage collection after intensive memory operations
    gc.collect()
    torch.cuda.empty_cache()
    print("train will be called")
    # Switch back to training mode
    model.train()


# --- DATA LOADING FOR UNLABELED DATA (for MoCo Pre-training) ---

def load_and_segment_signals_1d_no_folder_limit(
        csv_file, 
        base_dir, 
        filename_col='filename_hr', 
        max_files=5000, 
        segment_length=1000):
    
    df = pd.read_csv(csv_file).iloc[:max_files]
    loaded_segments = []
    for _, row in df.iterrows():
        try:
            full_path = os.path.join(base_dir, row[filename_col])
            record = wfdb.rdrecord(full_path)
            sig = record.p_signal[:segment_length, :]  # Take all 12 leads
            loaded_segments.append(sig)
        except:
            continue
    
    ecg_array = np.stack(loaded_segments, axis=0)  # Shape: (N, segment_length, 12)
    return torch.tensor(ecg_array.permute(0, 2, 1), dtype=torch.float32)  # Shape: (N, 12, segment_length)


def load_ecg_tensor_from_npy(npy_path='ecg_signals_simclr_12lead.npy'):
    ecg_array = np.load(npy_path)  # shape could be (N, 1000) or (N, 1000, 12)
    print(f"Loaded array shape: {ecg_array.shape}")
    
    # Check the number of dimensions
    if len(ecg_array.shape) == 2:
        # If 2D, assume single lead and add a channel dimension (N, 1000) -> (N, 1, 1000)
        ecg_array = ecg_array[:, None, :]  # shape: (N, 1, 1000)
        print(f"Converted to shape: {ecg_array.shape} (single lead data)")
    elif len(ecg_array.shape) == 3:
        # If 3D, assume it's already (N, segment_length, 12) or similar
        print(f"Using existing 3D shape: {ecg_array.shape}")
    else:
        raise ValueError(f"Unexpected shape of loaded array: {ecg_array.shape}")
    
    # Convert to tensor
    ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32)
    
    # If the channel dimension is not 12, warn the user (since model expects 12 channels)
    if ecg_tensor.shape[1] != 12:
        print(f"Warning: Data has {ecg_tensor.shape[1]} channels, but model expects 12 channels. Consider regenerating data with all 12 leads.")
        
    # If data is (N, segment_length, 12), permute to (N, 12, segment_length)
    if len(ecg_array.shape) == 3 and ecg_array.shape[-1] == 12:
        ecg_tensor = ecg_tensor.permute(0, 2, 1)  # shape: (N, 12, segment_length)
        print(f"Permuted to shape: {ecg_tensor.shape}")
    
    return ecg_tensor


# --- SUPERVISED CLASSIFIER USING PRE-TRAINED ENCODER ---
class ECGClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, num_classes=2,freeze_encoder=False):
        """
        Uses the pre-trained encoder to extract features from ECG,
        then applies a classification head.
        """
        super(ECGClassifier, self).__init__()
        self.encoder = encoder
        # Optionally freeze encoder parameters:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits


# --- LABELED DATASET FOR SUPERISED FINETUNING ---
class PTBXLDataset(Dataset):
    def __init__(self, csv_path, root_dir, use_lr=False, folds=[1,2,3,4,5]):
        self.root_dir = root_dir
        self.use_lr = use_lr
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['strat_fold'].isin(folds)].reset_index(drop=True)
        self.df['label'] = self.df['scp_codes'].apply(lambda s: 0 if "NORM" in ast.literal_eval(s) else 1)
        self.filename_col = 'filename_lr' if self.use_lr else 'filename_hr'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        path = os.path.join(self.root_dir, record[self.filename_col])
        h = wfdb.rdheader(path)          # <‑‑ zero‑cost header read
        if idx == 0:                     # only once
            print(f"[PTBXLDataset] sampling frequency reported by WFDB: {h.fs} Hz")
        try:
            signal, _ = wfdb.rdsamp(path)
        except:
            alt_path = path.replace('_lr', '').replace('_hr', '')
            signal, _ = wfdb.rdsamp(alt_path)

        # Transpose to shape (12, length)
        ecg = signal.T.astype(np.float32)
        return torch.tensor(ecg, dtype=torch.float32), torch.tensor(record['label'], dtype=torch.long)





def train_classifier(model, train_loader, val_loader=None, device='cpu',
                     metric_for_lr="f1_score",lr_factor=0.5,min_delta=1e-3, epochs=60, eval_every = 20,
                       lr=1e-3):
    
    # -------------------------------------------------------
    # 1.  Build a 1‑D numpy array of training labels
    # -------------------------------------------------------
    if hasattr(train_loader.dataset, "dataset"):          # e.g. torch.utils.data.Subset
        base = train_loader.dataset.dataset
        idxs = train_loader.dataset.indices
        y_train = base.df["label"].iloc[idxs].to_numpy(dtype=int)
    elif hasattr(train_loader.dataset, "df"):             # PTBXLDataset directly
        y_train = train_loader.dataset.df["label"].to_numpy(dtype=int)
    else:                                                 # fallback: iterate
        y_train = np.array([int(lbl) for _, lbl in train_loader.dataset])

    # full label set for PTB‑XL (binary); adapt if you have >2 classes
    classes_full = np.array([0, 1])

    # try to compute balanced weights; if only one class present, fall back to 1‑1
    try:
        w = compute_class_weight(class_weight="balanced",
                                 classes=classes_full,
                                 y=y_train)
    except ValueError:
        w = np.ones_like(classes_full, dtype=float)

    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("class weights →", class_weights.tolist())

    # -------------------------------------------------------
    # 2.  Loss, optimiser, housekeeping
    # -------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_best = -float("inf")
    model.to(device)
    step_tag   = lambda ep: f"ep{ep:03d}"

    for epoch in range(epochs):
        total_loss= 0
        model.train()
        train_loss = 0
        total_loss, total_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss   += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = total_correct / len(train_loader.dataset)

        print(f"from train classifier Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                   "train_acc":  train_acc})

        # --------------- validation ---------------
        if val_loader and (epoch+1) %  eval_every == 0: 
            val_metrics = evaluate_metrics(model, val_loader, device)
            # wandb.log({f"val_{k}": v for k, v in val_metrics.items()} | {"epoch": epoch+1})
            wandb.log({
                "val_acc_20": val_metrics['accuracy'],
                "val_prec_20": val_metrics['precision'],
                "val_recall_20": val_metrics['recall'],
                "val_f1_20": val_metrics['f1_score'],
                "classifier_epoch": epoch + 1
            })
            log_confusion_matrix(val_metrics["ground_truth"], 
                               val_metrics["predictions"],
                               step_tag=step_tag(epoch+1),
                               show_local=False)
            # LR scheduler on chosen metric
            current = val_metrics[metric_for_lr]
            if current < scheduler_best + min_delta:
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_factor
                print(f"LR ↓  by {lr_factor}, now {optimizer.param_groups[0]['lr']:.2e}")
            else:
                scheduler_best = current

    return scheduler_best
       

def official_evaluation(
    encoder, 
    labeled_csv, 
    labeled_root, 
    device,
    test_fold=[10],  # Official test fold
    # num_folds=5,
    batch_size=64,
    epochs=60,
    eval_every=20
):
   
    """Optimized 5-fold cross-validation with PTB-XL best practices."""
    # Official PTB-XL splits
    TRAIN_FOLDS = list(range(1, 9))  # Folds 1-8
    VAL_FOLD = [9]
    TEST_FOLD = test_fold

   
    # =================================================================
    # 1. Prepare Official Splits
    # =================================================================
    # Training data (folds 1-8)
    # Create full dataset once
    train_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=TRAIN_FOLDS
    )
    # Validation data (fold 9)
    val_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=[9]  # Official validation fold
    )
    # Test data (fold 10)
    test_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=TEST_FOLD
    )
    # =================================================================
    # 2. Create DataLoaders with Optimal Parameters
    # =================================================================
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2
    )

    # After dataset creation
    wandb.log({
        "train_class_dist": wandb.Histogram(train_ds.df['label']),
        "test_class_dist": wandb.Histogram(test_ds.df['label'])
    })

     # =================================================================
    # 3. Train Classifier with Validation Monitoring
    # =================================================================
    classifier = ECGClassifier(encoder, emb_dim=128, num_classes=2).to(device)
    print("evaluation classifier")
    best_metric = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=1e-4,
        eval_every=eval_every
    )
    print("evaluation classifier")
    # =================================================================
    # 4. Final Evaluation on Official Test Set
    # =================================================================
    classifier.eval()
    test_metrics = evaluate_metrics(classifier, test_loader, device)
    
    # Cleanup GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    return test_metrics
def evaluate_metrics(model, dataloader, device='cpu'):
    
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



# --- MAIN PIPELINE ---
if __name__ == '__main__':
     # Hardware setup
    
    # Phase 1: MoCo Pre-training
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='../data/ptbxl_database.csv')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--temperature',  type=float, default=0.5) 
    parser.add_argument('--fp16_precision', action='store_true') 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights during fine-tuning')
    args = parser.parse_args()

    wandb.init(project="simclr_runimp", name="pretrain", config={
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "in_channels": 12,
        "embedding_dim": 128
    })
    
    
    ecg_tensor = load_ecg_tensor_from_npy('ecg_signals_simclr_12lead.npy')
    print("tensor shape:", ecg_tensor.shape)
    if ecg_tensor.shape[1] != 12:
        raise ValueError(f"Pre-training data has {ecg_tensor.shape[1]} channels, but encoder expects 12 channels.")

    # (e) SimCLR dataset / dataloader
    ds      = AugmentationSimCLR(ecg_tensor, crop_size=1000)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    for batch_idx, batch in enumerate(loader):
        # print(f"Batch {batch_idx} structure: {len(batch)} items, shapes: {[x.shape for x in batch]}")
        break  # Check only the first batch

    
    # (f) model + trainer
    encoder   = ImprovedECGEncoder(in_channels=12)
    model    = SimCLR1D(encoder).to(args.device)

    optimizer  = optim.Adam(model.parameters(), lr=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    trainer    = SimCLRTrainer(model, optimizer, scheduler, args,
                        full_ecg_tensor = ecg_tensor,   # ➊ pass CPU tensor
                        viz_interval    = 20,           # ➋ every 20 epochs
                        viz_methods     = ["umap"])
    trainer.train(loader)

    torch.save(encoder.state_dict(), "simclr_encoder_pretrained.pth")
    wandb.finish()

    wandb.init(project='simclr_runimp', name='finetune')

    # Load dataset
    train_ds = PTBXLDataset(args.csv, args.data_dir, use_lr=False, folds=list(range(1,9)))
    val_ds = PTBXLDataset(args.csv, args.data_dir, use_lr=False, folds=[9])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Load encoder
    encoder = ImprovedECGEncoder(in_channels=12)
    encoder.load_state_dict(torch.load("simclr_encoder_pretrained.pth", map_location=args.device))

    # Check input shape consistency using a sample from the dataset
    sample_ecg, _ = train_ds[0]  # Get a sample input from the dataset
    if sample_ecg.shape[0] != 12:
        raise ValueError(f"Fine-tuning data has {sample_ecg.shape[0]} channels, but encoder expects 12 channels.")
    # Fine-tune
    model = ECGClassifier(encoder, emb_dim=128, num_classes=2, freeze_encoder=args.freeze_encoder)
    scheduler_best = train_classifier(model, train_loader, val_loader, args.device, epochs=60, lr=args.lr)
    # Log final metrics
    wandb.log({"best_validation_f1": scheduler_best})
    wandb.finish()

    # Cleanup resources
    gc.collect()
    torch.cuda.empty_cache()


     # Phase 3: Evaluation
    wandb.init(project="simclr_runimp", name="evaluation")
    final_metrics = official_evaluation(
        encoder= encoder,
        labeled_csv='../data/ptbxl_database.csv',
        labeled_root='../data',
        device=args.device,
        test_fold=[10],  # Official test fold
        batch_size=64,
        epochs=60
    )
    
    wandb.log(final_metrics)
    # ... eval code ...
    wandb.finish()